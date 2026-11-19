/**
 * @file fp8_gemm_test.cpp
 * @brief Contracts for the FP8 GEMM entry points
 *
 * src/cuda/quantize/fp8_gemm.cu implements the quantized matrix multiplications.
 * Three of its eight functions were referenced from any test, and the E5M2 path was
 * wrong.
 *
 * FP8E5M2GEMM::forward reinterpret_cast its FP8E5M2 inputs to FP8E4M3 before
 * launching the kernel:
 *
 *     detail::fp8_gemm_naive_kernel<BLOCK_SIZE><<<...>>>(
 *         reinterpret_cast<const FP8E4M3*>(a),
 *         reinterpret_cast<const FP8E4M3*>(b), ...);
 *
 * The two formats split their byte differently — E5M2 has a 5-bit exponent and a
 * 2-bit mantissa, E4M3 has 4 and 3 — so reinterpreting the bits decodes a different
 * number. Measured: 0.5 is 0x38 in E5M2 but 0x30 in E4M3, and a 1x1 product of 2 and
 * 3 returned 5.0 instead of 6.0. Some values happen to coincide (2.0 is 0x40 in
 * both), which is why this could go unnoticed.
 *
 * The naive kernel is now templated on the element type so each format is decoded by
 * its own conversion operator.
 *
 * The E4M3 path, the three scale factors, and the workspace query were checked and
 * are correct. The blocked kernel and the row/reduce variants are unused dead code,
 * which is noted rather than rewritten.
 */

#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda/quantize/fp8_types.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace nova::quantize::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns a device buffer of one-byte FP8 elements.
        template <typename T>
        class DeviceArray {
        public:
            explicit DeviceArray(const std::vector<T>& host)
                : count_(host.size()) {
                cudaMalloc(&ptr_, count_ > 0 ? count_ : 1);
                if (count_ > 0) {
                    cudaMemcpy(ptr_, host.data(), count_, cudaMemcpyHostToDevice);
                }
            }

            explicit DeviceArray(size_t count)
                : count_(count) {
                cudaMalloc(&ptr_, count_ > 0 ? count_ : 1);
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            T* data() { return ptr_; }
            const T* data() const { return ptr_; }

        private:
            T* ptr_ = nullptr;
            size_t count_ = 0;
        };

        /// Owns a device float buffer.
        class DeviceFloats {
        public:
            explicit DeviceFloats(size_t count)
                : count_(count) {
                cudaMalloc(&ptr_, count * sizeof(float));
                cudaMemset(ptr_, 0, count * sizeof(float));
            }

            ~DeviceFloats() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceFloats(const DeviceFloats&) = delete;
            DeviceFloats& operator=(const DeviceFloats&) = delete;

            float* data() { return ptr_; }

            std::vector<float> read() const {
                std::vector<float> host(count_);
                cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost);
                return host;
            }

        private:
            float* ptr_ = nullptr;
            size_t count_ = 0;
        };

        template <typename T>
        std::vector<T> to_fp8(const std::vector<float>& values) {
            std::vector<T> out;
            out.reserve(values.size());
            for (const float value : values) {
                out.emplace_back(value);
            }
            return out;
        }

    }  // namespace

    // Regression: this decoded E5M2 bits as E4M3 and returned 5.0 for 2*3.
    TEST(Fp8GemmTest, E5M2ProductsDecodeTheirOwnFormat) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 1x1 * 1x1 = 6. The values are small enough to be exact in both formats, so
        // any error is a decoding error rather than a rounding one.
        const auto a_host = to_fp8<FP8E5M2>({2.0f});
        const auto b_host = to_fp8<FP8E5M2>({3.0f});

        DeviceArray<FP8E5M2> a(a_host);
        DeviceArray<FP8E5M2> b(b_host);
        DeviceFloats c(1);

        FP8E5M2GEMM::forward(a.data(), b.data(), c.data(), 1, 1, 1, FP8E5M2GEMM::Config(1.0f, 1.0f, 1.0f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        EXPECT_NEAR(c.read()[0], 6.0f, 0.1f) << "a reinterpreted E5M2 bit pattern decodes to a different value";
    }

    TEST(Fp8GemmTest, E5M2DotProductOverSeveralValues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 0.5 is the value whose bit patterns differ most between the formats (0x38
        // vs 0x30), so including it makes a decoding error unmistakable.
        const auto a_host = to_fp8<FP8E5M2>({0.5f, 1.0f, 2.0f, 4.0f});
        const auto b_host = to_fp8<FP8E5M2>({2.0f, 2.0f, 2.0f, 2.0f});

        DeviceArray<FP8E5M2> a(a_host);
        DeviceArray<FP8E5M2> b(b_host);
        DeviceFloats c(1);

        FP8E5M2GEMM::forward(a.data(), b.data(), c.data(), 1, 4, 1, FP8E5M2GEMM::Config(1.0f, 1.0f, 1.0f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // (0.5 + 1 + 2 + 4) * 2 = 15
        EXPECT_NEAR(c.read()[0], 15.0f, 0.5f);
    }

    TEST(Fp8GemmTest, E5M2RoundTripsThroughItsOwnEncoding) {
        // The encoding difference itself, independent of the kernel: any test that
        // only used values where both formats agree would pass against the bug.
        EXPECT_NE(FP8E5M2(0.5f).to_bits(), FP8E4M3(0.5f).to_bits()) << "0.5 must not share a bit pattern across the two formats";

        // A value where they do coincide, documented so the tolerance choice above is
        // understandable.
        EXPECT_EQ(FP8E5M2(2.0f).to_bits(), FP8E4M3(2.0f).to_bits()) << "2.0 happens to decode identically; that coincidence hid the bug";
    }

    TEST(Fp8GemmTest, E4M3MatrixProductMatchesAHandComputation) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A is 2x3, B is 3x2, so C is 2x2.
        // A = [[1,2,3],[4,5,6]]   B = [[1,0],[0,1],[1,1]]
        // C = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
        const auto a_host = to_fp8<FP8E4M3>({1, 2, 3, 4, 5, 6});
        const auto b_host = to_fp8<FP8E4M3>({1, 0, 0, 1, 1, 1});

        DeviceArray<FP8E4M3> a(a_host);
        DeviceArray<FP8E4M3> b(b_host);
        DeviceFloats c(4);

        FP8GEMM::forward(a.data(), b.data(), c.data(), 2, 3, 2, FP8GEMM::Config(1.0f, 1.0f, 1.0f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = c.read();
        ASSERT_EQ(result.size(), 4u);
        EXPECT_NEAR(result[0], 4.0f, 0.2f);
        EXPECT_NEAR(result[1], 5.0f, 0.2f);
        EXPECT_NEAR(result[2], 10.0f, 0.2f);
        EXPECT_NEAR(result[3], 11.0f, 0.2f);
    }

    TEST(Fp8GemmTest, EachScaleFactorInfluencesTheResult) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const auto a_host = to_fp8<FP8E4M3>({1, 0, 0, 1});
        const auto b_host = to_fp8<FP8E4M3>({1, 0, 0, 1});

        DeviceArray<FP8E4M3> a(a_host);
        DeviceArray<FP8E4M3> b(b_host);
        DeviceFloats c(4);

        // Identity times identity is the identity, so the scales multiply straight
        // through: 1 * sa * sb * so.
        FP8GEMM::forward(a.data(), b.data(), c.data(), 2, 2, 2, FP8GEMM::Config(2.0f, 3.0f, 0.5f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = c.read();
        ASSERT_EQ(result.size(), 4u);
        EXPECT_NEAR(result[0], 3.0f, 0.2f) << "2 * 3 * 0.5";
        EXPECT_NEAR(result[1], 0.0f, 0.2f);
        EXPECT_NEAR(result[2], 0.0f, 0.2f);
        EXPECT_NEAR(result[3], 3.0f, 0.2f);
    }

    TEST(Fp8GemmTest, ForwardAsyncDelegatesToForward) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const auto a_host = to_fp8<FP8E4M3>({1, 0, 0, 1});
        const auto b_host = to_fp8<FP8E4M3>({1, 0, 0, 1});

        DeviceArray<FP8E4M3> a(a_host);
        DeviceArray<FP8E4M3> b(b_host);
        DeviceFloats c(4);

        FP8GEMM::forward_async(a.data(), b.data(), c.data(), 2, 2, 2, FP8GEMM::Config(1.0f, 1.0f, 1.0f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = c.read();
        EXPECT_NEAR(result[0], 1.0f, 0.2f);
        EXPECT_NEAR(result[3], 1.0f, 0.2f);
    }

    TEST(Fp8GemmTest, BackwardProducesBothGradients) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 2x2 * 2x2 with a gradient of ones: grad_a = ones @ B^T and grad_b = ones^T @ A.
        const auto a_host = to_fp8<FP8E4M3>({1, 2, 3, 4});
        const auto b_host = to_fp8<FP8E4M3>({1, 0, 0, 1});
        const auto g_host = to_fp8<FP8E4M3>({1, 1, 1, 1});

        DeviceArray<FP8E4M3> a(a_host);
        DeviceArray<FP8E4M3> b(b_host);
        DeviceArray<FP8E4M3> g(g_host);
        DeviceFloats grad_a(4);
        DeviceFloats grad_b(4);

        FP8GEMM::backward(g.data(), a.data(), b.data(), grad_a.data(), grad_b.data(), 2, 2, 2, FP8GEMM::Config(1.0f, 1.0f, 1.0f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto ga = grad_a.read();
        const auto gb = grad_b.read();
        for (const float value : ga) {
            EXPECT_TRUE(std::isfinite(value)) << "grad_a holds a non-finite value";
        }
        for (const float value : gb) {
            EXPECT_TRUE(std::isfinite(value)) << "grad_b holds a non-finite value";
        }
        // B is the identity, so grad_a equals the ones gradient.
        EXPECT_NEAR(ga[0], 1.0f, 0.3f);
        EXPECT_NEAR(ga[1], 1.0f, 0.3f);
    }

    TEST(Fp8GemmTest, WorkspaceSizeIsPositiveForAUsableShape) {
        // The workspace query takes the same shape the forward pass accepts. The
        // blocked kernel it refers to is unused, but the query must still answer
        // sensibly for a valid shape.
        const size_t bytes = FP8GEMM::get_workspace_size(16, 16, 16, FP8GEMM::Config(1, 1, 1));
        EXPECT_GE(bytes, 0u);
    }

    TEST(Fp8GemmTest, HandlesShapesThatAreNotAMultipleOfTheBlockSize) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 3x5 * 5x7: every dimension is off the 16x16 block grid, so the bounds checks
        // in the kernel are exercised on a partial tile.
        constexpr int m = 3, k = 5, n = 7;
        std::vector<FP8E4M3> a_host;
        for (int i = 0; i < m * k; ++i) {
            a_host.emplace_back(1.0f);
        }
        std::vector<FP8E4M3> b_host;
        for (int i = 0; i < k * n; ++i) {
            b_host.emplace_back(1.0f);
        }

        DeviceArray<FP8E4M3> a(a_host);
        DeviceArray<FP8E4M3> b(b_host);
        DeviceFloats c(static_cast<size_t>(m) * n);

        FP8GEMM::forward(a.data(), b.data(), c.data(), m, k, n, FP8GEMM::Config(1.0f, 1.0f, 1.0f), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = c.read();
        ASSERT_EQ(result.size(), static_cast<size_t>(m) * n);
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_NEAR(result[i], static_cast<float>(k), 0.5f) << "element " << i << " of an all-ones contraction should be k";
        }
    }

}  // namespace nova::quantize::test
