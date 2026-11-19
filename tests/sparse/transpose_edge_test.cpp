/**
 * @file transpose_edge_test.cpp
 * @brief Contracts for the matrix transpose entry points on empty extents
 *
 * src/matrix/ops.cu provides a plain and a tiled transpose. Both derived a grid from
 * the caller's extents without checking them:
 *
 *     dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE,
 *               (rows + TILE_SIZE - 1) / TILE_SIZE);
 *
 * A zero or negative extent makes one grid dimension zero, which CUDA rejects.
 * Measured before the fix:
 *
 *     transposeMatrix(rows=-4, cols=8) -> CUDA error: invalid configuration argument
 *     transposeMatrix(rows=0,  cols=8) -> the same
 *
 * The message names neither the parameter nor the cause, and transposing nothing is
 * a perfectly ordinary request. Both entry points now return without launching.
 *
 * The transpose index arithmetic was checked and is correct:
 * out[c * rows + r] = in[r * cols + c], for both the plain and the tiled kernel.
 */

#include <matrix/ops.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace transpose_edge_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns a device float array.
        class DeviceArray {
        public:
            explicit DeviceArray(const std::vector<float>& host)
                : count_(host.size()) {
                cudaMalloc(&ptr_, (count_ > 0 ? count_ : 1) * sizeof(float));
                if (count_ > 0) {
                    cudaMemcpy(ptr_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice);
                }
            }

            explicit DeviceArray(size_t count)
                : count_(count) {
                cudaMalloc(&ptr_, (count_ > 0 ? count_ : 1) * sizeof(float));
                cudaMemset(ptr_, 0, (count_ > 0 ? count_ : 1) * sizeof(float));
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            float* data() { return ptr_; }
            const float* data() const { return ptr_; }

            std::vector<float> read() const {
                std::vector<float> host(count_, 0.0f);
                if (count_ > 0) {
                    cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost);
                }
                return host;
            }

        private:
            float* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: these threw "invalid configuration argument".
    TEST(TransposeEdgeTest, EmptyExtentsAreNoOps) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();

        EXPECT_NO_THROW(transposeMatrix<float>(nullptr, nullptr, 0, 8));
        EXPECT_NO_THROW(transposeMatrix<float>(nullptr, nullptr, 8, 0));
        EXPECT_NO_THROW(transposeMatrix<float>(nullptr, nullptr, 0, 0));
        EXPECT_NO_THROW(transposeMatrix<float>(nullptr, nullptr, -4, 8));
        EXPECT_NO_THROW(transposeMatrix<float>(nullptr, nullptr, 8, -4));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "an empty transpose must not leave a launch error behind";
    }

    TEST(TransposeEdgeTest, TiledVariantAlsoHandlesEmptyExtents) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();

        EXPECT_NO_THROW(transposeMatrixTiled<float>(nullptr, nullptr, 0, 8));
        EXPECT_NO_THROW(transposeMatrixTiled<float>(nullptr, nullptr, 8, 0));
        EXPECT_NO_THROW(transposeMatrixTiled<float>(nullptr, nullptr, -2, -2));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    TEST(TransposeEdgeTest, PlainTransposeMatchesTheIndexArithmetic) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 2x3 becomes 3x2: out[c * rows + r] = in[r * cols + c].
        constexpr int rows = 2, cols = 3;
        const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        DeviceArray d_input(input);
        DeviceArray d_output(input.size());

        transposeMatrix<float>(d_input.data(), d_output.data(), rows, cols);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto output = d_output.read();
        ASSERT_EQ(output.size(), 6u);
        const std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(output[i], expected[i]) << "element " << i;
        }
    }

    TEST(TransposeEdgeTest, TiledTransposeMatchesThePlainOne) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A size that is not a multiple of the 16x16 tile exercises the partial tile.
        constexpr int rows = 5, cols = 7;
        std::vector<float> input(static_cast<size_t>(rows) * cols);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(i);
        }

        DeviceArray d_input(input);
        DeviceArray d_plain(input.size());
        DeviceArray d_tiled(input.size());

        transposeMatrix<float>(d_input.data(), d_plain.data(), rows, cols);
        transposeMatrixTiled<float>(d_input.data(), d_tiled.data(), rows, cols);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto plain = d_plain.read();
        const auto tiled = d_tiled.read();
        ASSERT_EQ(plain.size(), tiled.size());
        for (size_t i = 0; i < plain.size(); ++i) {
            EXPECT_FLOAT_EQ(tiled[i], plain[i]) << "the tiled kernel disagrees at element " << i;
        }
    }

    TEST(TransposeEdgeTest, SingleElementIsItsOwnTranspose) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        DeviceArray d_input(std::vector<float> {7.5f});
        DeviceArray d_output(1);

        transposeMatrix<float>(d_input.data(), d_output.data(), 1, 1);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        EXPECT_FLOAT_EQ(d_output.read()[0], 7.5f);
    }

    TEST(TransposeEdgeTest, SquareTransposeIsAnInvolution) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Transposing twice returns the original.
        constexpr int n = 4;
        std::vector<float> input(static_cast<size_t>(n) * n);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(i) * 0.5f;
        }

        DeviceArray d_input(input);
        DeviceArray d_first(input.size());
        DeviceArray d_second(input.size());

        transposeMatrix<float>(d_input.data(), d_first.data(), n, n);
        transposeMatrix<float>(d_first.data(), d_second.data(), n, n);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto twice = d_second.read();
        for (size_t i = 0; i < input.size(); ++i) {
            EXPECT_FLOAT_EQ(twice[i], input[i]) << "element " << i;
        }
    }

}  // namespace transpose_edge_test
