/**
 * @file softmax_kernel_test.cpp
 * @brief Device-correctness contracts for the softmax entry points
 *
 * tests/neural/softmax_test.cpp includes the softmax header but never calls into it:
 * its cases compute a reference distribution in host code and assert on that, so the
 * CUDA implementation was never executed by any test. That is how the defect below
 * survived.
 *
 * softmax() launched outer_dim blocks of 256 threads while its kernel writes one output
 * element per thread, with idx running over outer_dim * inner_dim. On a 64x1024 input
 * that is 16384 threads for 65536 elements, so three quarters of the output buffer was
 * never written and kept its previous contents. Measured before the fix: the row sums of
 * the result were off by up to 1.0, i.e. the output was not a distribution at all. After
 * the fix the worst deviation is 8.7e-07.
 *
 * The same measurement shows why softmax() is not the implementation to use for real
 * work: each thread walks the whole row to compute its own denominator, so a 64x1024
 * input takes 6.5 ms against softmax_stable's 0.037 ms. That is an algorithmic
 * property of this entry point, not a defect, and the cases below do not assert on
 * timing.
 */

#include <cuda/neural/softmax.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace cuda::neural::test {

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

            explicit DeviceArray(size_t count, float fill)
                : count_(count) {
                cudaMalloc(&ptr_, (count > 0 ? count : 1) * sizeof(float));
                std::vector<float> host(count, fill);
                cudaMemcpy(ptr_, host.data(), (count > 0 ? count : 1) * sizeof(float), cudaMemcpyHostToDevice);
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            float* data() { return ptr_; }

            std::vector<float> read() const {
                std::vector<float> host(count_, 0.0f);
                cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost);
                return host;
            }

        private:
            float* ptr_ = nullptr;
            size_t count_ = 0;
        };

        /// Runs an entry point into a buffer poisoned with a value no probability can equal, so
        /// an unwritten element is unmistakable.
        template <typename EntryPoint>
        std::vector<float> run_softmax(EntryPoint entry, const std::vector<float>& input, int outer, int inner) {
            DeviceArray d_input(input);
            DeviceArray d_output(input.size(), -12345.0f);
            entry(d_input.data(), d_output.data(), outer, inner, nullptr);
            cudaDeviceSynchronize();
            return d_output.read();
        }

    }  // namespace

    // Regression: three quarters of the output buffer was never written.
    TEST(SoftmaxKernelTest, EveryOutputElementIsWritten) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 64 rows of 1024, the shape that exposed the missing grid coverage.
        constexpr int outer = 64, inner = 1024;
        std::vector<float> input(static_cast<size_t>(outer) * inner);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((i * 7) % 23) * 0.3f;
        }

        const auto result = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, input, outer, inner);

        ASSERT_EQ(result.size(), input.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_NE(result[i], -12345.0f) << "element " << i << " was never written; the grid did not cover it";
        }
    }

    TEST(SoftmaxKernelTest, RowsSumToOne) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        constexpr int outer = 64, inner = 1024;
        std::vector<float> input(static_cast<size_t>(outer) * inner);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((i * 11) % 31) * 0.2f - 3.0f;
        }

        const auto result = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, input, outer, inner);

        for (int row = 0; row < outer; ++row) {
            double sum = 0.0;
            for (int col = 0; col < inner; ++col) {
                const float value = result[static_cast<size_t>(row) * inner + col];
                EXPECT_GE(value, 0.0f) << "row " << row << " col " << col;
                EXPECT_LE(value, 1.0f) << "row " << row << " col " << col;
                sum += value;
            }
            EXPECT_NEAR(sum, 1.0, 1e-5) << "row " << row << " does not sum to one";
        }
    }

    TEST(SoftmaxKernelTest, MatchesTheHostReference) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A small, hand-checkable case: the device result must equal the definition.
        constexpr int outer = 3, inner = 5;
        const std::vector<float> input = {
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            -1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
        };

        const auto result = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, input, outer, inner);

        for (int row = 0; row < outer; ++row) {
            const float* row_input = input.data() + row * inner;
            const float max_val = *std::max_element(row_input, row_input + inner);
            double sum = 0.0;
            for (int col = 0; col < inner; ++col) {
                sum += std::exp(static_cast<double>(row_input[col] - max_val));
            }
            for (int col = 0; col < inner; ++col) {
                const double expected = std::exp(static_cast<double>(row_input[col] - max_val)) / sum;
                EXPECT_NEAR(result[static_cast<size_t>(row) * inner + col], static_cast<float>(expected), 1e-6) << "row " << row << " col " << col;
            }
        }
    }

    TEST(SoftmaxKernelTest, LargeValuesDoNotOverflow) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Both entry points subtract the row maximum before exponentiating, so an input that
        // would overflow exp() on its own must still produce a valid distribution.
        constexpr int outer = 4, inner = 8;
        std::vector<float> input(static_cast<size_t>(outer) * inner, 1000.0f);
        for (int row = 0; row < outer; ++row) {
            input[static_cast<size_t>(row) * inner] = 1001.0f;
        }

        const auto result = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, input, outer, inner);

        for (int row = 0; row < outer; ++row) {
            double sum = 0.0;
            for (int col = 0; col < inner; ++col) {
                const float value = result[static_cast<size_t>(row) * inner + col];
                EXPECT_TRUE(std::isfinite(value)) << "row " << row << " col " << col;
                sum += value;
            }
            EXPECT_NEAR(sum, 1.0, 1e-5) << "row " << row;
        }
    }

    TEST(SoftmaxKernelTest, StableEntryPointAgreesWithTheSimpleOne) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Two implementations of the same function; they must produce the same numbers.
        constexpr int outer = 8, inner = 256;
        std::vector<float> input(static_cast<size_t>(outer) * inner);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin(static_cast<float>(i) * 0.1f) * 4.0f;
        }

        const auto simple = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, input, outer, inner);

        const auto stable = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax_stable(in, out, o, i, s); }, input, outer, inner);

        for (size_t i = 0; i < simple.size(); ++i) {
            EXPECT_NEAR(simple[i], stable[i], 1e-6) << "element " << i;
        }
    }

    TEST(SoftmaxKernelTest, ASingleRowAndASingleElementAreHandled) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Degenerate shapes: the grid arithmetic must not divide by zero or under-launch.
        const std::vector<float> single_element = {2.0f};
        const auto one_by_one = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, single_element, 1, 1);
        ASSERT_EQ(one_by_one.size(), 1u);
        EXPECT_FLOAT_EQ(one_by_one[0], 1.0f) << "softmax of a single element is 1";

        const std::vector<float> one_row = {1.0f, 1.0f, 1.0f, 1.0f};
        const auto row = run_softmax([](const float* in, float* out, int o, int i, cudaStream_t s) { softmax(in, out, o, i, s); }, one_row, 1, 4);
        for (const float value : row) {
            EXPECT_NEAR(value, 0.25f, 1e-6) << "four equal logits should be uniform";
        }
    }

    TEST(SoftmaxKernelTest, EmptyInputIsANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();
        EXPECT_NO_THROW(softmax(nullptr, nullptr, 0, 16, nullptr));
        EXPECT_NO_THROW(softmax(nullptr, nullptr, 16, 0, nullptr));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "an empty softmax must not leave a launch error behind";
    }

    // softmax_stable() and log_softmax() had no dimension guard, while softmax() beside them
    // did. Both launch with the row count as the grid dimension:
    //
    //     softmax_stable_kernel<<<outer_dim, block_size, shared_mem, stream>>>(...)
    //
    // so outer_dim <= 0 is an invalid launch configuration. Measured directly: a launch with
    // a zero grid dimension returns cudaErrorInvalidConfiguration (9). A caller that passed
    // an empty batch got that error surfaced as an exception from CUDA_CHECK, at a line that
    // says nothing about which argument was wrong.
    TEST(SoftmaxKernelTest, AZeroRowCountIsANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        std::vector<float> buffer(4, 1.0f);
        DeviceArray d(buffer);

        // Every entry point must accept "no rows" as nothing to do.
        EXPECT_NO_THROW(softmax(d.data(), d.data(), 0, 4, nullptr));
        EXPECT_NO_THROW(softmax_stable(d.data(), d.data(), 0, 4, nullptr));
        EXPECT_NO_THROW(log_softmax(d.data(), d.data(), 0, 4, nullptr));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "a computed empty input must not leave an error behind";

        // Negative dimensions are caller errors, and must not be launched either.
        EXPECT_NO_THROW(softmax_stable(d.data(), d.data(), -1, 4, nullptr));
        EXPECT_NO_THROW(log_softmax(d.data(), d.data(), -1, 4, nullptr));
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

}  // namespace cuda::neural::test
