/**
 * @file fp8_activation_kernels_test.cpp
 * @brief Contracts for the FP8 activation launch wrappers
 *
 * src/cuda/quantize/fp8_activation.cu wraps four element-wise activation kernels.
 * Only two of its seven functions were referenced from a test, and the launch
 * wrappers had a defect shared with the f32 activations:
 *
 *   relu_kernel<<<grid_for(n), kBlockSize, 0, stream>>>(...);
 *
 * grid_for(0) is 0, and CUDA rejects a zero-sized grid with "invalid configuration
 * argument". report_launch_error prints that to stderr and CLEARS it by reading it,
 * so the caller's later cudaGetLastError() returns success: an empty tensor
 * silently failed to launch and nothing reported it. Measured on the pre-fix code:
 *
 *   relu(n=0): CUDA error in relu: invalid configuration argument
 *   no throw; cudaGetLastError=0 (no error)
 *
 * All four wrappers now return early when there is nothing to do.
 *
 * The maths live in fp8_activation.hpp and were checked and are correct: relu
 * clamping at zero, the exact erf-based GELU, the sigmoid, and leaky relu's slope.
 */

#include <cuda/quantize/fp8_activation.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

namespace nova::quantize::cuda::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Owns a device buffer of FP8 elements (one byte each).
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
                cudaMemset(ptr_, 0, count_ > 0 ? count_ : 1);
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

            std::vector<T> read() const {
                std::vector<T> host(count_);
                if (count_ > 0) {
                    cudaMemcpy(host.data(), ptr_, count_, cudaMemcpyDeviceToHost);
                }
                return host;
            }

        private:
            T* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: this launched a zero-sized grid and the error was swallowed.
    TEST(Fp8ActivationLaunchTest, EmptyInputsDoNotLaunchAndLeaveNoError) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A one-element buffer backs each call so no null pointer reaches the runtime;
        // the count passed to the wrapper is what must short-circuit.
        DeviceArray<FP8E4M3> input(1);
        DeviceArray<FP8E4M3> output(1);

        // The wrappers report launch failures by printing to stderr and then READING
        // the error to clear it, so the post-call cudaGetLastError() is success either
        // way. What distinguishes the fix is on the stderr stream: the pre-fix code
        // prints "CUDA error in relu: invalid configuration argument" for an empty
        // tensor. Capture stderr to observe that directly.
        testing::internal::CaptureStderr();

        EXPECT_NO_THROW(relu(input.data(), output.data(), 0, nullptr));
        EXPECT_NO_THROW(gelu(input.data(), output.data(), 0, nullptr));
        EXPECT_NO_THROW(sigmoid(input.data(), output.data(), 0, nullptr));
        EXPECT_NO_THROW(leaky_relu(input.data(), output.data(), 0, 0.01f, nullptr));

        const std::string diagnostics = testing::internal::GetCapturedStderr();

        EXPECT_TRUE(diagnostics.empty()) << "an empty tensor launched anyway and reported: " << diagnostics;
        EXPECT_EQ(diagnostics.find("invalid configuration argument"), std::string::npos) << "a zero-sized grid was launched: " << diagnostics;
    }

    TEST(Fp8ActivationLaunchTest, ReluClampsNegativesAndKeepsPositives) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> values = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f};
        std::vector<FP8E4M3> host_input;
        host_input.reserve(values.size());
        for (const float value : values) {
            host_input.emplace_back(value);
        }

        DeviceArray<FP8E4M3> input(host_input);
        DeviceArray<FP8E4M3> output(host_input.size());

        relu(input.data(), output.data(), host_input.size(), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const float expected = values[i] < 0.0f ? 0.0f : values[i];
            EXPECT_NEAR(static_cast<float>(result[i]), expected, 0.02f) << "element " << i << " of " << values[i];
        }
    }

    TEST(Fp8ActivationLaunchTest, SigmoidIsBoundedAndCentred) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> values = {-4.0f, 0.0f, 4.0f};
        std::vector<FP8E4M3> host_input;
        for (const float value : values) {
            host_input.emplace_back(value);
        }

        DeviceArray<FP8E4M3> input(host_input);
        DeviceArray<FP8E4M3> output(host_input.size());

        sigmoid(input.data(), output.data(), host_input.size(), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), 3u);
        // FP8 E4M3 has three mantissa bits, so the tolerance is one representation
        // step rather than the 1e-6 an f32 test would use.
        EXPECT_NEAR(static_cast<float>(result[0]), 0.0179862f, 0.02f);
        EXPECT_NEAR(static_cast<float>(result[1]), 0.5f, 0.02f);
        EXPECT_NEAR(static_cast<float>(result[2]), 0.9820138f, 0.02f);
    }

    TEST(Fp8ActivationLaunchTest, LeakyReluUsesAlphaForNegatives) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const float alpha = 0.5f;
        std::vector<FP8E4M3> host_input;
        for (const float value : {-1.0f, 1.0f}) {
            host_input.emplace_back(value);
        }

        DeviceArray<FP8E4M3> input(host_input);
        DeviceArray<FP8E4M3> output(host_input.size());

        leaky_relu(input.data(), output.data(), host_input.size(), alpha, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), 2u);
        EXPECT_NEAR(static_cast<float>(result[0]), -0.5f, 0.05f) << "-1 * 0.5";
        EXPECT_NEAR(static_cast<float>(result[1]), 1.0f, 0.05f) << "positives pass through unchanged";
    }

    TEST(Fp8ActivationLaunchTest, GeluMatchesTheExactDefinition) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The exact GELU is x * 0.5 * (1 + erf(x / sqrt(2))).
        std::vector<float> values;
        for (const double x : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
            values.push_back(static_cast<float>(x));
        }

        std::vector<FP8E4M3> host_input;
        for (const float value : values) {
            host_input.emplace_back(value);
        }

        DeviceArray<FP8E4M3> input(host_input);
        DeviceArray<FP8E4M3> output(host_input.size());

        gelu(input.data(), output.data(), host_input.size(), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const double x = static_cast<double>(values[i]);
            const double exact = x * 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
            EXPECT_NEAR(static_cast<double>(static_cast<float>(result[i])), exact, 0.05) << "gelu(" << x << ")";
        }
    }

    TEST(Fp8ActivationLaunchTest, ProcessesMoreThanOneBlock) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 1000 elements span four 256-thread blocks, so the tail and the block stride
        // are both exercised.
        constexpr size_t count = 1000;
        std::vector<FP8E4M3> host_input;
        host_input.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            host_input.emplace_back(-1.0f);
        }

        DeviceArray<FP8E4M3> input(host_input);
        DeviceArray<FP8E4M3> output(count);

        relu(input.data(), output.data(), count, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), count);
        for (size_t i = 0; i < count; ++i) {
            ASSERT_NEAR(static_cast<float>(result[i]), 0.0f, 0.02f) << "element " << i << " was not clamped";
        }
    }

    TEST(Fp8ActivationLaunchTest, E5M2VariantBehavesTheSameWay) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Both FP8 formats are instantiated; E5M2 has a wider exponent and fewer
        // mantissa bits, so the tolerance is looser.
        std::vector<FP8E5M2> host_input;
        for (const float value : {-2.0f, 2.0f}) {
            host_input.emplace_back(value);
        }

        DeviceArray<FP8E5M2> input(host_input);
        DeviceArray<FP8E5M2> output(host_input.size());

        relu(input.data(), output.data(), host_input.size(), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), 2u);
        EXPECT_NEAR(static_cast<float>(result[0]), 0.0f, 0.01f);
        EXPECT_NEAR(static_cast<float>(result[1]), 2.0f, 0.1f);
    }

}  // namespace nova::quantize::cuda::test
