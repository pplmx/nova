/**
 * @file launch_error_check_test.cpp
 * @brief Every kernel launch must report its own failure
 *
 * A launch that fails leaves its error pending on the CUDA error channel. An entry point
 * that does not check clears it only by accident, when some later CUDA call happens to
 * call cudaGetLastError itself — so the failure surfaces at an unrelated place, on an
 * unrelated call, and is reported as if it belonged there.
 *
 * Comparing entry points inside each directory turned up six that launched without
 * checking, while their siblings did:
 *
 *     sync_batch_norm_forward_inference   (18 siblings checked, this one did not)
 *     fused_matmul_bias, _relu, _sigmoid  (kernel_fusion.cu's other kernels checked)
 *     magnitude, power_spectrum           (fft.cu's plan entry points checked)
 *     apply_activation                    (launched inside a switch, never checked)
 *
 * All six now check. Four of them additionally accepted an empty extent, which produced a
 * zero-sized grid — the failure that made the missing check observable:
 *
 *     sync_batch_norm_forward_inference(batch=0)  -> cudaErrorInvalidConfiguration (9)
 *        left pending, with no exception raised
 */

#include <cuda/fft/fft.h>
#include <cuda/neural/fusion/kernel_fusion.h>
#include <cuda/neural/sync_batch_norm.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <stdexcept>
#include <vector>

namespace launch_error_check_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

    }  // namespace

    // Regression: this left cudaErrorInvalidConfiguration pending and raised nothing.
    TEST(LaunchErrorCheckTest, InferenceBatchNormRejectsAnEmptyBatch) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();

        EXPECT_THROW(cuda::neural::sync_batch_norm_forward_inference(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 4, 2, 1e-5f, nullptr), std::invalid_argument);

        EXPECT_THROW(cuda::neural::sync_batch_norm_forward_inference(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 4, 0, 1e-5f, nullptr), std::invalid_argument);

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "the failure must be raised at the call, not left pending";
    }

    TEST(LaunchErrorCheckTest, InferenceBatchNormStillRunsForARealShape) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A 1x2x2 inference pass: gamma = beta = 1, running_mean = 0, running_var = 1.
        constexpr int batch = 1, features = 2, spatial = 2;
        const size_t count = static_cast<size_t>(batch) * features * spatial;

        const std::vector<float> input(count, 0.5f);
        const std::vector<float> gamma(features, 1.0f);
        const std::vector<float> beta(features, 1.0f);
        const std::vector<float> mean(features, 0.0f);
        const std::vector<float> var(features, 1.0f);

        float* d_input = nullptr;
        float* d_output = nullptr;
        float* d_gamma = nullptr;
        float* d_beta = nullptr;
        float* d_mean = nullptr;
        float* d_var = nullptr;

        ASSERT_EQ(cudaMalloc(&d_input, count * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, count * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_gamma, features * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_beta, features * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_mean, features * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_var, features * sizeof(float)), cudaSuccess);

        ASSERT_EQ(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_gamma, gamma.data(), features * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_beta, beta.data(), features * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_mean, mean.data(), features * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_var, var.data(), features * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

        EXPECT_NO_THROW(cuda::neural::sync_batch_norm_forward_inference(d_input, d_output, d_gamma, d_beta, d_mean, d_var, batch, features, spatial, 1e-5f, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);

        // Normalising 0.5 with mean 0 and unit variance gives 0.5; gamma 1 and beta 1 give
        // 1.5.
        std::vector<float> result(count, 0.0f);
        ASSERT_EQ(cudaMemcpy(result.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(result[i], 1.5f, 1e-4f) << "element " << i;
        }

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_gamma);
        cudaFree(d_beta);
        cudaFree(d_mean);
        cudaFree(d_var);
    }

    // Regression: the three fusion entry points accepted an empty batch.
    TEST(LaunchErrorCheckTest, FusionEntryPointsAcceptAnEmptyBatch) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();

        EXPECT_NO_THROW(cuda::neural::fusion::fused_matmul_bias(nullptr, nullptr, nullptr, 0, 8, nullptr));
        EXPECT_NO_THROW(cuda::neural::fusion::fused_matmul_bias_relu(nullptr, nullptr, nullptr, 0, 8, nullptr));
        EXPECT_NO_THROW(cuda::neural::fusion::fused_matmul_bias_sigmoid(nullptr, nullptr, nullptr, 0, 8, nullptr));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "an empty batch must not leave a launch error behind";
    }

    // Regression: the two FFT helpers accepted zero bins.
    TEST(LaunchErrorCheckTest, SpectrumHelpersAcceptZeroBins) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();

        EXPECT_NO_THROW(cuda::fft::magnitude(nullptr, nullptr, 0, nullptr));
        EXPECT_NO_THROW(cuda::fft::power_spectrum(nullptr, nullptr, 0, nullptr));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "zero bins must not leave a launch error behind";
    }

    TEST(LaunchErrorCheckTest, MagnitudeMatchesItsDefinition) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // magnitude of (3, 4) is 5; of (5, 12) is 13.
        const std::vector<cuComplex> input = {make_cuFloatComplex(3.0f, 4.0f), make_cuFloatComplex(5.0f, 12.0f)};

        cuComplex* d_input = nullptr;
        float* d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, input.size() * sizeof(cuComplex)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, input.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, input.data(), input.size() * sizeof(cuComplex), cudaMemcpyHostToDevice), cudaSuccess);

        cuda::fft::magnitude(d_input, d_output, input.size(), nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<float> result(input.size(), 0.0f);
        ASSERT_EQ(cudaMemcpy(result.data(), d_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_NEAR(result[0], 5.0f, 1e-5f);
        EXPECT_NEAR(result[1], 13.0f, 1e-5f);

        cudaFree(d_input);
        cudaFree(d_output);
    }

}  // namespace launch_error_check_test
