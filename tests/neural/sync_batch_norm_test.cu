#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuda/neural/sync_batch_norm.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"
#include <cmath>
#include <random>
#include <vector>

namespace cuda::neural {

class SyncBatchNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&original_device_);
    }

    void TearDown() override {
        cudaSetDevice(original_device_);
    }

    int original_device_ = 0;
};

TEST_F(SyncBatchNormTest, Constructor) {
    SyncBatchNorm bn(128, 1e-5f, 0.1f);
    EXPECT_EQ(bn.num_features(), 128);
    EXPECT_EQ(bn.eps(), 1e-5f);
    EXPECT_EQ(bn.momentum(), 0.1f);
    EXPECT_TRUE(bn.is_training());
}

TEST_F(SyncBatchNormTest, ModeSwitching) {
    SyncBatchNorm bn(64);

    bn.set_training(true);
    EXPECT_TRUE(bn.is_training());

    bn.set_training(false);
    EXPECT_FALSE(bn.is_training());
}

TEST_F(SyncBatchNormTest, SingleGPU_ForwardTraining) {
    auto& mesh = mesh::DeviceMesh::instance();
    // DeviceMesh reports device_count() == 0 until initialize() is called; the
    // mesh tests initialize it in their own SetUp, but this suite runs earlier
    // in the process, so without this the tests skipped even on a real GPU.
    mesh.initialize();
    if (mesh.device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int batch_size = 4;
    const int num_features = 32;
    const int spatial_size = 8;
    const int n = batch_size * num_features * spatial_size;

    std::vector<float> input(n);
    std::vector<float> output(n);
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 256) / 128.0f - 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    SyncBatchNorm bn(num_features);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bn.forward_training(d_input, d_output, batch_size, spatial_size, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> result(n);
    cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "Output is NaN at index " << i;
        EXPECT_FALSE(std::isinf(result[i])) << "Output is Inf at index " << i;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(SyncBatchNormTest, SingleGPU_ForwardInference) {
    auto& mesh = mesh::DeviceMesh::instance();
    mesh.initialize();
    if (mesh.device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int batch_size = 4;
    const int num_features = 32;
    const int spatial_size = 8;
    const int n = batch_size * num_features * spatial_size;

    std::vector<float> input(n);
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 256) / 128.0f - 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    SyncBatchNorm bn(num_features);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bn.forward_training(d_input, d_output, batch_size, spatial_size, stream);
    cudaStreamSynchronize(stream);

    bn.set_training(false);
    bn.forward_inference(d_input, d_output, batch_size, spatial_size, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> result(n);
    cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "Output is NaN at index " << i;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(SyncBatchNormTest, RunningStatsInitialized) {
    SyncBatchNorm bn(64);

    std::vector<float> mean(64);
    std::vector<float> var(64);
    cudaMemcpy(mean.data(), bn.running_mean(), 64 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(var.data(), bn.running_var(), 64 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(mean[i], 0.0f) << "Running mean should be initialized to 0";
        EXPECT_EQ(var[i], 0.0f) << "Running var should be initialized to 0";
    }
}

TEST_F(SyncBatchNormTest, GammaBetaInitialization) {
    SyncBatchNorm bn(64);

    std::vector<float> gamma(64);
    std::vector<float> beta(64);
    cudaMemcpy(gamma.data(), bn.gamma(), 64 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(beta.data(), bn.beta(), 64 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(gamma[i], 1.0f) << "Gamma should be initialized to 1";
        EXPECT_EQ(beta[i], 0.0f) << "Beta should be initialized to 0";
    }
}

TEST_F(SyncBatchNormTest, SingleGPU_Backward) {
    auto& mesh = mesh::DeviceMesh::instance();
    mesh.initialize();
    if (mesh.device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int batch_size = 4;
    const int num_features = 32;
    const int spatial_size = 8;
    const int n = batch_size * num_features * spatial_size;

    std::vector<float> input(n);
    std::vector<float> d_output(n);
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 256) / 128.0f - 1.0f;
        d_output[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    }

    float *d_input, *d_output_dev, *d_result, *d_d_input, *d_d_gamma, *d_d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_dev, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_gamma, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_beta, num_features * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_dev, d_output.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    SyncBatchNorm bn(num_features);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bn.forward_training(d_input, d_result, batch_size, spatial_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bn.backward(d_input, d_output_dev, d_d_input, d_d_gamma, d_d_beta,
                batch_size, spatial_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> d_input_result(n);
    std::vector<float> d_gamma_result(num_features);
    std::vector<float> d_beta_result(num_features);
    CUDA_CHECK(cudaMemcpy(d_input_result.data(), d_d_input, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_gamma_result.data(), d_d_gamma, num_features * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_beta_result.data(), d_d_beta, num_features * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        EXPECT_FALSE(std::isnan(d_input_result[i])) << "d_input is NaN at index " << i;
        EXPECT_FALSE(std::isinf(d_input_result[i])) << "d_input is Inf at index " << i;
    }
    for (int i = 0; i < num_features; ++i) {
        EXPECT_FALSE(std::isnan(d_gamma_result[i])) << "d_gamma is NaN at index " << i;
        EXPECT_FALSE(std::isinf(d_gamma_result[i])) << "d_gamma is Inf at index " << i;
        EXPECT_FALSE(std::isnan(d_beta_result[i])) << "d_beta is NaN at index " << i;
        EXPECT_FALSE(std::isinf(d_beta_result[i])) << "d_beta is Inf at index " << i;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_dev));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_d_input));
    CUDA_CHECK(cudaFree(d_d_gamma));
    CUDA_CHECK(cudaFree(d_d_beta));
}

// Verifies backward() gradients against an independent CPU reference of the
// batch-normalization backward pass. The previous GPU implementation computed
// dx's denominator from the variance gradient (sqrtf(d_var)) which is NaN for
// negative d_var, so every output gradient was NaN.
TEST_F(SyncBatchNormTest, BackwardMatchesCPUReference) {
    auto& mesh = mesh::DeviceMesh::instance();
    mesh.initialize();
    if (mesh.device_count() < 1) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int batch_size = 3;
    const int num_features = 8;
    const int spatial_size = 5;
    const int n = batch_size * num_features * spatial_size;
    const float eps = 1e-5f;
    const int count_per_feature = batch_size * spatial_size;
    const float inv_n = 1.0f / static_cast<float>(count_per_feature);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(n);
    std::vector<float> d_output(n);
    std::vector<float> gamma(num_features);
    for (int i = 0; i < n; ++i) {
        input[i] = dist(rng);
        d_output[i] = dist(rng) * 0.1f;
    }
    for (int c = 0; c < num_features; ++c) gamma[c] = 0.5f + 0.1f * static_cast<float>(c);

    // ---- CPU reference ----
    auto idx_of = [&](int b, int c, int s) { return (b * num_features + c) * spatial_size + s; };
    std::vector<float> ref_mean(num_features), ref_var(num_features);
    for (int c = 0; c < num_features; ++c) {
        float s = 0.0f;
        for (int b = 0; b < batch_size; ++b)
            for (int sp = 0; sp < spatial_size; ++sp) s += input[idx_of(b, c, sp)];
        ref_mean[c] = s * inv_n;
        float v = 0.0f;
        for (int b = 0; b < batch_size; ++b)
            for (int sp = 0; sp < spatial_size; ++sp) {
                float d = input[idx_of(b, c, sp)] - ref_mean[c];
                v += d * d;
            }
        ref_var[c] = v * inv_n;
    }

    std::vector<float> ref_d_input(n, 0.0f);
    std::vector<float> ref_d_gamma(num_features, 0.0f);
    std::vector<float> ref_d_beta(num_features, 0.0f);

    for (int c = 0; c < num_features; ++c) {
        float inv_sigma = 1.0f / std::sqrt(ref_var[c] + eps);
        float sigma3 = inv_sigma * inv_sigma * inv_sigma;

        float dvar = 0.0f, sum_dxhat = 0.0f, sum_dy = 0.0f, sum_dy_xhat = 0.0f, sum_centered = 0.0f;
        std::vector<float> dxhat(count_per_feature), centered(count_per_feature);
        int k = 0;
        for (int b = 0; b < batch_size; ++b)
            for (int sp = 0; sp < spatial_size; ++sp, ++k) {
                int idx = idx_of(b, c, sp);
                centered[k] = input[idx] - ref_mean[c];
                dxhat[k] = d_output[idx] * gamma[c];
                sum_dxhat += dxhat[k];
                sum_dy += d_output[idx];
                sum_dy_xhat += d_output[idx] * centered[k] * inv_sigma;
                sum_centered += centered[k];
                dvar += dxhat[k] * centered[k];
            }
        dvar *= -0.5f * sigma3;
        float dmean = sum_dxhat * (-inv_sigma) + dvar * (-2.0f * inv_n) * sum_centered;

        ref_d_gamma[c] = sum_dy_xhat;
        ref_d_beta[c] = sum_dy;
        for (int k = 0; k < count_per_feature; ++k) {
            int b = k / spatial_size, sp = k % spatial_size;
            ref_d_input[idx_of(b, c, sp)] =
                dxhat[k] * inv_sigma + dvar * (2.0f * centered[k] * inv_n) + dmean * inv_n;
        }
    }

    // ---- GPU ----
    float *d_input, *d_output_dev, *d_result, *d_d_input, *d_d_gamma, *d_d_beta, *d_gamma;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_dev, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_gamma, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_beta, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, num_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_dev, d_output.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), num_features * sizeof(float), cudaMemcpyHostToDevice));

    SyncBatchNorm bn(num_features, eps);
    CUDA_CHECK(cudaMemcpy(bn.mutable_gamma(), d_gamma, num_features * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bn.forward_training(d_input, d_result, batch_size, spatial_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bn.backward(d_input, d_output_dev, d_d_input, d_d_gamma, d_d_beta, batch_size, spatial_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> gpu_d_input(n), gpu_d_gamma(num_features), gpu_d_beta(num_features);
    CUDA_CHECK(cudaMemcpy(gpu_d_input.data(), d_d_input, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_d_gamma.data(), d_d_gamma, num_features * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_d_beta.data(), d_d_beta, num_features * sizeof(float), cudaMemcpyDeviceToHost));

    for (int c = 0; c < num_features; ++c) {
        EXPECT_NEAR(gpu_d_gamma[c], ref_d_gamma[c], 1e-3f) << "d_gamma[" << c << "]";
        EXPECT_NEAR(gpu_d_beta[c], ref_d_beta[c], 1e-3f) << "d_beta[" << c << "]";
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(gpu_d_input[i], ref_d_input[i], 1e-3f) << "d_input[" << i << "]";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_dev));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_d_input));
    CUDA_CHECK(cudaFree(d_d_gamma));
    CUDA_CHECK(cudaFree(d_d_beta));
    CUDA_CHECK(cudaFree(d_gamma));
}

} // namespace cuda::neural
