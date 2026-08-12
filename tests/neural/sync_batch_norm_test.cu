#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuda/neural/sync_batch_norm.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/nccl/nccl_context.h"
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cuda::neural {

namespace {

// Same thread-per-rank harness as the NCCL suite / distributed ops / matmul:
// a barrier so every rank enters together, then `per_rank(d)` on one thread per
// visible device, each pinned with cudaSetDevice (per-thread state). First
// exception on any thread is rethrown after all threads join.
class RankBarrier {
public:
    explicit RankBarrier(int count) : total_(count) {}

    void wait() {
        std::unique_lock<std::mutex> lk(mutex_);
        ++arrived_;
        cv_.wait(lk, [this] { return arrived_ >= total_; });
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int total_;
    int arrived_ = 0;
};

template <typename F>
void run_per_rank(int device_count, F&& per_rank) {
    RankBarrier barrier(device_count);
    std::vector<std::thread> threads;
    threads.reserve(device_count);
    std::mutex error_mutex;
    std::exception_ptr first_error;
    for (int d = 0; d < device_count; ++d) {
        threads.emplace_back([d, &barrier, &per_rank, &error_mutex, &first_error]() {
            bool failed = false;
            try {
                CUDA_CHECK(cudaSetDevice(d));
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
                failed = true;
            }
            barrier.wait();
            if (failed) {
                return;
            }
            try {
                per_rank(d);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

// Genuine multi-GPU run is possible when >= 2 GPUs, NCCL_TESTS_AVAILABLE="1",
// and the shared NcclContext initializes (idempotent). GTEST_SKIP is a
// void-return macro, so callers issue the skip.
bool multigpu_nccl_ready() {
    auto& mesh_ref = mesh::DeviceMesh::instance();
    mesh_ref.initialize();  // device_count() is 0 until the mesh is initialized
    if (mesh_ref.device_count() < 2) {
        return false;
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr || std::string(nccl_env) != "1") {
        return false;
    }
    auto& ctx = cuda::nccl::NcclContext::instance();
    try {
        ctx.initialize();
    } catch (const std::exception&) {
        return false;
    }
    return ctx.has_nccl();
}

}  // namespace

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
    if (mesh.device_count() != 1) {
        GTEST_SKIP() << "Requires single GPU (multi-GPU covered by MultiGpu_MatchesGlobalBatchReference)";
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
    if (mesh.device_count() != 1) {
        GTEST_SKIP() << "Requires single GPU (multi-GPU covered by MultiGpu_MatchesGlobalBatchReference)";
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
    if (mesh.device_count() != 1) {
        GTEST_SKIP() << "Requires single GPU (multi-GPU covered by MultiGpu_MatchesGlobalBatchReference)";
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
    if (mesh.device_count() != 1) {
        GTEST_SKIP() << "Requires single GPU (multi-GPU covered by MultiGpu_MatchesGlobalBatchReference)";
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

// Verifies the multi-GPU SyncBatchNorm path against an independent host
// reference computed over the GLOBAL batch (all ranks' shards concatenated, one
// logical rank). Before this fix, the per-rank all-reduce of saved_mean_/
// saved_var_ summed the local statistics without dividing by the number of
// ranks, so normalization used R x the true global mean/variance and the
// multi-GPU forward/backward diverged from the single-rank result (and the
// tests above now gate to a single GPU). See task-v17c-syncbn-multigpu-backward.
TEST_F(SyncBatchNormTest, MultiGpu_MatchesGlobalBatchReference) {
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "Requires >= 2 GPUs with NCCL_TESTS_AVAILABLE=1";
    }

    const int ranks = mesh::DeviceMesh::instance().device_count();
    const int per_rank_batch = 3;
    const int num_features = 8;
    const int spatial_size = 5;
    const int global_batch = per_rank_batch * ranks;
    const int n_per_rank = per_rank_batch * num_features * spatial_size;
    const int n_global = global_batch * num_features * spatial_size;
    const float eps = 1e-5f;
    const float inv_n_global = 1.0f / static_cast<float>(global_batch * spatial_size);

    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Global host data: shard r occupies batch rows [r*per_rank_batch, (r+1)*...).
    std::vector<float> h_input(n_global), h_dout(n_global);
    std::vector<float> gamma(num_features), beta(num_features);
    for (int i = 0; i < n_global; ++i) {
        h_input[i] = dist(rng);
        h_dout[i] = dist(rng) * 0.1f;
    }
    for (int c = 0; c < num_features; ++c) {
        gamma[c] = 0.5f + 0.1f * static_cast<float>(c);
        beta[c] = -0.25f + 0.05f * static_cast<float>(c);
    }

    // ---- Host reference over the GLOBAL batch (single logical rank) ----
    auto idx_of = [&](int b, int c, int s) { return (b * num_features + c) * spatial_size + s; };
    std::vector<float> ref_mean(num_features), ref_var(num_features);
    for (int c = 0; c < num_features; ++c) {
        float s = 0.0f;
        for (int b = 0; b < global_batch; ++b)
            for (int sp = 0; sp < spatial_size; ++sp) s += h_input[idx_of(b, c, sp)];
        ref_mean[c] = s * inv_n_global;
        float v = 0.0f;
        for (int b = 0; b < global_batch; ++b)
            for (int sp = 0; sp < spatial_size; ++sp) {
                float d = h_input[idx_of(b, c, sp)] - ref_mean[c];
                v += d * d;
            }
        ref_var[c] = v * inv_n_global;
    }

    // Forward output: (x - mean)/sqrt(var+eps) * gamma + beta.
    std::vector<float> ref_out(n_global);
    for (int i = 0; i < n_global; ++i) {
        int c = (i / spatial_size) % num_features;
        ref_out[i] = (h_input[i] - ref_mean[c]) / std::sqrt(ref_var[c] + eps) * gamma[c] + beta[c];
    }

    // Backward gradients: the standard BN formulas over the global batch. The
    // d_mean calculation keeps the sum(x - mean) term the kernel omits (it is
    // ~0 over the full batch by definition of the mean); the residual is far
    // below the 1e-3 tolerance.
    std::vector<float> ref_d_input(n_global, 0.0f);
    std::vector<float> ref_d_gamma(num_features, 0.0f);
    std::vector<float> ref_d_beta(num_features, 0.0f);
    for (int c = 0; c < num_features; ++c) {
        float inv_sigma = 1.0f / std::sqrt(ref_var[c] + eps);
        float sigma3 = inv_sigma * inv_sigma * inv_sigma;
        float dvar = 0.0f, sum_dxhat = 0.0f, sum_dy = 0.0f, sum_dy_xhat = 0.0f, sum_centered = 0.0f;
        std::vector<float> centered(global_batch * spatial_size);
        std::vector<float> dxhat(global_batch * spatial_size);
        int k = 0;
        for (int b = 0; b < global_batch; ++b)
            for (int sp = 0; sp < spatial_size; ++sp, ++k) {
                int idx = idx_of(b, c, sp);
                centered[k] = h_input[idx] - ref_mean[c];
                dxhat[k] = h_dout[idx] * gamma[c];
                sum_dxhat += dxhat[k];
                sum_dy += h_dout[idx];
                sum_dy_xhat += h_dout[idx] * centered[k] * inv_sigma;
                sum_centered += centered[k];
                dvar += dxhat[k] * centered[k];
            }
        dvar *= -0.5f * sigma3;
        float dmean = sum_dxhat * (-inv_sigma) + dvar * (-2.0f * inv_n_global) * sum_centered;
        ref_d_gamma[c] = sum_dy_xhat;
        ref_d_beta[c] = sum_dy;
        k = 0;
        for (int b = 0; b < global_batch; ++b)
            for (int sp = 0; sp < spatial_size; ++sp, ++k) {
                int idx = idx_of(b, c, sp);
                ref_d_input[idx] = dxhat[k] * inv_sigma + dvar * (2.0f * centered[k] * inv_n_global) +
                                   dmean * inv_n_global;
            }
    }

    // ---- Multi-GPU: one thread per rank, each over its own shard ----
    std::vector<std::vector<float>> out_shards(ranks, std::vector<float>(n_per_rank));
    std::vector<std::vector<float>> din_shards(ranks, std::vector<float>(n_per_rank));
    std::vector<std::vector<float>> dgamma_shards(ranks, std::vector<float>(num_features));
    std::vector<std::vector<float>> dbeta_shards(ranks, std::vector<float>(num_features));

    run_per_rank(ranks, [&](int r) {
        const int n_local = per_rank_batch * num_features * spatial_size;
        cuda::memory::Buffer<float> d_in(n_local);
        cuda::memory::Buffer<float> d_dout(n_local);
        cuda::memory::Buffer<float> d_out(n_local);
        cuda::memory::Buffer<float> d_din(n_local);
        cuda::memory::Buffer<float> d_dgamma(num_features);
        cuda::memory::Buffer<float> d_dbeta(num_features);

        // This rank's shard = global rows [r*per_rank_batch, (r+1)*per_rank_batch).
        const int base = r * n_local;
        std::vector<float> shard_in(n_local), shard_dout(n_local);
        for (int i = 0; i < n_local; ++i) {
            shard_in[i] = h_input[base + i];
            shard_dout[i] = h_dout[base + i];
        }
        d_in.copy_from(shard_in.data(), n_local);
        d_dout.copy_from(shard_dout.data(), n_local);

        SyncBatchNorm bn(num_features, eps);
        CUDA_CHECK(cudaMemcpy(bn.mutable_gamma(), gamma.data(), num_features * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bn.mutable_beta(), beta.data(), num_features * sizeof(float),
                              cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        bn.forward_training(d_in.data(), d_out.data(), per_rank_batch, spatial_size, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bn.backward(d_in.data(), d_dout.data(), d_din.data(), d_dgamma.data(), d_dbeta.data(),
                    per_rank_batch, spatial_size, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        d_out.copy_to(out_shards[r].data(), n_local);
        d_din.copy_to(din_shards[r].data(), n_local);
        d_dgamma.copy_to(dgamma_shards[r].data(), num_features);
        d_dbeta.copy_to(dbeta_shards[r].data(), num_features);
        CUDA_CHECK(cudaStreamDestroy(stream));
    });

    // ---- Assert: multi-GPU path == single-rank global-batch reference ----
    for (int r = 0; r < ranks; ++r) {
        SCOPED_TRACE("rank " + std::to_string(r));
        int base = r * n_per_rank;
        for (int i = 0; i < n_per_rank; ++i) {
            EXPECT_NEAR(out_shards[r][i], ref_out[base + i], 1e-3f)
                << "forward output elem " << i;
            EXPECT_NEAR(din_shards[r][i], ref_d_input[base + i], 1e-3f)
                << "d_input elem " << i;
        }
        for (int c = 0; c < num_features; ++c) {
            EXPECT_NEAR(dgamma_shards[r][c], ref_d_gamma[c], 1e-3f) << "d_gamma[" << c << "]";
            EXPECT_NEAR(dbeta_shards[r][c], ref_d_beta[c], 1e-3f) << "d_beta[" << c << "]";
        }
    }
}

} // namespace cuda::neural
