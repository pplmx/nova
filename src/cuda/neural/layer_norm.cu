#include "cuda/neural/layer_norm.h"

#include "cuda/device/error.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cuda::neural {

LayerNormResult::LayerNormResult(int size) : size(size) {
    output = new float[size];
    mean = new float[size];
    variance = new float[size];

    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_variance, size * sizeof(float)));
}

LayerNormResult::~LayerNormResult() {
    clear();
}

void LayerNormResult::upload() {
    CUDA_CHECK(cudaMemcpy(d_output, output, size * sizeof(float), cudaMemcpyHostToDevice));
}

void LayerNormResult::download() {
    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void LayerNormResult::clear() {
    delete[] output;
    delete[] mean;
    delete[] variance;

    if (d_output) cudaFree(d_output);
    if (d_mean) cudaFree(d_mean);
    if (d_variance) cudaFree(d_variance);

    output = nullptr;
    mean = nullptr;
    variance = nullptr;
    d_output = nullptr;
    d_mean = nullptr;
    d_variance = nullptr;
}

namespace {

__global__ void layer_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* variance,
    int batch_size,
    int normalized_shape,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* x = input + idx * normalized_shape;

    float sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        sum += x[i];
    }
    float mean_val = sum / normalized_shape;
    mean[idx] = mean_val;

    float var_sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        float diff = x[i] - mean_val;
        var_sum += diff * diff;
    }
    float var_val = var_sum / normalized_shape;
    variance[idx] = var_val;

    float inv_std = rsqrtf(var_val + eps);

    float* y = output + idx * normalized_shape;
    for (int i = 0; i < normalized_shape; ++i) {
        float normalized = (x[i] - mean_val) * inv_std;
        if (gamma && beta) {
            y[i] = gamma[i] * normalized + beta[i];
        } else {
            y[i] = normalized;
        }
    }
}

}  // anonymous namespace

void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* variance,
    int batch_size,
    int normalized_shape,
    float eps,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = batch_size;

    layer_norm_kernel<<<grid_size, block_size, 0, stream>>>(
        input, gamma, beta, output, mean, variance,
        batch_size, normalized_shape, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

void layer_norm_inference(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int normalized_shape,
    float eps,
    cudaStream_t stream
) {
    // Inference normalizes with internally computed per-row stats (the API
    // takes no mean/variance). Route through layer_norm() with scratch — the
    // previous dedicated kernel dereferenced null mean/var unconditionally (a
    // latent device fault if ever called; dead in-tree — fixed while the file
    // was in the v2.28 review surface).
    cuda::memory::Buffer<float> mean(batch_size);
    cuda::memory::Buffer<float> var(batch_size);
    layer_norm(input, gamma, beta, output, mean.data(), var.data(),
               batch_size, normalized_shape, eps, stream);
}

// ============================================================================
// v2.28 trainable LayerNorm (TASK-041 / DEC-014): device forward + analytic
// backward with trainable affine gamma/beta. Collective-free — each row
// normalizes independently over the (replicated) hidden dim.
// ============================================================================

namespace {

// Forward: y = gamma * xhat + beta per row, one block per row.
__global__ void ln_forward_trainable_kernel(
    const float* x, const float* gamma, const float* beta, float* y, int m,
    int h, float eps) {
    const int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + row * h;
    float sum = 0.0f;
    for (int j = 0; j < h; ++j) sum += xr[j];
    const float mean = sum / h;
    float var = 0.0f;
    for (int j = 0; j < h; ++j) {
        const float d = xr[j] - mean;
        var += d * d;
    }
    const float inv = rsqrtf(var / h + eps);
    float* yr = y + row * h;
    for (int j = 0; j < h; ++j) {
        const float xhat = (xr[j] - mean) * inv;
        yr[j] = gamma[j] * xhat + beta[j];
    }
}

// Backward pass 1 (one block per row): row mean/var/inv + the row sums
// sum_dxhat and sum_dxhat*xhat that dL/dx needs (stats[4*row …]), and the
// dgamma/dbeta row contributions accumulated with atomics (the affine grads
// are sums over the batch rows; float atomicAdd order only perturbs the sum at
// ~ulp level, far inside the parity tolerance).
__global__ void ln_backward_stats_kernel(
    const float* x, const float* gamma, const float* dy, float* dgamma,
    float* dbeta, float* stats, int m, int h, float eps) {
    const int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + row * h;
    const float* dyr = dy + row * h;
    float sum = 0.0f;
    for (int j = 0; j < h; ++j) sum += xr[j];
    const float mean = sum / h;
    float var = 0.0f;
    for (int j = 0; j < h; ++j) {
        const float d = xr[j] - mean;
        var += d * d;
    }
    const float inv = rsqrtf(var / h + eps);
    float sdx = 0.0f, sdxh = 0.0f;
    for (int j = 0; j < h; ++j) {
        const float xhat = (xr[j] - mean) * inv;
        const float dxhat = dyr[j] * gamma[j];
        sdx += dxhat;
        sdxh += dxhat * xhat;
        atomicAdd(&dgamma[j], dyr[j] * xhat);
        atomicAdd(&dbeta[j], dyr[j]);
    }
    float* s = stats + row * 4;
    s[0] = mean;
    s[1] = inv;
    s[2] = sdx;
    s[3] = sdxh;
}

// Backward pass 2 (element-wise): dL/dx = inv*(dxhat - mean(dxhat) -
// xhat*mean(dxhat*xhat)).
__global__ void ln_backward_input_kernel(
    const float* x, const float* gamma, const float* dy, const float* stats,
    float* dx, int m, int h) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * h) return;
    const int row = idx / h;
    const int j = idx % h;
    const float* s = stats + row * 4;
    const float mean = s[0];
    const float inv = s[1];
    const float sdx = s[2];
    const float sdxh = s[3];
    const float xhat = (x[idx] - mean) * inv;
    const float dxhat = dy[idx] * gamma[j];
    dx[idx] = inv * (dxhat - sdx / h - xhat * (sdxh / h));
}

}  // namespace

void layer_norm_backward(
    const float* input, const float* gamma, const float* grad_output,
    float* grad_input, float* grad_gamma, float* grad_beta, int batch_size,
    int normalized_shape, float eps, cudaStream_t stream) {
    cuda::memory::Buffer<float> stats(static_cast<size_t>(batch_size) * 4);
    CUDA_CHECK(cudaMemsetAsync(grad_gamma, 0, sizeof(float) * normalized_shape,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(grad_beta, 0, sizeof(float) * normalized_shape,
                               stream));
    ln_backward_stats_kernel<<<batch_size, 1, 0, stream>>>(
        input, gamma, grad_output, grad_gamma, grad_beta, stats.data(),
        batch_size, normalized_shape, eps);
    const int grid = (batch_size * normalized_shape + 255) / 256;
    ln_backward_input_kernel<<<grid, 256, 0, stream>>>(
        input, gamma, grad_output, stats.data(), grad_input, batch_size,
        normalized_shape);
    CUDA_CHECK(cudaGetLastError());
}

LayerNorm::LayerNorm(int hidden, float eps) : hidden_(hidden), eps_(eps) {
    if (hidden_ <= 0) {
        throw std::invalid_argument("LayerNorm: hidden must be positive");
    }
    d_gamma_ = std::make_unique<cuda::memory::Buffer<float>>(hidden_);
    d_beta_ = std::make_unique<cuda::memory::Buffer<float>>(hidden_);
}

LayerNorm::~LayerNorm() = default;

void LayerNorm::ensure_stats(int batch) {
    if (d_stats_ && stats_batch_ >= batch) return;
    d_stats_ = std::make_unique<cuda::memory::Buffer<float>>(
        static_cast<size_t>(batch) * 4);
    stats_batch_ = batch;
}

void LayerNorm::set_weight(const float* gamma, const float* beta) {
    d_gamma_->copy_from(gamma, hidden_);
    d_beta_->copy_from(beta, hidden_);
}

void LayerNorm::forward(const float* input, float* output, int batch,
                        cudaStream_t stream) {
    ln_forward_trainable_kernel<<<batch, 1, 0, stream>>>(
        input, d_gamma_->data(), d_beta_->data(), output, batch, hidden_, eps_);
    CUDA_CHECK(cudaGetLastError());
}

void LayerNorm::backward(const float* input, const float* grad_output,
                         float* grad_input, float* grad_gamma,
                         float* grad_beta, int batch, cudaStream_t stream) {
    ensure_stats(batch);
    CUDA_CHECK(cudaMemsetAsync(grad_gamma, 0, sizeof(float) * hidden_, stream));
    CUDA_CHECK(cudaMemsetAsync(grad_beta, 0, sizeof(float) * hidden_, stream));
    ln_backward_stats_kernel<<<batch, 1, 0, stream>>>(
        input, d_gamma_->data(), grad_output, grad_gamma, grad_beta,
        d_stats_->data(), batch, hidden_, eps_);
    const int grid = (batch * hidden_ + 255) / 256;
    ln_backward_input_kernel<<<grid, 256, 0, stream>>>(
        input, d_gamma_->data(), grad_output, d_stats_->data(), grad_input,
        batch, hidden_);
    CUDA_CHECK(cudaGetLastError());
}

void LayerNorm::copy_weights(float* gamma, float* beta) const {
    d_gamma_->copy_to(gamma, hidden_);
    d_beta_->copy_to(beta, hidden_);
}

void LayerNorm::step(optimizers::AdamWOptimizer& opt_gamma,
                     optimizers::AdamWOptimizer& opt_beta,
                     const float* grad_gamma, const float* grad_beta,
                     int step_no, cudaStream_t stream) {
    opt_gamma.step(d_gamma_->data(), grad_gamma, hidden_, step_no, stream);
    opt_beta.step(d_beta_->data(), grad_beta, hidden_, step_no, stream);
}

int LayerNorm::hidden() const { return hidden_; }
float LayerNorm::eps() const { return eps_; }

}  // namespace cuda::neural
