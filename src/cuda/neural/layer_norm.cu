#include "cuda/neural/layer_norm.h"

#include "cuda/device/error.h"

#include <cmath>
#include <cstdint>
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

constexpr int kLnBlock = 256;  // block size for the per-row trainable kernels

// Block-wide sum over kLnBlock lanes (all threads active; kLnBlock % 32 == 0
// so the full-warp shuffle masks are valid). `red` must hold kLnBlock/32
// floats; the reduced value is broadcast to ALL threads on return.
//
// The entry __syncthreads() is load-bearing: this helper both returns `red[0]`
// (a read every thread issues after the final barrier) AND writes `red[wid]`
// (lane 0's STS) at its start. Kernels call it several times on the same
// buffer, so without the entry barrier the first writer of the next call races
// the last reader of the previous call (attention_kernels.cu had the same
// fault — cpp-reviewer, compute-sanitizer racecheck-verified).
__device__ float ln_block_reduce_sum(float v, float* red) {
    __syncthreads();
    constexpr unsigned kFull = 0xffffffffu;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFull, v, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float acc = (threadIdx.x < kLnBlock / 32) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(kFull, acc, o);
        if (lane == 0) red[0] = acc;
    }
    __syncthreads();
    return red[0];
}

// Forward: y = gamma * xhat + beta per row, one BLOCK per row (v2.29, TASK-045).
// The row mean/variance are block reductions over the hidden dim instead of
// the v2.28 one-thread-per-row serial sums (fine at hidden<=64, a per-row
// serialization bottleneck at hidden>=512 — Round-28 LEARN M2). The reduction
// order differs from the serial sum only at ~ulp level, inside the parity
// tolerance.
__global__ void ln_forward_trainable_kernel(
    const float* x, const float* gamma, const float* beta, float* y, int m,
    int h, float eps) {
    const int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + static_cast<size_t>(row) * h;
    float* yr = y + static_cast<size_t>(row) * h;
    __shared__ float red[kLnBlock / 32];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < h; j += kLnBlock) sum += xr[j];
    const float mean = ln_block_reduce_sum(sum, red) / h;
    float var = 0.0f;
    for (int j = threadIdx.x; j < h; j += kLnBlock) {
        const float d = xr[j] - mean;
        var += d * d;
    }
    var = ln_block_reduce_sum(var, red);
    const float inv = rsqrtf(var / h + eps);
    for (int j = threadIdx.x; j < h; j += kLnBlock) {
        const float xhat = (xr[j] - mean) * inv;
        yr[j] = gamma[j] * xhat + beta[j];
    }
}

// Backward pass 1 (one block per row): row mean/var/inv + the row sums
// sum_dxhat and sum_dxhat*xhat that dL/dx needs (stats[4*row …]), and the
// dgamma/dbeta row contributions accumulated with atomics (the affine grads
// are sums over the batch rows; float atomicAdd order only perturbs the sum at
// ~ulp level, far inside the parity tolerance). Stats are block reductions
// (v2.29, TASK-045), same ~ulp deviation budget as forward.
__global__ void ln_backward_stats_kernel(
    const float* x, const float* gamma, const float* dy, float* dgamma,
    float* dbeta, float* stats, int m, int h, float eps) {
    const int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + static_cast<size_t>(row) * h;
    const float* dyr = dy + static_cast<size_t>(row) * h;
    __shared__ float red[kLnBlock / 32];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < h; j += kLnBlock) sum += xr[j];
    const float mean = ln_block_reduce_sum(sum, red) / h;
    float var = 0.0f;
    for (int j = threadIdx.x; j < h; j += kLnBlock) {
        const float d = xr[j] - mean;
        var += d * d;
    }
    var = ln_block_reduce_sum(var, red);
    const float inv = rsqrtf(var / h + eps);
    float sdx = 0.0f, sdxh = 0.0f;
    for (int j = threadIdx.x; j < h; j += kLnBlock) {
        const float xhat = (xr[j] - mean) * inv;
        const float dxhat = dyr[j] * gamma[j];
        sdx += dxhat;
        sdxh += dxhat * xhat;
        atomicAdd(&dgamma[j], dyr[j] * xhat);
        atomicAdd(&dbeta[j], dyr[j]);
    }
    sdx = ln_block_reduce_sum(sdx, red);
    sdxh = ln_block_reduce_sum(sdxh, red);
    float* s = stats + static_cast<size_t>(row) * 4;
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
    if (idx >= static_cast<int64_t>(m) * h) return;
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

// ceil(batch*hidden / 256) grid count computed in 64-bit so the product can't
// wrap (a wrapped grid would launch too few blocks and silently compute nothing
// for the remaining elements).
int layer_norm_bwd_grid(int m, int h) {
    const int64_t total = static_cast<int64_t>(m) * h;
    const int64_t grid = (total + 255) / 256;
    if (grid > static_cast<int64_t>(2147483647)) {
        throw std::invalid_argument(
            "layer_norm_backward: batch * hidden exceeds the gridDim.x max");
    }
    return static_cast<int>(grid);
}

}  // namespace

void layer_norm_backward(
    const float* input, const float* gamma, const float* grad_output,
    float* grad_input, float* grad_gamma, float* grad_beta, int batch_size,
    int normalized_shape, float eps, cudaStream_t stream) {
    // Routes through the v2.29 device entry (block-reduction kernels). This
    // stateless free function allocates its per-call stats scratch inside the
    // detail:: entry (the M1 per-call-buffer debt); the LayerNorm class keeps
    // its cached stats buffer below.
    detail::layer_norm_backward_trainable(
        input, gamma, grad_output, grad_input, grad_gamma, grad_beta,
        batch_size, normalized_shape, eps, stream);
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
    detail::layer_norm_forward_trainable(input, d_gamma_->data(),
                                         d_beta_->data(), output, batch,
                                         hidden_, eps_, stream);
}

void LayerNorm::backward(const float* input, const float* grad_output,
                         float* grad_input, float* grad_gamma,
                         float* grad_beta, int batch, cudaStream_t stream) {
    ensure_stats(batch);
    CUDA_CHECK(cudaMemsetAsync(grad_gamma, 0, sizeof(float) * hidden_, stream));
    CUDA_CHECK(cudaMemsetAsync(grad_beta, 0, sizeof(float) * hidden_, stream));
    ln_backward_stats_kernel<<<batch, kLnBlock, 0, stream>>>(
        input, d_gamma_->data(), grad_output, grad_gamma, grad_beta,
        d_stats_->data(), batch, hidden_, eps_);
    const int grid = layer_norm_bwd_grid(batch, hidden_);
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

namespace detail {

void layer_norm_forward_trainable(const float* x, const float* gamma,
                                  const float* beta, float* y, int m, int h,
                                  float eps, cudaStream_t stream) {
    if (m <= 0 || h <= 0) {
        throw std::invalid_argument(
            "layer_norm_forward_trainable requires positive m and h");
    }
    ln_forward_trainable_kernel<<<m, kLnBlock, 0, stream>>>(x, gamma, beta, y,
                                                            m, h, eps);
    CUDA_CHECK(cudaGetLastError());
}

void layer_norm_backward_trainable(const float* x, const float* gamma,
                                   const float* dy, float* dx, float* dgamma,
                                   float* dbeta, int m, int h, float eps,
                                   cudaStream_t stream) {
    if (m <= 0 || h <= 0) {
        throw std::invalid_argument(
            "layer_norm_backward_trainable requires positive m and h");
    }
    cuda::memory::Buffer<float> stats(static_cast<size_t>(m) * 4);
    CUDA_CHECK(cudaMemsetAsync(dgamma, 0, sizeof(float) * h, stream));
    CUDA_CHECK(cudaMemsetAsync(dbeta, 0, sizeof(float) * h, stream));
    ln_backward_stats_kernel<<<m, kLnBlock, 0, stream>>>(
        x, gamma, dy, dgamma, dbeta, stats.data(), m, h, eps);
    const int grid = layer_norm_bwd_grid(m, h);
    ln_backward_input_kernel<<<grid, 256, 0, stream>>>(
        x, gamma, dy, stats.data(), dx, m, h);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace detail

}  // namespace cuda::neural
