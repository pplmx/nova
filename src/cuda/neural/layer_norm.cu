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

__global__ void layer_norm_inference_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int normalized_shape,
    float eps,
    const float* mean,
    const float* variance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * normalized_shape) return;

    int batch_idx = idx / normalized_shape;
    int inner_idx = idx % normalized_shape;

    float mean_val = mean[batch_idx];
    float var_val = variance[batch_idx];
    float inv_std = rsqrtf(var_val + eps);

    float normalized = (input[idx] - mean_val) * inv_std;
    if (gamma && beta) {
        output[idx] = gamma[inner_idx] * normalized + beta[inner_idx];
    } else {
        output[idx] = normalized;
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
    int block_size = 256;
    int grid_size = (batch_size * normalized_shape + block_size - 1) / block_size;

    layer_norm_inference_kernel<<<grid_size, block_size, 0, stream>>>(
        input, gamma, beta, output,
        batch_size, normalized_shape, eps,
        nullptr, nullptr
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// v2.28 trainable LayerNorm. P1-RED: provisional throwing stubs — the parity
// tests pin the forward/backward/step contracts and go RED here; the real
// device kernels land in P2 (TASK-041).
// ============================================================================

void layer_norm_backward(
    const float*, const float*, const float*, float*, float*, float*, int,
    int, float, cudaStream_t) {
    throw std::runtime_error("layer_norm_backward: not implemented (P1 RED stub)");
}

LayerNorm::LayerNorm(int hidden, float eps) : hidden_(hidden), eps_(eps) {
    if (hidden_ <= 0) {
        throw std::invalid_argument("LayerNorm: hidden must be positive");
    }
    d_gamma_ = std::make_unique<cuda::memory::Buffer<float>>(hidden_);
    d_beta_ = std::make_unique<cuda::memory::Buffer<float>>(hidden_);
}

LayerNorm::~LayerNorm() = default;

void LayerNorm::set_weight(const float*, const float*) {
    throw std::runtime_error("LayerNorm::set_weight: not implemented (P1 RED stub)");
}

void LayerNorm::forward(const float*, float*, int, cudaStream_t) {
    throw std::runtime_error("LayerNorm::forward: not implemented (P1 RED stub)");
}

void LayerNorm::backward(const float*, const float*, float*, float*, float*,
                         int, cudaStream_t) {
    throw std::runtime_error("LayerNorm::backward: not implemented (P1 RED stub)");
}

void LayerNorm::copy_weights(float*, float*) const {
    throw std::runtime_error("LayerNorm::copy_weights: not implemented (P1 RED stub)");
}

void LayerNorm::step(optimizers::AdamWOptimizer&, optimizers::AdamWOptimizer&,
                     const float*, const float*, int, cudaStream_t) {
    throw std::runtime_error("LayerNorm::step: not implemented (P1 RED stub)");
}

int LayerNorm::hidden() const { return hidden_; }
float LayerNorm::eps() const { return eps_; }

}  // namespace cuda::neural
