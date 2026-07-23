#include "cuda/neural/fusion/kernel_fusion.h"

#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace cuda::neural::fusion {

namespace {

template <typename T>
__global__ void matmul_bias_kernel(
    const T* matmul_out,
    const T* bias,
    T* output,
    int batch_size,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features;
    if (idx >= n) return;

    output[idx] = matmul_out[idx] + bias[idx % num_features];
}

template <typename T>
__global__ void matmul_bias_relu_kernel(
    const T* matmul_out,
    const T* bias,
    T* output,
    int batch_size,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features;
    if (idx >= n) return;

    T val = matmul_out[idx] + bias[idx % num_features];
    output[idx] = val > 0 ? val : 0;
}

template <typename T>
__global__ void matmul_bias_sigmoid_kernel(
    const T* matmul_out,
    const T* bias,
    T* output,
    int batch_size,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features;
    if (idx >= n) return;

    T val = matmul_out[idx] + bias[idx % num_features];
    output[idx] = 1.0f / (1.0f + expf(-val));
}

template <typename T>
__global__ void compute_row_sum_kernel(
    const T* input,
    T* sum,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int b = blockIdx.x;
    int i = blockIdx.y;

    if (b >= batch_size || i >= seq_len) return;

    int idx = b * seq_len + i;
    T local_sum = 0;
    int offset = idx * hidden_size;

    for (int j = 0; j < hidden_size; ++j) {
        local_sum += expf(input[offset + j]);
    }

    if (threadIdx.x == 0) {
        sum[idx] = local_sum;
    }
}

template <typename T>
__global__ void fused_layernorm_softmax_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    const T* row_sum,
    int batch_size,
    int seq_len,
    int hidden_size,
    T eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * seq_len * hidden_size;
    if (idx >= n) return;

    int b = idx / (seq_len * hidden_size);
    int i = (idx / hidden_size) % seq_len;
    int j = idx % hidden_size;

    T mean = 0;
    for (int k = 0; k < hidden_size; ++k) {
        mean += input[(b * seq_len + i) * hidden_size + k];
    }
    mean /= hidden_size;

    T variance = 0;
    for (int k = 0; k < hidden_size; ++k) {
        T diff = input[(b * seq_len + i) * hidden_size + k] - mean;
        variance += diff * diff;
    }
    variance /= hidden_size;

    T std = sqrtf(variance + eps);
    T normalized = (input[idx] - mean) / std;
    T layernormed = gamma[j] * normalized + beta[j];

    output[idx] = expf(layernormed) / row_sum[b * seq_len + i];
}

} // anonymous namespace

FusionKernelRegistry& FusionKernelRegistry::instance() {
    static FusionKernelRegistry registry;
    return registry;
}

void FusionKernelRegistry::register_fusion(const std::string& name, bool enabled) {
    if (name == "matmul_bias") matmul_bias_enabled_ = enabled;
    else if (name == "matmul_bias_activation") matmul_bias_activation_enabled_ = enabled;
    else if (name == "layernorm_softmax") layernorm_softmax_enabled_ = enabled;
}

void FusionKernelRegistry::enable_fusion(const std::string& name) {
    register_fusion(name, true);
}

void FusionKernelRegistry::disable_fusion(const std::string& name) {
    register_fusion(name, false);
}

bool FusionKernelRegistry::is_fusion_enabled(const std::string& name) const {
    if (name == "matmul_bias") return matmul_bias_enabled_;
    else if (name == "matmul_bias_activation") return matmul_bias_activation_enabled_;
    else if (name == "layernorm_softmax") return layernorm_softmax_enabled_;
    return false;
}

void FusionKernelRegistry::set_config(const FusionConfig& config) {
    config_ = config;
}

FusionConfig FusionKernelRegistry::get_config() const {
    return config_;
}

void fused_matmul_bias(
    const float* matmul_out,
    const float* bias,
    float* output,
    int batch_size,
    int num_features,
    cudaStream_t stream
) {
    int n = batch_size * num_features;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    matmul_bias_kernel<float><<<grid_size, block_size, 0, stream>>>(
        matmul_out, bias, output, batch_size, num_features);
}

void fused_matmul_bias_relu(
    const float* matmul_out,
    const float* bias,
    float* output,
    int batch_size,
    int num_features,
    cudaStream_t stream
) {
    int n = batch_size * num_features;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    matmul_bias_relu_kernel<float><<<grid_size, block_size, 0, stream>>>(
        matmul_out, bias, output, batch_size, num_features);
}

void fused_matmul_bias_sigmoid(
    const float* matmul_out,
    const float* bias,
    float* output,
    int batch_size,
    int num_features,
    cudaStream_t stream
) {
    int n = batch_size * num_features;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    matmul_bias_sigmoid_kernel<float><<<grid_size, block_size, 0, stream>>>(
        matmul_out, bias, output, batch_size, num_features);
}

FusedLayerNormSoftmax::FusedLayerNormSoftmax(int hidden_size, float eps)
    : hidden_size_(hidden_size), eps_(eps) {}

FusedLayerNormSoftmax::~FusedLayerNormSoftmax() {}

void FusedLayerNormSoftmax::forward(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    cudaStream_t stream
) {
    float* row_sum;
    CUDA_CHECK(cudaMalloc(&row_sum, batch_size * seq_len * sizeof(float)));

    dim3 block(256);
    dim3 grid(batch_size, seq_len);
    compute_row_sum_kernel<float><<<grid, block, 0, stream>>>(
        input, row_sum, batch_size, seq_len, hidden_size_);

    int n = batch_size * seq_len * hidden_size_;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    fused_layernorm_softmax_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input, gamma, beta, output, row_sum,
        batch_size, seq_len, hidden_size_, eps_);

    CUDA_CHECK(cudaFree(row_sum));
}

bool should_use_fused_kernel(const std::string& op, int batch_size, int num_features) {
    auto& registry = FusionKernelRegistry::instance();
    auto config = registry.get_config();

    int total_elements = batch_size * num_features;
    if (total_elements < config.min_elements_for_fusion) {
        return false;
    }

    if (op == "matmul_bias" && config.fuse_matmul_bias) {
        return registry.is_fusion_enabled("matmul_bias");
    }
    else if (op == "matmul_bias_activation" && config.fuse_matmul_bias_activation) {
        return registry.is_fusion_enabled("matmul_bias_activation");
    }
    else if (op == "layernorm_softmax" && config.fuse_layernorm_softmax) {
        return registry.is_fusion_enabled("layernorm_softmax");
    }

    return false;
}

} // namespace cuda::neural::fusion
