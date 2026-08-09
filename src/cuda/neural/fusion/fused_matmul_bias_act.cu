#include "cuda/neural/fusion/fused_matmul_bias_act.h"
#include "cuda/neural/fusion/kernel_fusion.h"
#include "cuda/neural/matmul.h"

#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <limits>

namespace cuda::neural::fusion {

namespace {

template <typename T>
__global__ void apply_bias_kernel(T* data, const T* bias, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] += bias[idx % (blockDim.x * gridDim.x)];
    }
}

template <typename T>
__global__ void apply_bias_kernel_row(T* data, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        data[idx] += bias[col];
    }
}

}  // namespace

FusedMatmulBiasAct::FusedMatmulBiasAct(const MatmulBiasActConfig& config)
    : config_(config) {
    // Passing a valid out-pointer: cudaGetDevice(nullptr) returns
    // cudaErrorInvalidValue AND leaves a sticky last-error that breaks the
    // CUDA error state for every subsequent cudaGetLastError() check in the
    // process (observed poisoning FusedMatmulBiasAct tests and cascading into
    // later suites). We only want a driver-availability probe here.
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    cuda_fusion_available_ = (err == cudaSuccess);

    if (config_.use_cuda_fusion && cuda_fusion_available_) {
        CUBLAS_CHECK(cublasLtCreate(&lt_handle_));
    }

    if (config_.max_workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_, config_.max_workspace_bytes));
    }
}

FusedMatmulBiasAct::~FusedMatmulBiasAct() {
    if (lt_handle_) {
        cublasLtDestroy(lt_handle_);
    }
    if (workspace_) {
        cudaFree(workspace_);
    }
}

void FusedMatmulBiasAct::forward(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int m,
    int n,
    int k,
    cudaStream_t stream
) {
    if (config_.use_cuda_fusion && cuda_fusion_available_) {
        forward_fallback(A, B, bias, C, m, n, k, stream);
    } else {
        forward_fallback(A, B, bias, C, m, n, k, stream);
    }
}

void FusedMatmulBiasAct::forward_fallback(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int m,
    int n,
    int k,
    cudaStream_t stream
) {
    cublasHandle_t handle = config_.handle;
    if (!handle) {
        handle = cuda::neural::get_cublas_handle();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (stream) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n,
        A, k,
        &beta,
        C, n
    ));

    int num_elements = m * n;
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    if (stream) {
        apply_bias_kernel_row<float><<<grid_size, block_size, 0, stream>>>(
            C, bias, m, n);
    } else {
        apply_bias_kernel_row<float><<<grid_size, block_size>>>(
            C, bias, m, n);
    }

    if (config_.activation != ActivationType::None) {
        if (stream) {
            apply_activation<float>(C, num_elements, config_.activation,
                                   config_.relu_threshold, stream);
        } else {
            apply_activation<float>(C, num_elements, config_.activation,
                                   config_.relu_threshold, nullptr);
        }
    }
}

void FusedMatmulBiasAct::set_activation(ActivationType activation) {
    config_.activation = activation;
}

ActivationType FusedMatmulBiasAct::get_activation() const {
    return config_.activation;
}

void FusedMatmulBiasAct::enable_cuda_fusion() {
    config_.use_cuda_fusion = true;
}

void FusedMatmulBiasAct::disable_cuda_fusion() {
    config_.use_cuda_fusion = false;
}

bool FusedMatmulBiasAct::is_cuda_fusion_enabled() const {
    return config_.use_cuda_fusion && cuda_fusion_available_;
}

FusionPolicyManager& FusionPolicyManager::instance() {
    static FusionPolicyManager instance;
    return instance;
}

void FusionPolicyManager::set_policy(const FusionPolicy& policy) {
    policy_ = policy;
}

FusionPolicy FusionPolicyManager::get_policy() const {
    return policy_;
}

bool FusionPolicyManager::should_fuse(
    const std::string& op_name,
    int element_count
) const {
    if (element_count < policy_.min_elements_for_fusion) {
        return false;
    }

    if (op_name == "matmul_bias" && policy_.fuse_matmul_bias) {
        return true;
    }
    if (op_name == "matmul_bias_activation" && policy_.fuse_matmul_bias_activation) {
        return true;
    }
    if (op_name == "layernorm_softmax" && policy_.fuse_layernorm_softmax) {
        return true;
    }

    return false;
}

void FusionPolicyManager::enable_op(const std::string& op_name) {
    if (op_name == "matmul_bias") {
        policy_.fuse_matmul_bias = true;
    } else if (op_name == "matmul_bias_activation") {
        policy_.fuse_matmul_bias_activation = true;
    } else if (op_name == "layernorm_softmax") {
        policy_.fuse_layernorm_softmax = true;
    }
}

void FusionPolicyManager::disable_op(const std::string& op_name) {
    if (op_name == "matmul_bias") {
        policy_.fuse_matmul_bias = false;
    } else if (op_name == "matmul_bias_activation") {
        policy_.fuse_matmul_bias_activation = false;
    } else if (op_name == "layernorm_softmax") {
        policy_.fuse_layernorm_softmax = false;
    }
}

std::string FusionPolicyManager::get_policy_summary() const {
    return "FusionPolicy{matmul_bias=" + std::to_string(policy_.fuse_matmul_bias) +
           ", matmul_bias_activation=" + std::to_string(policy_.fuse_matmul_bias_activation) +
           ", layernorm_softmax=" + std::to_string(policy_.fuse_layernorm_softmax) + "}";
}

template <typename T>
__global__ void relu_kernel(T* data, size_t num_elements, T threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = max(data[idx], threshold);
    }
}

template <typename T>
__global__ void sigmoid_kernel(T* data, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = static_cast<T>(1.0f) / (static_cast<T>(1.0f) + exp(-data[idx]));
    }
}

template <typename T>
__global__ void tanh_kernel(T* data, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = tanh(data[idx]);
    }
}

template <typename T>
__global__ void gelu_kernel(T* data, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        T x = data[idx];
        T cdf = 0.5f * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        data[idx] = x * cdf;
    }
}

template <typename T>
void apply_activation(
    T* data,
    size_t num_elements,
    ActivationType activation,
    float threshold,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    if (stream) {
        switch (activation) {
            case ActivationType::ReLU:
                relu_kernel<T><<<grid_size, block_size, 0, stream>>>(
                    data, num_elements, static_cast<T>(threshold));
                break;
            case ActivationType::Sigmoid:
                sigmoid_kernel<T><<<grid_size, block_size, 0, stream>>>(
                    data, num_elements);
                break;
            case ActivationType::Tanh:
                tanh_kernel<T><<<grid_size, block_size, 0, stream>>>(
                    data, num_elements);
                break;
            case ActivationType::GELU:
                gelu_kernel<T><<<grid_size, block_size, 0, stream>>>(
                    data, num_elements);
                break;
            case ActivationType::None:
                break;
        }
    } else {
        switch (activation) {
            case ActivationType::ReLU:
                relu_kernel<T><<<grid_size, block_size>>>(
                    data, num_elements, static_cast<T>(threshold));
                break;
            case ActivationType::Sigmoid:
                sigmoid_kernel<T><<<grid_size, block_size>>>(
                    data, num_elements);
                break;
            case ActivationType::Tanh:
                tanh_kernel<T><<<grid_size, block_size>>>(
                    data, num_elements);
                break;
            case ActivationType::GELU:
                gelu_kernel<T><<<grid_size, block_size>>>(
                    data, num_elements);
                break;
            case ActivationType::None:
                break;
        }
    }
}

template void apply_activation<float>(
    float*, size_t, ActivationType, float, cudaStream_t);

}  // namespace cuda::neural::fusion
