#ifndef NOVA_CUDA_QUANTIZE_FP8_ACTIVATION_HPP
#define NOVA_CUDA_QUANTIZE_FP8_ACTIVATION_HPP

#include <cuda/quantize/fp8_types.hpp>
#include <cuda_runtime.h>
#include <cmath>

namespace nova {
namespace quantize {
namespace cuda {

namespace activation {

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float geluf(float x) {
    constexpr float sqrt_2_inv = 0.7071067811865476f;
    constexpr float half = 0.5f;
    float cdf = half * (1.0f + erff(x * sqrt_2_inv));
    return x * cdf;
}

template<typename T>
__device__ __forceinline__ T relu_fp8(T x) {
    if (static_cast<float>(x) < 0.0f) {
        return T(0.0f);
    }
    return x;
}

template<typename T>
__device__ __forceinline__ T gelu_fp8(T x) {
    float f = static_cast<float>(x);
    float result = geluf(f);
    return T(result);
}

template<typename T>
__device__ __forceinline__ T sigmoid_fp8(T x) {
    float f = static_cast<float>(x);
    float result = sigmoidf(f);
    return T(result);
}

template<typename T>
__device__ __forceinline__ T leaky_relu_fp8(T x, float alpha = 0.01f) {
    float f = static_cast<float>(x);
    float result = f >= 0.0f ? f : alpha * f;
    return T(result);
}

} // namespace activation

template<typename T>
__global__ void relu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::relu_fp8(input[idx]);
    }
}

template<typename T>
__global__ void gelu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::gelu_fp8(input[idx]);
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::sigmoid_fp8(input[idx]);
    }
}

template<typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, size_t n, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::leaky_relu_fp8(input[idx], alpha);
    }
}

template<typename T>
void relu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void gelu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void sigmoid(const T* input, T* output, size_t n, cudaStream_t stream = 0);

template<typename T>
void leaky_relu(const T* input, T* output, size_t n, float alpha = 0.01f, cudaStream_t stream = 0);

template<typename T>
T relu_fp8_scalar(T x) {
    return activation::relu_fp8(x);
}

template<typename T>
T gelu_fp8_scalar(T x) {
    return activation::gelu_fp8(x);
}

template void relu<FP8E4M3>(const FP8E4M3*, FP8E4M3*, size_t, cudaStream_t);
template void relu<FP8E5M2>(const FP8E5M2*, FP8E5M2*, size_t, cudaStream_t);
template void gelu<FP8E4M3>(const FP8E4M3*, FP8E4M3*, size_t, cudaStream_t);
template void gelu<FP8E5M2>(const FP8E5M2*, FP8E5M2*, size_t, cudaStream_t);
template void sigmoid<FP8E4M3>(const FP8E4M3*, FP8E4M3*, size_t, cudaStream_t);
template void sigmoid<FP8E5M2>(const FP8E5M2*, FP8E5M2*, size_t, cudaStream_t);
template void leaky_relu<FP8E4M3>(const FP8E4M3*, FP8E4M3*, size_t, float, cudaStream_t);
template void leaky_relu<FP8E5M2>(const FP8E5M2*, FP8E5M2*, size_t, float, cudaStream_t);

template<typename T>
void relu(const T* input, T* output, size_t n, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in relu: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void gelu(const T* input, T* output, size_t n, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    gelu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gelu: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void sigmoid(const T* input, T* output, size_t n, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in sigmoid: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void leaky_relu(const T* input, T* output, size_t n, float alpha, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n, alpha);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in leaky_relu: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_FP8_ACTIVATION_HPP
