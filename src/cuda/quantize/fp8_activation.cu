/**
 * @file fp8_activation.cu
 * @brief FP8 activation kernels and their host-callable entry points
 *
 * These used to live in fp8_activation.hpp behind `#ifdef __CUDACC__`. A plain
 * .cpp translation unit therefore compiled the entry points with an empty body,
 * so calling cuda::relu() from host code linked and returned without writing the
 * output buffer. Keeping the definitions in a CUDA translation unit gives every
 * caller the real implementation.
 */

#include <cuda/quantize/fp8_activation.hpp>

#include <cstdio>

namespace nova {
namespace quantize {
namespace cuda {

namespace {

template<typename T>
__global__ void relu_kernel(const T* input, T* output, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::relu_fp8(input[idx]);
    }
}

template<typename T>
__global__ void gelu_kernel(const T* input, T* output, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::gelu_fp8(input[idx]);
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::sigmoid_fp8(input[idx]);
    }
}

template<typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, std::size_t n,
                                 float alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = activation::leaky_relu_fp8(input[idx], alpha);
    }
}

constexpr std::size_t kBlockSize = 256;

inline std::size_t grid_for(std::size_t n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

/// True when the launch has nothing to do. A zero-sized grid is rejected by CUDA
/// with "invalid configuration argument", which report_launch_error only prints to
/// stderr — so an empty tensor produced a launch failure that no caller could
/// observe through cudaGetLastError, because reading the error clears it.
inline bool empty_launch(std::size_t n) { return n == 0; }

inline void report_launch_error(const char* name) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", name, cudaGetErrorString(err));
    }
}

}  // namespace

template<typename T>
void relu(const T* input, T* output, std::size_t n, cudaStream_t stream) {
    if (empty_launch(n)) {
        return;
    }
    relu_kernel<<<grid_for(n), kBlockSize, 0, stream>>>(input, output, n);
    report_launch_error("relu");
}

template<typename T>
void gelu(const T* input, T* output, std::size_t n, cudaStream_t stream) {
    if (empty_launch(n)) {
        return;
    }
    gelu_kernel<<<grid_for(n), kBlockSize, 0, stream>>>(input, output, n);
    report_launch_error("gelu");
}

template<typename T>
void sigmoid(const T* input, T* output, std::size_t n, cudaStream_t stream) {
    if (empty_launch(n)) {
        return;
    }
    sigmoid_kernel<<<grid_for(n), kBlockSize, 0, stream>>>(input, output, n);
    report_launch_error("sigmoid");
}

template<typename T>
void leaky_relu(const T* input, T* output, std::size_t n, float alpha,
                cudaStream_t stream) {
    if (empty_launch(n)) {
        return;
    }
    leaky_relu_kernel<<<grid_for(n), kBlockSize, 0, stream>>>(input, output, n, alpha);
    report_launch_error("leaky_relu");
}

template void relu<FP8E4M3>(const FP8E4M3*, FP8E4M3*, std::size_t, cudaStream_t);
template void relu<FP8E5M2>(const FP8E5M2*, FP8E5M2*, std::size_t, cudaStream_t);
template void gelu<FP8E4M3>(const FP8E4M3*, FP8E4M3*, std::size_t, cudaStream_t);
template void gelu<FP8E5M2>(const FP8E5M2*, FP8E5M2*, std::size_t, cudaStream_t);
template void sigmoid<FP8E4M3>(const FP8E4M3*, FP8E4M3*, std::size_t, cudaStream_t);
template void sigmoid<FP8E5M2>(const FP8E5M2*, FP8E5M2*, std::size_t, cudaStream_t);
template void leaky_relu<FP8E4M3>(const FP8E4M3*, FP8E4M3*, std::size_t, float,
                                  cudaStream_t);
template void leaky_relu<FP8E5M2>(const FP8E5M2*, FP8E5M2*, std::size_t, float,
                                  cudaStream_t);

}  // namespace cuda
}  // namespace quantize
}  // namespace nova
