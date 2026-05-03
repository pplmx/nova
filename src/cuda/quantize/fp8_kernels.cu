#include <cuda/quantize/fp8_kernels.hpp>
#include <cuda_runtime.h>
#include <cstdio>

namespace nova {
namespace quantize {
namespace cuda {

namespace detail {

template<typename DstT>
__global__ void quantize_f32_kernel(const float* src, DstT* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = DstT(src[idx]);
    }
}

template<typename SrcT>
__global__ void dequantize_fp8_kernel(const SrcT* src, float* dst, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]) * scale;
    }
}

} // namespace detail

void quantize_f32_to_fp8e4m3(const float* src, FP8E4M3* dst, size_t n, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    detail::quantize_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_f32_to_fp8e4m3: %s\n", cudaGetErrorString(err));
    }
}

void quantize_f32_to_fp8e5m2(const float* src, FP8E5M2* dst, size_t n, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    detail::quantize_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_f32_to_fp8e5m2: %s\n", cudaGetErrorString(err));
    }
}

void dequantize_fp8e4m3_to_f32(const FP8E4M3* src, float* dst, size_t n, float scale, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    detail::dequantize_fp8_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, n, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_fp8e4m3_to_f32: %s\n", cudaGetErrorString(err));
    }
}

void dequantize_fp8e5m2_to_f32(const FP8E5M2* src, float* dst, size_t n, float scale, cudaStream_t stream) {
    constexpr size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    detail::dequantize_fp8_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, n, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_fp8e5m2_to_f32: %s\n", cudaGetErrorString(err));
    }
}

namespace detail {

__global__ void quantize_batched_f32_kernel(
    const float** src, FP8E4M3** dst, int batch_size, const size_t* sizes, const float* scales) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    size_t n = sizes[batch_idx];
    float scale = scales[batch_idx];

    const float* src_batch = src[batch_idx];
    FP8E4M3* dst_batch = dst[batch_idx];

    size_t tid = threadIdx.x;
    size_t block_size = blockDim.x;

    for (size_t idx = tid; idx < n; idx += block_size) {
        float val = src_batch[idx] * scale;
        dst_batch[idx] = FP8E4M3(val);
    }
}

__global__ void dequantize_batched_fp8_kernel(
    const FP8E4M3** src, float** dst, int batch_size, const size_t* sizes, const float* scales) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    size_t n = sizes[batch_idx];
    float scale = scales[batch_idx];

    const FP8E4M3* src_batch = src[batch_idx];
    float* dst_batch = dst[batch_idx];

    size_t tid = threadIdx.x;
    size_t block_size = blockDim.x;

    for (size_t idx = tid; idx < n; idx += block_size) {
        dst_batch[idx] = static_cast<float>(src_batch[idx]) * scale;
    }
}

} // namespace detail

void quantize_batched_f32_to_fp8e4m3(
    const float** src, FP8E4M3** dst, int batch_size, const size_t* sizes,
    const float* scales, cudaStream_t stream) {

    detail::quantize_batched_f32_kernel<<<batch_size, 256, 0, stream>>>(
        src, dst, batch_size, sizes, scales);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_batched_f32_to_fp8e4m3: %s\n", cudaGetErrorString(err));
    }
}

void dequantize_batched_fp8e4m3_to_f32(
    const FP8E4M3** src, float** dst, int batch_size, const size_t* sizes,
    const float* scales, cudaStream_t stream) {

    detail::dequantize_batched_fp8_kernel<<<batch_size, 256, 0, stream>>>(
        src, dst, batch_size, sizes, scales);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_batched_fp8e4m3_to_f32: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace quantize
} // namespace nova
