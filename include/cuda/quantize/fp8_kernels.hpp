#ifndef NOVA_CUDA_QUANTIZE_FP8_KERNELS_HPP
#define NOVA_CUDA_QUANTIZE_FP8_KERNELS_HPP

#include <cuda/quantize/fp8_types.hpp>
#include <cuda_runtime.h>

namespace nova {
namespace quantize {
namespace cuda {

void quantize_f32_to_fp8e4m3(const float* src, FP8E4M3* dst, size_t n, cudaStream_t stream = 0);

void quantize_f32_to_fp8e5m2(const float* src, FP8E5M2* dst, size_t n, cudaStream_t stream = 0);

void dequantize_fp8e4m3_to_f32(const FP8E4M3* src, float* dst, size_t n, float scale, cudaStream_t stream = 0);

void dequantize_fp8e5m2_to_f32(const FP8E5M2* src, float* dst, size_t n, float scale, cudaStream_t stream = 0);

void quantize_batched_f32_to_fp8e4m3(
    const float** src, FP8E4M3** dst, int batch_size, const size_t* sizes,
    const float* scales, cudaStream_t stream = 0);

void dequantize_batched_fp8e4m3_to_f32(
    const FP8E4M3** src, float** dst, int batch_size, const size_t* sizes,
    const float* scales, cudaStream_t stream = 0);

} // namespace cuda
} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_FP8_KERNELS_HPP
