/**
 * @file fp8_kernels.hpp
 * @brief FP8 quantization CUDA kernels
 * @defgroup fp8_kernels FP8 Kernels
 * @ingroup quantize
 *
 * Provides CUDA kernels for FP8 quantization and dequantization.
 *
 * @note Uses hardware-accelerated conversion on H100/H200
 * @see fp8_types.hpp For type definitions
 */

#ifndef NOVA_CUDA_QUANTIZE_FP8_KERNELS_HPP
#define NOVA_CUDA_QUANTIZE_FP8_KERNELS_HPP

#include <cuda/quantize/fp8_types.hpp>
#include <cuda_runtime.h>

namespace nova {
    namespace quantize {
        namespace cuda {

            /**
             * @brief Quantize float32 to FP8E4M3
             * @param src Source float32 array
             * @param[out] dst Destination FP8E4M3 array
             * @param n Number of elements
             * @param stream CUDA stream
             */
            void quantize_f32_to_fp8e4m3(const float* src, FP8E4M3* dst, size_t n, cudaStream_t stream = 0);

            /**
             * @brief Quantize float32 to FP8E5M2
             * @param src Source float32 array
             * @param[out] dst Destination FP8E5M2 array
             * @param n Number of elements
             * @param stream CUDA stream
             */
            void quantize_f32_to_fp8e5m2(const float* src, FP8E5M2* dst, size_t n, cudaStream_t stream = 0);

            /**
             * @brief Dequantize FP8E4M3 to float32
             * @param src Source FP8E4M3 array
             * @param[out] dst Destination float32 array
             * @param n Number of elements
             * @param scale Scale factor
             * @param stream CUDA stream
             */
            void dequantize_fp8e4m3_to_f32(const FP8E4M3* src, float* dst, size_t n, float scale, cudaStream_t stream = 0);

            void dequantize_fp8e5m2_to_f32(const FP8E5M2* src, float* dst, size_t n, float scale, cudaStream_t stream = 0);

            void quantize_batched_f32_to_fp8e4m3(const float** src, FP8E4M3** dst, int batch_size, const size_t* sizes, const float* scales, cudaStream_t stream = 0);

            void dequantize_batched_fp8e4m3_to_f32(const FP8E4M3** src, float** dst, int batch_size, const size_t* sizes, const float* scales, cudaStream_t stream = 0);

        }  // namespace cuda
    }  // namespace quantize
}  // namespace nova

#endif  // NOVA_CUDA_QUANTIZE_FP8_KERNELS_HPP
