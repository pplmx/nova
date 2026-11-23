/**
 * @file int8_kernels.hpp
 * @brief INT8 quantization CUDA kernels
 * @defgroup int8_kernels INT8 Kernels
 * @ingroup quantize
 *
 * Provides CUDA kernels for INT8 quantization and dequantization operations.
 *
 * @note Optimized for NVIDIA Tensor Cores
 * @see quantize_tensor.hpp For type definitions
 */

#ifndef NOVA_CUDA_QUANTIZE_INT8_KERNELS_HPP
#define NOVA_CUDA_QUANTIZE_INT8_KERNELS_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace nova {
    namespace quantize {
        namespace cuda {

            /**
             * @brief Parameters for INT8 quantization
             * @struct QuantizationParams
             * @ingroup int8_kernels
             */
            struct QuantizationParams {
                /** @brief Quantization scale */
                float scale {1.0f};

                /**
                 * @brief Zero point for asymmetric quantization, expressed in the DATA's units
                 *
                 * The kernels compute
                 *
                 *     quantize:    q = round((v - zero_point) / scale)
                 *     dequantize:  v = q * scale + zero_point
                 *
                 * so zero_point is a value in the signal's own range, not an integer code. For a
                 * calibrated asymmetric range it is the centre of [min, max], which maps min to
                 * code -127 and max to +127.
                 *
                 * This is stated explicitly because three different conventions were previously in
                 * use (this one, an integer zero point paired with q = v/scale + zp, and a third
                 * in the calibrators) and asymmetric round trips did not return their input.
                 */
                float zero_point {0.0f};

                /** @brief Use symmetric quantization (no zero point) */
                bool symmetric {true};

                /** @brief Default constructor */
                QuantizationParams() = default;

                /**
                 * @brief Construct quantization params
                 * @param s Scale value
                 * @param zp Zero point
                 * @param sym Symmetric flag
                 */
                QuantizationParams(float s, float zp = 0.0f, bool sym = true)
                    : scale(s),
                      zero_point(zp),
                      symmetric(sym) {}
            };

            void quantize_f32_to_int8(const float* src, int8_t* dst, size_t n, QuantizationParams params, cudaStream_t stream = 0);

            void dequantize_int8_to_f32(const int8_t* src, float* dst, size_t n, QuantizationParams params, cudaStream_t stream = 0);

            void quantize_f32_to_int8_async(const float* src, int8_t* dst, size_t n, QuantizationParams params, cudaStream_t stream = 0);

            void dequantize_int8_to_f32_async(const int8_t* src, float* dst, size_t n, QuantizationParams params, cudaStream_t stream = 0);

            void quantize_f32_to_int8_with_calibration(const float* src, int8_t* dst, size_t n, float min_val, float max_val, cudaStream_t stream = 0);

            void build_histogram(const float* data, uint32_t* histogram, size_t n, float min_val, float max_val, int num_bins = 256, cudaStream_t stream = 0);

            void compute_minmax(const float* data, size_t n, float* min_val, float* max_val, cudaStream_t stream = 0);

        }  // namespace cuda
    }  // namespace quantize
}  // namespace nova

#endif  // NOVA_CUDA_QUANTIZE_INT8_KERNELS_HPP
