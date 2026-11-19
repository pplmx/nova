/**
 * @file quantize_ops.hpp
 * @brief Core quantization operations
 * @defgroup quantize_ops Quantization Operations
 * @ingroup quantize
 *
 * Provides core quantization arithmetic operations including
 * quantized matrix multiplication.
 *
 * @see quantize_tensor.hpp For type definitions
 */

#ifndef NOVA_CUDA_QUANTIZE_OPS_HPP
#define NOVA_CUDA_QUANTIZE_OPS_HPP

#include <cuda/quantize/quantize_tensor.hpp>
#include <cuda/quantize/qat.hpp>
#include <vector>
#include <cmath>

namespace nova {
    namespace quantize {

        /**
         * @brief Supported numerical precisions
         * @enum Precision
         * @ingroup quantize_ops
         */
        // Precision lives in qat.hpp. Defining it again here made the same name mean two different
        // things in nova::quantize: three enumerators in this header, five in qat.hpp. Any
        // translation unit including both failed to compile ("multiple definition of 'enum class
        // nova::quantize::Precision'"), and a build that pulled them into separate translation units
        // would have been an ODR violation rather than a compile error. This header uses FP32, FP16
        // and INT8, all of which qat.hpp already provides.

        /**
         * @brief Quantized matrix multiplication
         * @class QuantizedMatmul
         * @ingroup quantize_ops
         */
        class QuantizedMatmul {
        public:
            /**
             * @brief Forward pass of quantized matmul
             * @param a Quantized input A
             * @param[out] c Output accumulator
             */
            static void forward(const QuantizedInt8& a, const QuantizedInt8& b, QuantizedInt8& output, int m, int k, int n);

            static std::vector<float> mixed_precision(const float* a, const int8_t* b, const float* scale_b, int m, int k, int n, Precision output_precision);
        };

        inline void QuantizedMatmul::forward(const QuantizedInt8& a, const QuantizedInt8& b, QuantizedInt8& output, int m, int k, int n) {
            const int8_t* a_data = a.data();
            const int8_t* b_data = b.data();
            std::vector<int8_t> output_data(m * n);

            float scale_a = a.metadata().scale;
            float scale_b = b.metadata().scale;
            float output_scale = scale_a * scale_b;

            // The accumulator holds the INTEGER dot product of the two int8 quantized
            // operands (|sum| can reach 127*127*k), not a real value:
            //
            //   real[i][j] = scale_a * scale_b * int_sum[i][j]
            //
            // Two bugs lived here. First, the raw integer was scaled by scale_a*scale_b
            // and stored directly as the int8 result, so the byte was meaningless.
            // Second, even after dequantizing correctly, reusing a's scale as the
            // output scale cannot work: k can push |real| far past 127 * scale_a, so
            // every entry saturated.
            //
            // An int8 result needs its own scale. Derive it from the largest magnitude
            // in the integer accumulator, which keeps the full int8 range in use and
            // makes the store lossless up to one quantization step.
            std::vector<int32_t> int_sums(static_cast<size_t>(m) * static_cast<size_t>(n), 0);
            int32_t max_abs_sum = 0;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    int32_t sum = 0;
                    for (int p = 0; p < k; ++p) {
                        sum += static_cast<int32_t>(a_data[i * k + p]) * static_cast<int32_t>(b_data[p * n + j]);
                    }
                    int_sums[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = sum;
                    max_abs_sum = std::max(max_abs_sum, std::abs(sum));
                }
            }

            // real_max = output_scale * max_abs_sum must fit in 127 * out_scale, so
            // out_scale = scale_a * scale_b * max_abs_sum / 127.
            const float result_scale = (max_abs_sum > 0) ? (output_scale * static_cast<float>(max_abs_sum) / 127.0f) : 1.0f;
            const float inverse_result_scale = 1.0f / result_scale;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    const float real = static_cast<float>(int_sums[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)]) * output_scale;
                    float value = real * inverse_result_scale;
                    value = std::max(-127.0f, std::min(127.0f, value));
                    output_data[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = static_cast<int8_t>(std::lround(value));
                }
            }

            QuantizationMetadata meta = a.metadata();
            meta.scale = result_scale;
            meta.zero_point = 0.0f;
            output = QuantizedInt8(std::move(output_data), meta, {m, n});
        }

        inline std::vector<float> QuantizedMatmul::mixed_precision(const float* a, const int8_t* b, const float* scale_b, int m, int k, int n, Precision output_precision) {
            std::vector<float> output(m * n, 0.0f);

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; ++p) {
                        float b_val = static_cast<float>(b[p * n + j]) * scale_b[p];
                        sum += a[i * k + p] * b_val;
                    }
                    output[i * n + j] = sum;
                }
            }

            return output;
        }

        inline void quantized_matmul(const QuantizedInt8& a, const QuantizedInt8& b, QuantizedInt8& output, int m, int k, int n) {
            QuantizedMatmul::forward(a, b, output, m, k, n);
        }

        inline std::vector<float> mixed_precision_matmul(const float* a, const int8_t* b, const float* scale_b, int m, int k, int n, Precision precision = Precision::FP32) {
            return QuantizedMatmul::mixed_precision(a, b, scale_b, m, k, n, precision);
        }

    }  // namespace quantize
}  // namespace nova

#endif  // NOVA_CUDA_QUANTIZE_OPS_HPP
