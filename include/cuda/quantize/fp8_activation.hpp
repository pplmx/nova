/**
 * @file fp8_activation.hpp
 * @brief FP8 activation functions
 * @defgroup fp8_activation FP8 Activations
 * @ingroup quantize
 *
 * Provides FP8-compatible activation functions for quantized neural networks.
 *
 * @see fp8_types.hpp For FP8 type definitions
 */

#ifndef NOVA_CUDA_QUANTIZE_FP8_ACTIVATION_HPP
#define NOVA_CUDA_QUANTIZE_FP8_ACTIVATION_HPP

#include <cuda/quantize/fp8_types.hpp>
#include <cuda_runtime.h>
#include <cmath>

namespace nova {
    namespace quantize {
        namespace cuda {

            namespace activation {

                /** @brief Sigmoid activation */
                __device__ __forceinline__ float sigmoidf(float x) {
                    return 1.0f / (1.0f + expf(-x));
                }

                __device__ __forceinline__ float geluf(float x) {
                    constexpr float sqrt_2_inv = 0.7071067811865476f;
                    constexpr float half = 0.5f;
                    float cdf = half * (1.0f + erff(x * sqrt_2_inv));
                    return x * cdf;
                }

                template <typename T>
                __device__ __forceinline__ T relu_fp8(T x) {
                    if (static_cast<float>(x) < 0.0f) {
                        return T(0.0f);
                    }
                    return x;
                }

                template <typename T>
                __device__ __forceinline__ T gelu_fp8(T x) {
                    float f = static_cast<float>(x);
                    float result = geluf(f);
                    return T(result);
                }

                template <typename T>
                __device__ __forceinline__ T sigmoid_fp8(T x) {
                    float f = static_cast<float>(x);
                    float result = sigmoidf(f);
                    return T(result);
                }

                template <typename T>
                __device__ __forceinline__ T leaky_relu_fp8(T x, float alpha = 0.01f) {
                    float f = static_cast<float>(x);
                    float result = f >= 0.0f ? f : alpha * f;
                    return T(result);
                }

            }  // namespace activation

            // The kernels and the host-callable entry points live in fp8_activation.cu.
            // They used to be defined here inside `#ifdef __CUDACC__`, which meant a plain
            // .cpp translation unit compiled the functions with an EMPTY body: calling
            // cuda::relu() from host code linked successfully and returned without touching
            // the output buffer. Only a .cu file ever saw the real implementation.

            template <typename T>
            void relu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

            template <typename T>
            void gelu(const T* input, T* output, size_t n, cudaStream_t stream = 0);

            template <typename T>
            void sigmoid(const T* input, T* output, size_t n, cudaStream_t stream = 0);

            template <typename T>
            void leaky_relu(const T* input, T* output, size_t n, float alpha = 0.01f, cudaStream_t stream = 0);

            // Scalar helpers are genuinely device-shared, so they stay inline here.
            template <typename T>
            __device__ __forceinline__ T relu_fp8_scalar(T x) {
                return activation::relu_fp8(x);
            }

            template <typename T>
            __device__ __forceinline__ T gelu_fp8_scalar(T x) {
                return activation::gelu_fp8(x);
            }

        }  // namespace cuda
    }  // namespace quantize
}  // namespace nova

#endif  // NOVA_CUDA_QUANTIZE_FP8_ACTIVATION_HPP
