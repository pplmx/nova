/**
 * @file fp8_gemm.hpp
 * @brief FP8 matrix multiplication
 * @defgroup fp8_gemm FP8 GEMM
 * @ingroup quantize
 *
 * Provides FP8 general matrix multiplication optimized for transformer workloads.
 * Uses NVIDIA FP8 Tensor Cores for high throughput.
 *
 * @note Requires H100 or newer GPU
 * @see fp8_types.hpp For FP8 type definitions
 */

#ifndef NOVA_CUDA_QUANTIZE_FP8_GEMM_HPP
#define NOVA_CUDA_QUANTIZE_FP8_GEMM_HPP

#include <cuda/quantize/fp8_types.hpp>
#include <cuda_runtime.h>

namespace nova {
    namespace quantize {

        /**
         * @brief FP8 matrix multiplication interface
         * @class FP8GEMM
         * @ingroup fp8_gemm
         */
        class FP8GEMM {
        public:
            /**
             * @brief Configuration for FP8 GEMM
             * @struct Config
             */
            struct Config {
                /** @brief Scale for input A */
                float scale_a;
                /** @brief Scale for input B */
                float scale_b;
                /** @brief Scale for output */
                float scale_out;
                bool use_tensor_core;
                int num_streams;

                Config()
                    : scale_a(1.0f),
                      scale_b(1.0f),
                      scale_out(1.0f),
                      use_tensor_core(false),
                      num_streams(1) {}

                Config(float sa, float sb, float so, bool utc = false, int ns = 1)
                    : scale_a(sa),
                      scale_b(sb),
                      scale_out(so),
                      use_tensor_core(utc),
                      num_streams(ns) {}
            };

            static void forward(const FP8E4M3* a, const FP8E4M3* b, float* output, int m, int k, int n, Config config = Config {}, cudaStream_t stream = 0);

            static void forward_async(const FP8E4M3* a, const FP8E4M3* b, float* output, int m, int k, int n, Config config = Config {}, cudaStream_t stream = 0);

            static void backward(const FP8E4M3* grad_output,
                                 const FP8E4M3* a,
                                 const FP8E4M3* b,
                                 float* grad_a,
                                 float* grad_b,
                                 int m,
                                 int k,
                                 int n,
                                 Config config = Config {},
                                 cudaStream_t stream = 0);

            static size_t get_workspace_size(int m, int k, int n, const Config& config);
        };

        class FP8E5M2GEMM {
        public:
            struct Config {
                float scale_a;
                float scale_b;
                float scale_out;
                int num_streams;

                Config()
                    : scale_a(1.0f),
                      scale_b(1.0f),
                      scale_out(1.0f),
                      num_streams(1) {}

                Config(float sa, float sb, float so, int ns = 1)
                    : scale_a(sa),
                      scale_b(sb),
                      scale_out(so),
                      num_streams(ns) {}
            };

            static void forward(const FP8E5M2* a, const FP8E5M2* b, float* output, int m, int k, int n, Config config = Config {}, cudaStream_t stream = 0);
        };

    }  // namespace quantize
}  // namespace nova

#endif  // NOVA_CUDA_QUANTIZE_FP8_GEMM_HPP
