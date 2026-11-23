#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace nova {
    namespace quantize {

        namespace detail {
            // fp8_gemm_kernel (a shared-memory tiled variant) used to live here. It was never
            // launched, and it could not be: it calls __syncthreads() inside its k-loop, after
            // `if (row >= m || col >= n) return;` -- threads of a block that exit early, while
            // the rest wait at the barrier, is a deadlock/UB. Wiring it up would have hung.
            // Removed rather than repaired: there is no caller, and the correct form (no early
            // return before a barrier) should be introduced together with a test.

            /// Element-wise multiply-accumulate over a k-length dot product. Templated on the
            /// FP8 format so each encoding is decoded by its own conversion operator: an E5M2
            /// value reinterpreted as E4M3 decodes to a different number (0.5 is 0x38 in E5M2
            /// and 0x30 in E4M3), which silently produced wrong results.
            template <int BLOCK_SIZE, typename T>
            __global__ void fp8_gemm_naive_kernel(
                const T* __restrict__ a, const T* __restrict__ b, float* __restrict__ c, int m, int k, int n, float scale_a, float scale_b, float scale_out) {
                const int row = blockIdx.y * blockDim.y + threadIdx.y;
                const int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= m || col >= n) {
                    return;
                }

                float sum = 0.0f;

                for (int i = 0; i < k; ++i) {
                    float a_val = static_cast<float>(a[row * k + i]) * scale_a;
                    float b_val = static_cast<float>(b[i * n + col]) * scale_b;
                    sum += a_val * b_val;
                }

                c[row * n + col] = sum * scale_out;
            }

            // fp8_gemm_row_kernel and fp8_gemm_reduce_kernel used to live here, a two-stage
            // split-K GEMM that nothing called. The row kernel has the same barrier defect as the
            // tiled kernel above (__syncthreads() after `if (row >= m) return;`), so it would
            // have hung for any m not a multiple of the block height.

        }  // namespace detail

        size_t FP8GEMM::get_workspace_size(int m, int k, int n, const Config& config) {
            size_t a_size = static_cast<size_t>(m) * k * sizeof(FP8E4M3);
            size_t b_size = static_cast<size_t>(k) * n * sizeof(FP8E4M3);
            size_t c_size = static_cast<size_t>(m) * n * sizeof(float);

            size_t alignment = 256;
            size_t total =
                ((a_size + alignment - 1) / alignment) * alignment + ((b_size + alignment - 1) / alignment) * alignment + ((c_size + alignment - 1) / alignment) * alignment;
            return total;
        }

        void FP8GEMM::forward(const FP8E4M3* a, const FP8E4M3* b, float* output, int m, int k, int n, Config config, cudaStream_t stream) {
            constexpr int BLOCK_SIZE = 16;

            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

            detail::fp8_gemm_naive_kernel<BLOCK_SIZE, FP8E4M3><<<grid_dim, block_dim, 0, stream>>>(a, b, output, m, k, n, config.scale_a, config.scale_b, config.scale_out);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error in FP8GEMM::forward: %s\n", cudaGetErrorString(err));
            }
        }

        void FP8GEMM::forward_async(const FP8E4M3* a, const FP8E4M3* b, float* output, int m, int k, int n, Config config, cudaStream_t stream) {
            forward(a, b, output, m, k, n, config, stream);
        }

        void FP8GEMM::backward(
            const FP8E4M3* grad_output, const FP8E4M3* a, const FP8E4M3* b, float* grad_a, float* grad_b, int m, int k, int n, Config config, cudaStream_t stream) {
            FP8GEMM::Config grad_config;
            grad_config.scale_a = config.scale_a;
            grad_config.scale_b = config.scale_out;

            FP8GEMM::forward(grad_output, b, grad_a, m, n, k, grad_config, stream);

            grad_config.scale_a = config.scale_out;
            grad_config.scale_b = config.scale_b;

            FP8GEMM::forward(grad_output, a, grad_b, n, m, k, grad_config, stream);
        }

        void FP8E5M2GEMM::forward(const FP8E5M2* a, const FP8E5M2* b, float* output, int m, int k, int n, Config config, cudaStream_t stream) {
            constexpr int BLOCK_SIZE = 16;

            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Instantiate with the caller's own format. Reinterpreting E5M2 as E4M3, as
            // this did, decodes every value through the wrong exponent/mantissa split: a
            // 1x1 product of 2 and 3 returned 5.0 instead of 6.0 because 3.0's E5M2 bit
            // pattern is not 3.0 in E4M3.
            detail::fp8_gemm_naive_kernel<BLOCK_SIZE, FP8E5M2><<<grid_dim, block_dim, 0, stream>>>(a, b, output, m, k, n, config.scale_a, config.scale_b, config.scale_out);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error in FP8E5M2GEMM::forward: %s\n", cudaGetErrorString(err));
            }
        }

    }  // namespace quantize
}  // namespace nova
