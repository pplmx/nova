#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda_runtime.h>
#include "cuda/device/error.h"
#include <cstdio>
#include <stdexcept>
#include <string>
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
            /// A_TRANSPOSED selects a[i * m + row] instead of a[row * k + i], so that the same kernel
            /// can compute both operands of a GEMM backward pass: grad_b needs A^T * grad_output,
            /// which is not expressible as a plain forward() call.
            template <int BLOCK_SIZE, typename T, bool A_TRANSPOSED = false, bool B_TRANSPOSED = false>
            __global__ void fp8_gemm_naive_kernel(
                const T* __restrict__ a, const T* __restrict__ b, float* __restrict__ c, int m, int k, int n, float scale_a, float scale_b, float scale_out) {
                const int row = blockIdx.y * blockDim.y + threadIdx.y;
                const int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= m || col >= n) {
                    return;
                }

                // Address arithmetic is done in size_t, not int. Every index below is a product of
                // two dimensions, and with m * k (or k * n, or m * n) past INT_MAX those products
                // wrapped inside a signed int and the kernel read or wrote outside its own buffers.
                // Measured with all three dimensions legal and every buffer allocated by the
                // caller at its full 4 GiB: m = 2, k = INT_MAX, n = 2 -- `row * k + i` is
                // 1 * 2147483647 + i for row 1, so it goes negative for every i >= 1, and the
                // launch died with "an illegal memory access was encountered". The row/col
                // comparisons above stay int because they compare against m and n, which are the
                // tile bounds and fit.
                const size_t row_offset = static_cast<size_t>(row);
                const size_t col_offset = static_cast<size_t>(col);
                const size_t k_size = static_cast<size_t>(k);
                const size_t m_size = static_cast<size_t>(m);
                const size_t n_size = static_cast<size_t>(n);

                float sum = 0.0f;

                for (int i = 0; i < k; ++i) {
                    const size_t i_size = static_cast<size_t>(i);
                    float a_val = static_cast<float>(A_TRANSPOSED ? a[i_size * m_size + row_offset] : a[row_offset * k_size + i_size]) * scale_a;
                    // B_TRANSPOSED treats b as an (n x k) row-major matrix and reads column i of it.
                    float b_val = static_cast<float>(B_TRANSPOSED ? b[col_offset * k_size + i_size] : b[i_size * n_size + col_offset]) * scale_b;
                    sum += a_val * b_val;
                }

                c[row_offset * n_size + col_offset] = sum * scale_out;
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

        namespace {

            /// Both launch dimensions come from m and n, so a non-positive one asks for a zero-sized
            /// grid, which CUDA rejects with "invalid configuration argument" at the launch line.
            /// forward() reported that with a fprintf and returned normally, so the caller saw no
            /// error; a negative dimension reaches dim3's unsigned constructor instead.
            void require_positive_dims(int m, int k, int n, const char* fn) {
                if (m <= 0 || k <= 0 || n <= 0) {
                    throw ::std::invalid_argument(::std::string(fn) + ": m, k and n must be positive, got m=" + ::std::to_string(m) + " k=" + ::std::to_string(k)
                                                  + " n=" + ::std::to_string(n));
                }
            }

        }  // namespace

        void FP8GEMM::forward(const FP8E4M3* a, const FP8E4M3* b, float* output, int m, int k, int n, Config config, cudaStream_t stream) {
            require_positive_dims(m, k, n, "FP8GEMM::forward");

            constexpr int BLOCK_SIZE = 16;

            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

            detail::fp8_gemm_naive_kernel<BLOCK_SIZE, FP8E4M3><<<grid_dim, block_dim, 0, stream>>>(a, b, output, m, k, n, config.scale_a, config.scale_b, config.scale_out);

            CUDA_CHECK(cudaGetLastError());
        }

        void FP8GEMM::forward_async(const FP8E4M3* a, const FP8E4M3* b, float* output, int m, int k, int n, Config config, cudaStream_t stream) {
            forward(a, b, output, m, k, n, config, stream);
        }

        void FP8GEMM::backward(
            const FP8E4M3* grad_output, const FP8E4M3* a, const FP8E4M3* b, float* grad_a, float* grad_b, int m, int k, int n, Config config, cudaStream_t stream) {
            require_positive_dims(m, k, n, "FP8GEMM::backward");
            // For C = A(m x k) * B(k x n) and upstream gradient G(m x n):
            //   grad_a = G * B^T   -> (m x k)
            //   grad_b = A^T * G   -> (k x n)
            //
            // grad_a is a plain product of G with B^T, which is what the old code computed.
            // grad_b is NOT: it needs A transposed on the left, and the previous implementation
            // passed (grad_output, a, grad_b, n, m, k), which computes G^T * A rather than A^T * G.
            // Measured with A = B = [[1,2],[3,4]] and G = ones: grad_b came back [[4,6],[4,6]]
            // where [[4,4],[6,6]] is correct -- transposed. There was no test for backward().
            //
            // scale_out also went unset in both calls, leaving the default 1.0 and dropping the
            // caller's output scaling.
            // grad_a = G(m x n) * B^T(n x k). The kernel computes a(row x p) * b(p x col) with b
            // indexed row-major as b[p * n + col]; feeding it B^T means reading b as
            // host_b[col * k + p], i.e. b_transposed with its own leading dimension. Passing b
            // unmodified computes G * B, not G * B^T -- which is what the original code did, and
            // it is only indistinguishable from correct when B happens to be symmetric.
            FP8GEMM::Config grad_a_config;
            grad_a_config.scale_a = config.scale_out;
            grad_a_config.scale_b = config.scale_b;
            grad_a_config.scale_out = 1.0f;
            detail::fp8_gemm_naive_kernel<16, FP8E4M3, false, true><<<dim3((k + 15) / 16, (m + 15) / 16), dim3(16, 16), 0, stream>>>(
                grad_output, b, grad_a, m, n, k, grad_a_config.scale_a, grad_a_config.scale_b, grad_a_config.scale_out);

            FP8GEMM::Config grad_b_config;
            grad_b_config.scale_a = config.scale_b;
            grad_b_config.scale_b = config.scale_out;
            grad_b_config.scale_out = 1.0f;
            // grad_b = A^T * G, with A read transposed.
            detail::fp8_gemm_naive_kernel<16, FP8E4M3, true><<<dim3((n + 15) / 16, (k + 15) / 16), dim3(16, 16), 0, stream>>>(
                a, grad_output, grad_b, k, m, n, grad_b_config.scale_a, grad_b_config.scale_b, grad_b_config.scale_out);
        }

        void FP8E5M2GEMM::forward(const FP8E5M2* a, const FP8E5M2* b, float* output, int m, int k, int n, Config config, cudaStream_t stream) {
            // Same geometry as FP8GEMM::forward above, and it was the entry left out when that one
            // gained the check: a non-positive m or n asks for a zero-sized grid, and the error was
            // reported only with a fprintf.
            require_positive_dims(m, k, n, "FP8E5M2GEMM::forward");

            constexpr int BLOCK_SIZE = 16;

            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Instantiate with the caller's own format. Reinterpreting E5M2 as E4M3, as
            // this did, decodes every value through the wrong exponent/mantissa split: a
            // 1x1 product of 2 and 3 returned 5.0 instead of 6.0 because 3.0's E5M2 bit
            // pattern is not 3.0 in E4M3.
            detail::fp8_gemm_naive_kernel<BLOCK_SIZE, FP8E5M2><<<grid_dim, block_dim, 0, stream>>>(a, b, output, m, k, n, config.scale_a, config.scale_b, config.scale_out);

            CUDA_CHECK(cudaGetLastError());
        }

    }  // namespace quantize
}  // namespace nova
