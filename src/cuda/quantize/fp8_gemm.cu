#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace nova {
namespace quantize {

namespace detail {

template<int BLOCK_SIZE>
__global__ void fp8_gemm_kernel(
    const FP8E4M3* __restrict__ a,
    const FP8E4M3* __restrict__ b,
    float* __restrict__ c,
    int m, int k, int n,
    float scale_a, float scale_b, float scale_out) {

    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    for (int kb = 0; kb < k; kb += BLOCK_SIZE) {
        __syncthreads();

        float a_val = 0.0f;
        float b_val = 0.0f;

        const int a_col = kb + threadIdx.x;
        const int b_row = kb + threadIdx.y;

        if (row < m && a_col < k) {
            a_val = static_cast<float>(a[row * k + a_col]) * scale_a;
        }

        if (b_row < k && col < n) {
            b_val = static_cast<float>(b[b_row * n + col]) * scale_b;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE && (kb + i) < k; ++i) {
            const int a_col_local = i + threadIdx.x;
            const int b_row_local = i + threadIdx.y;

            if (row < m && a_col_local < k && col < n && b_row_local < k) {
                float a_local = (threadIdx.x == a_col_local) ? a_val :
                    static_cast<float>(a[row * k + kb + a_col_local]) * scale_a;
                float b_local = (threadIdx.y == b_row_local) ? b_val :
                    static_cast<float>(b[(kb + b_row_local) * n + col]) * scale_b;
                sum += a_local * b_local;
            }
        }
    }

    if (row < m && col < n) {
        c[row * n + col] = sum * scale_out;
    }
}

template<int BLOCK_SIZE>
__global__ void fp8_gemm_naive_kernel(
    const FP8E4M3* __restrict__ a,
    const FP8E4M3* __restrict__ b,
    float* __restrict__ c,
    int m, int k, int n,
    float scale_a, float scale_b, float scale_out) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    for (int i = 0; i < k; ++i) {
        float a_val = static_cast<float>(a[row * k + i]) * scale_a;
        float b_val = static_cast<float>(b[i * n + col]) * scale_b;
        sum += a_val * b_val;
    }

    c[row * n + col] = sum * scale_out;
}

__global__ void fp8_gemm_row_kernel(
    const FP8E4M3* __restrict__ a,
    const FP8E4M3* __restrict__ b,
    float* __restrict__ c,
    int m, int k, int n,
    float scale_a, float scale_b, float scale_out) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    extern __shared__ float shmem[];

    for (int col_start = 0; col_start < n; col_start += blockDim.y) {
        float sum = 0.0f;

        for (int kb = 0; kb < k; kb += blockDim.y) {
            __syncthreads();

            int num_threads = min(static_cast<int>(blockDim.y), k - kb);
            int tid = threadIdx.y;

            if (tid < num_threads && col_start + tid < n) {
                shmem[tid] = static_cast<float>(b[(kb + tid) * n + col_start + tid]) * scale_b;
            }

            __syncthreads();

            for (int i = 0; i < num_threads && (kb + i) < k; ++i) {
                if (col_start + i < n) {
                    float a_val = static_cast<float>(a[row * k + kb + i]) * scale_a;
                    float b_val = shmem[i];
                    sum += a_val * b_val;
                }
            }
        }

        if (col_start + threadIdx.y < n) {
            atomicAdd(&c[row * n + col_start + threadIdx.y], sum * scale_out);
        }
    }
}

__global__ void fp8_gemm_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int m, int n,
    float scale_out,
    int num_partials) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int p = 0; p < num_partials; ++p) {
        sum += input[(p * m + row) * n + col];
    }

    output[row * n + col] = sum * scale_out;
}

} // namespace detail

size_t FP8GEMM::get_workspace_size(int m, int k, int n, const Config& config) {
    size_t a_size = static_cast<size_t>(m) * k * sizeof(FP8E4M3);
    size_t b_size = static_cast<size_t>(k) * n * sizeof(FP8E4M3);
    size_t c_size = static_cast<size_t>(m) * n * sizeof(float);

    size_t alignment = 256;
    size_t total = ((a_size + alignment - 1) / alignment) * alignment +
                   ((b_size + alignment - 1) / alignment) * alignment +
                   ((c_size + alignment - 1) / alignment) * alignment;
    return total;
}

void FP8GEMM::forward(
    const FP8E4M3* a, const FP8E4M3* b,
    float* output,
    int m, int k, int n,
    Config config,
    cudaStream_t stream) {

    constexpr int BLOCK_SIZE = 16;

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (m + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    detail::fp8_gemm_naive_kernel<BLOCK_SIZE><<<grid_dim, block_dim, 0, stream>>>(
        a, b, output, m, k, n,
        config.scale_a, config.scale_b, config.scale_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in FP8GEMM::forward: %s\n", cudaGetErrorString(err));
    }
}

void FP8GEMM::forward_async(
    const FP8E4M3* a, const FP8E4M3* b,
    float* output,
    int m, int k, int n,
    Config config,
    cudaStream_t stream) {

    forward(a, b, output, m, k, n, config, stream);
}

void FP8GEMM::backward(
    const FP8E4M3* grad_output,
    const FP8E4M3* a, const FP8E4M3* b,
    float* grad_a, float* grad_b,
    int m, int k, int n,
    Config config,
    cudaStream_t stream) {

    FP8GEMM::Config grad_config;
    grad_config.scale_a = config.scale_a;
    grad_config.scale_b = config.scale_out;

    FP8GEMM::forward(
        grad_output, b,
        grad_a,
        m, n, k,
        grad_config,
        stream
    );

    grad_config.scale_a = config.scale_out;
    grad_config.scale_b = config.scale_b;

    FP8GEMM::forward(
        grad_output, a,
        grad_b,
        n, m, k,
        grad_config,
        stream
    );
}

void FP8E5M2GEMM::forward(
    const FP8E5M2* a, const FP8E5M2* b,
    float* output,
    int m, int k, int n,
    Config config,
    cudaStream_t stream) {

    constexpr int BLOCK_SIZE = 16;

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (m + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    detail::fp8_gemm_naive_kernel<BLOCK_SIZE><<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const FP8E4M3*>(a),
        reinterpret_cast<const FP8E4M3*>(b),
        output, m, k, n,
        config.scale_a, config.scale_b, config.scale_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in FP8E5M2GEMM::forward: %s\n", cudaGetErrorString(err));
    }
}

} // namespace quantize
} // namespace nova
