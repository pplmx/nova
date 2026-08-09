/**
 * @file tensor_parallel_matmul.cpp
 * @brief Tensor parallelism implementation
 */

#include "cuda/neural/tensor_parallel_matmul.h"
#include "cuda/neural/matmul.h"
#include "cuda/device/error.h"

#include <algorithm>

namespace cuda::neural {

TensorParallelMatmul::TensorParallelMatmul(
    ::cuda::nccl::NcclContext& ctx,
    TensorParallelStrategy strategy)
    : ctx_(ctx),
      reducer_(ctx),
      strategy_(strategy) {}

void TensorParallelMatmul::matmul(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k) {

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    matmul_async(A, B, C, m, n, k, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void TensorParallelMatmul::matmul_async(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    cudaStream_t stream) {

    int tp_degree = ctx_.device_count();
    if (tp_degree <= 1) {
        MatmulOptions opts;
        opts.handle = get_cublas_handle();
        cuda::neural::matmul(A, B, C, m, n, k, opts);
        return;
    }

    if (strategy_ == TensorParallelStrategy::ColumnParallel) {
        int local_n = n / tp_degree;
        int offset = local_n * k;

        const float* B_part = B + offset;
        float* C_part = C;

        MatmulOptions opts;
        opts.handle = get_cublas_handle();

        cuda::neural::matmul(A, B_part, C_part, m, local_n, k, opts);

        if (ctx_.has_nccl()) {
            reducer_.all_reduce_async(
                C_part, C_part, m * local_n,
                ncclFloat32, ncclSum, stream);
        }

        float* recv_buffer = C + m * local_n;
        for (int rank = 1; rank < tp_degree; ++rank) {
            int part_offset = rank * local_n * k;
            const float* part_B = B + part_offset;
            float* part_C = C + rank * m * local_n;

            cuda::neural::matmul(A, part_B, part_C, m, local_n, k, opts);

            if (ctx_.has_nccl()) {
                reducer_.all_reduce_async(
                    part_C, part_C, m * local_n,
                    ncclFloat32, ncclSum, stream);
            }
        }
    } else {
        MatmulOptions opts;
        opts.handle = get_cublas_handle();

        int local_m = m / tp_degree;
        int offset = local_m * k;

        const float* A_part = A + offset;
        float* C_part = C + offset;

        cuda::neural::matmul(A_part, B, C_part, local_m, n, k, opts);
    }
}

int TensorParallelMatmul::tp_degree() const {
    return std::max(1, ctx_.device_count());
}

TensorParallelStrategy TensorParallelMatmul::strategy() const {
    return strategy_;
}

int recommend_tp_degree(int m, int n, int k) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    constexpr size_t MIN_MEMORY_PER_GPU = 256 * 1024 * 1024;  // 256 MB minimum
    constexpr size_t BYTES_PER_ELEMENT = sizeof(float);

    size_t weight_size = static_cast<size_t>(n) * static_cast<size_t>(k) * BYTES_PER_ELEMENT;
    size_t activation_size = static_cast<size_t>(m) * static_cast<size_t>(k) * BYTES_PER_ELEMENT;

    for (int tp = device_count; tp >= 1; --tp) {
        size_t weight_per_gpu = weight_size / tp;
        size_t activation_per_gpu = activation_size;

        if (weight_per_gpu + activation_per_gpu < MIN_MEMORY_PER_GPU * 10) {
            return tp;
        }
    }

    return 1;
}

}  // namespace cuda::neural
