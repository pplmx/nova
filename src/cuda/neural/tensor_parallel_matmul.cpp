/**
 * @file tensor_parallel_matmul.cpp
 * @brief Tensor parallelism implementation
 */

#include "cuda/neural/tensor_parallel_matmul.h"
#include "cuda/neural/matmul.h"
#include "cuda/device/error.h"

#include <algorithm>
#include <stdexcept>

namespace cuda::neural {

TensorParallelMatmul::TensorParallelMatmul(
    ::cuda::nccl::NcclContext& ctx,
    TensorParallelStrategy strategy)
    : ctx_(ctx),
      reducer_(ctx),
      strategy_(strategy) {}

TensorParallelMatmul::~TensorParallelMatmul() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

cublasHandle_t TensorParallelMatmul::ensure_handle() {
    if (!handle_) {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    return handle_;
}

namespace {

[[noreturn]] void throw_shape_error(const char* what) {
    throw std::invalid_argument(what);
}

// Row-major m x k A and k x n B (same convention as cuda::neural::matmul /
// matmul.cu's matmul_kernel: C[row][col] = sum_i A[row][k+i] * B[i][n+col]).
// cuBLAS is column-major, so the established matmul.cu trick computes C^T =
// B^T A^T by swapping m/n and passing the row-major matrices with their full
// leading dimensions.
void column_block_sgemm(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    int rank,
    int local_n,
    cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Narrow the output to the rank's column block [rank*local_n, +local_n)
    // and stride through the full leading dimensions (ldb = ldc = n) so the
    // block lands in the strided, zero-padded full-width output.
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        local_n,
        m,
        k,
        &alpha,
        B + static_cast<size_t>(rank) * local_n,
        n,
        A,
        k,
        &beta,
        C + static_cast<size_t>(rank) * local_n,
        n));
}

}  // namespace

void TensorParallelMatmul::matmul(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k) {

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    // RAII stream guard: matmul_async can throw on shape-validation errors
    // (e.g. a non-divisible n/m), and the stream must not leak then.
    struct StreamGuard {
        cudaStream_t s;
        ~StreamGuard() {
            if (s != nullptr) {
                cudaStreamDestroy(s);
            }
        }
    } guard{stream};

    matmul_async(A, B, C, m, n, k, stream);

    // Blocking entry point; device-wide sync covers both the op stream and any
    // default-stream fallback work.
    CUDA_CHECK(cudaDeviceSynchronize());
}

void TensorParallelMatmul::matmul_async(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    cudaStream_t stream) {

    if (m <= 0 || n <= 0 || k <= 0) {
        throw_shape_error("tensor-parallel matmul requires positive m, n, k");
    }

    int tp_degree = ctx_.device_count();
    if (tp_degree <= 1) {
        MatmulOptions opts;
        opts.handle = get_cublas_handle();
        cuda::neural::matmul(A, B, C, m, n, k, opts);
        return;
    }

    // Rank is the thread's currently-active device resolved through the NCCL
    // group (thread-per-rank harness). rank_of_device() validates membership
    // and throws NcclException when the active device is not in the group —
    // unlike a raw device index, it stays correct for non-default NCCL groups
    // (e.g. {0,2,5} or reordered device ids) and fails fast on OOB devices
    // instead of reading/writing the wrong B/C block.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const int rank = ctx_.rank_of_device(device);

    cublasHandle_t handle = ensure_handle();

    if (strategy_ == TensorParallelStrategy::ColumnParallel) {
        // Split B along n: each rank owns columns [rank*local_n, +local_n).
        if (n % tp_degree != 0) {
            throw_shape_error(
                "column-parallel tensor-parallel matmul requires n divisible "
                "by the tp degree");
        }
        // The full result is the block-wise AllReduce of every rank's padded
        // column block; without a live NCCL context no rank can reconstruct it.
        // Fail fast rather than silently returning a partial output.
        if (!ctx_.has_nccl()) {
            throw_shape_error(
                "column-parallel tensor-parallel matmul on multi-GPU requires "
                "an initialized NCCL context");
        }
        const int local_n = n / tp_degree;

        // Zero the full-width output so the block-wise AllReduce sums each
        // rank's padded column block into the complete result. All of this is
        // ordered on the caller's stream (memset -> block gemm -> AllReduce).
        CUDA_CHECK(cudaMemsetAsync(
            C, 0, static_cast<size_t>(m) * n * sizeof(float), stream));

        column_block_sgemm(handle, A, B, C, m, n, k, rank, local_n, stream);

        if (ctx_.has_nccl()) {
            // Propagate collective failures instead of silently returning
            // success with garbage output (a dead/aborted comm, timeout, or
            // async NCCL error otherwise folds into an unchecked NcclResult).
            ::cuda::nccl::NcclResult result = reducer_.all_reduce_async(
                C, C, static_cast<size_t>(m) * n,
                ncclFloat32, ncclSum, stream);
            if (!result.ok()) {
                throw ::cuda::nccl::NcclException(
                    result.error_message.c_str(), result.code,
                    "TensorParallelMatmul::matmul_async column AllReduce",
                    __FILE__, __LINE__);
            }
        }
    } else {
        // Split A along m: each rank owns rows [rank*local_m, +local_m); B is
        // replicated. Output C is partitioned (rank's contiguous row block).
        if (m % tp_degree != 0) {
            throw_shape_error(
                "row-parallel tensor-parallel matmul requires m divisible by "
                "the tp degree");
        }
        const int local_m = m / tp_degree;

        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            local_m,
            k,
            &alpha,
            B,
            n,
            A + static_cast<size_t>(rank) * local_m * k,
            k,
            &beta,
            C,
            n));
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
