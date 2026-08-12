#pragma once

/**
 * @file tensor_parallel_matmul.h
 * @brief Tensor parallelism for matrix multiplication
 *
 * Implements column-parallel and row-parallel matmul strategies
 * where weight matrices are partitioned across GPUs.
 *
 * Column-Parallel: Split weights along output dimension
 * Row-Parallel: Split weights along input dimension
 */

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/neural/matmul.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::neural {

/**
 * @enum TensorParallelStrategy
 * @brief Strategy for tensor-parallel matrix multiplication
 */
enum class TensorParallelStrategy {
    /** Split weight matrix along output dimension (n)
     *  Each GPU computes A @ B_part, then AllReduce
     *  Input A is replicated, output C is identical on all GPUs */
    ColumnParallel,

    /** Split weight matrix along input dimension (k)
     *  Each GPU computes A_part @ B, no communication needed
     *  Input A must be partitioned, output C is partitioned */
    RowParallel
};

/**
 * @class TensorParallelMatmul
 * @brief Tensor-parallel matrix multiplication
 *
 * Performs matrix multiplication with weight matrix partitioned
 * across multiple GPUs for memory efficiency and parallelism.
 *
 * @example
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 *
 * TensorParallelMatmul tpm(ctx, TensorParallelStrategy::ColumnParallel);
 * tpm.matmul(A, B, C, m, n, k);
 * @endcode
 */
class TensorParallelMatmul {
public:
    /**
     * @brief Construct tensor-parallel matmul
     * @param ctx NCCL context for collective operations
     * @param strategy Column-parallel or row-parallel
     */
    TensorParallelMatmul(
        ::cuda::nccl::NcclContext& ctx,
        TensorParallelStrategy strategy);

    ~TensorParallelMatmul();

    // Non-copyable, non-movable: the NcclContext reference member and the
    // non-movable NcclAllReduce (deleted base copy-ctor) forbid moves. Objects
    // are constructed in place / under unique_ptr, so neither is ever needed.
    TensorParallelMatmul(const TensorParallelMatmul&) = delete;
    TensorParallelMatmul& operator=(const TensorParallelMatmul&) = delete;
    TensorParallelMatmul(TensorParallelMatmul&&) = delete;
    TensorParallelMatmul& operator=(TensorParallelMatmul&&) = delete;

    /**
     * @brief Tensor-parallel matmul (synchronous)
     * @param A Input matrix [m x k]
     * @param B Weight matrix [k x n] (partitioned)
     * @param C Output matrix [m x n] (identical after AllReduce)
     * @param m Rows in A and C
     * @param n Columns in B and C
     * @param k Inner dimension
     */
    void matmul(
        const float* A,
        const float* B,
        float* C,
        int m,
        int n,
        int k);

    /**
     * @brief Tensor-parallel matmul (asynchronous)
     * @param A Input matrix
     * @param B Weight matrix
     * @param C Output matrix
     * @param m Rows
     * @param n Columns
     * @param k Inner dimension
     * @param stream CUDA stream
     */
    void matmul_async(
        const float* A,
        const float* B,
        float* C,
        int m,
        int n,
        int k,
        cudaStream_t stream);

    /**
     * @brief Get current TP degree
     * @return Number of GPUs participating
     */
    [[nodiscard]] int tp_degree() const;

    /**
     * @brief Get strategy
     * @return Column or row parallel
     */
    [[nodiscard]] TensorParallelStrategy strategy() const;

    /**
     * @brief Calculate required buffer size for tensor-parallel operation
     * @param m Rows
     * @param n Columns
     * @param k Inner dimension
     * @param tp_degree Number of GPUs
     * @return Required bytes
     */
    static constexpr size_t required_buffer_size(
        int m, int n, int k, int tp_degree) {
        if (tp_degree <= 1) {
            return 0;
        }
        return (static_cast<size_t>(m) * static_cast<size_t>(n) *
                sizeof(float) * 2) / tp_degree;
    }

private:
    // Lazily-created per-instance cuBLAS handle bound to the op's stream. A
    // shared handle (get_cublas_handle) is not thread-safe and cannot be stream
    // bound per rank; distinct instances (one per rank in the thread-per-rank
    // harness) each own one, so the block GEMM orders correctly with the NCCL
    // collective on the caller's stream.
    cublasHandle_t handle_ = nullptr;

    cublasHandle_t ensure_handle();

    ::cuda::nccl::NcclContext& ctx_;
    ::cuda::nccl::NcclAllReduce reducer_;
    TensorParallelStrategy strategy_;
};

/**
 * @brief Estimate recommended TP degree based on matrix dimensions
 * @param m Rows in A
 * @param n Columns in B
 * @param k Inner dimension
 * @return Recommended TP degree (1-8)
 */
[[nodiscard]]
int recommend_tp_degree(int m, int n, int k);

}  // namespace cuda::neural
