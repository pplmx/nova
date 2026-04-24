#pragma once

/**
 * @file matmul.h
 * @brief Multi-GPU distributed matrix multiply
 *
 * Row-wise split multi-GPU matmul using DistributedAllGather for output
 * aggregation. Each GPU computes its partition of output rows, then all-gather
 * reconstructs the full result on each device.
 *
 * @example
 * @code
 * // Single-GPU fallback (automatic when device_count <= 1)
 * DistributedMatmul::matmul(A, B, C, m, n, k);
 *
 * // Multi-GPU: requires all GPUs calling simultaneously
 * // Each GPU computes rows [rank * m/n, (rank+1) * m/n) of C
 * // After all-gather, all GPUs have the full result in C
 * @endcode
 */

#include "common.h"

#include <cuda_runtime.h>

namespace cuda::distributed {

/**
 * @enum ParallelismStrategy
 * @brief Strategy for distributing computation across GPUs
 */
enum class ParallelismStrategy {
    DataParallel,     // Row-partition input A (this phase)
    TensorParallel,   // Column-partition weights (v1.2)
    PipelineParallel  // Layer pipeline (v1.2)
};

/**
 * @struct DistributedMatmulOptions
 * @brief Options for distributed matmul operations
 */
struct DistributedMatmulOptions {
    /** Parallelism strategy to use */
    ParallelismStrategy strategy = ParallelismStrategy::DataParallel;

    /** Scalar multiplier for the product (A @ B) */
    float alpha = 1.0f;

    /** Scalar multiplier for accumulated C */
    float beta = 0.0f;

    /** Transpose A before multiplication */
    bool trans_a = false;

    /** Transpose B before multiplication */
    bool trans_b = false;
};

/**
 * @class DistributedMatmul
 * @brief Multi-GPU matrix multiply with row-wise data parallel distribution
 *
 * Implements row-wise split multi-GPU matrix multiply:
 * - Each GPU computes a contiguous block of output rows
 * - Weight matrix B is replicated on all GPUs
 * - Uses DistributedAllGather to reconstruct full output on each GPU
 *
 * @note Single-GPU fallback: Delegates to cuda::neural::matmul
 *       No multi-GPU primitives are called when device_count <= 1
 *
 * @example
 * @code
 * DistributedMatmulOptions opts;
 * opts.alpha = 1.0f;
 * opts.beta = 0.0f;
 *
 * DistributedMatmul::matmul(A, B, C, m, n, k, opts);
 * @endcode
 */
class DistributedMatmul {
public:
    /**
     * @brief Distributed matrix multiply (synchronous)
     *
     * Distributes computation across all available GPUs.
     * On single-GPU systems, delegates to cuda::neural::matmul.
     *
     * @param A Input matrix [m x k] (row-major)
     * @param B Weight matrix [k x n] (row-major)
     * @param C Output matrix [m x n] (row-major)
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param options Operation options (default: all defaults)
     */
    static void matmul(
        const float* A,
        const float* B,
        float* C,
        int m,
        int n,
        int k,
        DistributedMatmulOptions options = {}
    );

    /**
     * @brief Distributed matrix multiply (asynchronous)
     *
     * Non-blocking version with explicit stream. Useful for
     * overlapping with other GPU operations.
     *
     * @param A Input matrix
     * @param B Weight matrix
     * @param C Output matrix
     * @param m Rows in A and C
     * @param n Columns in B and C
     * @param k Inner dimension
     * @param stream CUDA stream for async operations
     * @param options Operation options
     */
    static void matmul_async(
        const float* A,
        const float* B,
        float* C,
        int m,
        int n,
        int k,
        cudaStream_t stream,
        DistributedMatmulOptions options = {}
    );

    /**
     * @brief Single-GPU matmul (exposed for testing)
     *
     * Delegates to cuda::neural::matmul. Used for single-GPU fallback
     * and as reference implementation for correctness testing.
     *
     * @param A Input matrix
     * @param B Weight matrix
     * @param C Output matrix
     * @param m Rows in A and C
     * @param n Columns in B and C
     * @param k Inner dimension
     * @param options Operation options
     */
    static void matmul_single_gpu(
        const float* A,
        const float* B,
        float* C,
        int m,
        int n,
        int k,
        DistributedMatmulOptions options = {}
    );

    /**
     * @brief Check if multi-GPU execution is needed
     * @return true if device_count > 1, false otherwise
     */
    static bool needs_multi_gpu();
};

}  // namespace cuda::distributed
