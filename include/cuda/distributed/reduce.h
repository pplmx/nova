#pragma once

/**
 * @file reduce.h
 * @brief Multi-GPU all-reduce using ring algorithm
 *
 * Implements distributed reduction across multiple GPUs using the ring
 * all-reduce algorithm. Each GPU contributes its local data, and after
 * the operation, all GPUs have the reduced result.
 *
 * Ring all-reduce algorithm:
 * - Phase 1 (Reduce-scatter): Each GPU sends/receives N-1 times, reducing data
 * - Phase 2 (All-gather): Each GPU sends/receives N-1 times, gathering results
 * - Total: 2N-2 steps, O(N) bandwidth per GPU
 *
 * @example
 * @code
 * // Each GPU has local data, after call all GPUs have the sum
 * DistributedReduce::all_reduce(
 *     send_data, recv_data, count, ReductionOp::Sum);
 * @endcode
 */

#include "common.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::distributed {

/**
 * @class DistributedReduce
 * @brief Multi-GPU all-reduce collective operation
 *
 * Provides static methods for all-reduce operations across all GPUs in the mesh.
 * Uses the ring all-reduce algorithm for optimal bandwidth utilization.
 *
 * @note Single-GPU fallback: If only 1 GPU is available, copies send_data to
 *       recv_data (identity operation).
 */
class DistributedReduce {
public:
    /**
     * @brief All-reduce across all GPUs in the mesh
     *
     * Performs a collective reduction operation where each GPU contributes
     * its local data, and all GPUs receive the same reduced result.
     *
     * @param send_data Input data on each GPU
     * @param recv_data Output buffer (same on all GPUs after call)
     * @param count Number of elements
     * @param op Reduction operation to perform
     * @param dtype CUDA data type (default: CUDA_R_32F)
     *
     * @note send_data and recv_data can be the same pointer for in-place operation
     */
    static void all_reduce(
        const void* send_data,
        void* recv_data,
        size_t count,
        ReductionOp op,
        cudaDataType dtype = cudaDataType::CUDA_R_32F);

    /**
     * @brief Async all-reduce with explicit stream
     *
     * Async variant that returns immediately. The caller must ensure
     * proper synchronization before accessing recv_data.
     *
     * @param send_data Input data on each GPU
     * @param recv_data Output buffer
     * @param count Number of elements
     * @param op Reduction operation
     * @param stream Base CUDA stream (operations use device-specific streams)
     * @param dtype CUDA data type
     */
    static void all_reduce_async(
        const void* send_data,
        void* recv_data,
        size_t count,
        ReductionOp op,
        cudaStream_t stream,
        cudaDataType dtype = cudaDataType::CUDA_R_32F);

    /**
     * @brief Check if multi-GPU operation is needed
     * @return true if device count > 1
     */
    static bool needs_multi_gpu();
};

}  // namespace cuda::distributed
