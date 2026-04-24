#pragma once

/**
 * @file nccl_barrier.h
 * @brief NCCL-based barrier synchronization
 *
 * Provides explicit synchronization points across all devices in the mesh.
 * Uses ncclBarrier internally with safe_nccl_call for error handling.
 *
 * @example
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 *
 * NcclBarrier barrier(ctx);
 * auto result = barrier.barrier_async(ctx.get_stream(device));
 * // All devices in the mesh synchronize here
 * @endcode
 */

#include "cuda/nccl/nccl_collective.h"

#include <cuda_runtime.h>

namespace cuda::nccl {

/**
 * @class NcclBarrier
 * @brief NCCL barrier for multi-device synchronization
 *
 * Synchronizes all devices in the NCCL communicator.
 * Each device must call barrier() for the synchronization to complete.
 *
 * @note Unlike cudaStreamSynchronize which only syncs one device,
 *       NcclBarrier syncs ALL devices in the mesh.
 */
class NcclBarrier : public NcclCollective {
public:
    /**
     * @brief Construct barrier operator
     * @param ctx Initialized NCCL context
     */
    explicit NcclBarrier(NcclContext& ctx);

    // Non-copyable
    NcclBarrier(const NcclBarrier&) = delete;
    NcclBarrier& operator=(const NcclBarrier&) = delete;

    // Movable
    NcclBarrier(NcclBarrier&&) = default;
    NcclBarrier& operator=(NcclBarrier&&) = default;

    /**
     * @brief Async barrier synchronization using device 0's stream
     *
     * @param stream CUDA stream for ordering
     * @return NcclResult with status
     */
    NcclResult barrier_async(cudaStream_t stream);

    /**
     * @brief Async barrier with device-specific stream
     *
     * @param device Device index
     * @param stream CUDA stream
     * @return NcclResult with status
     */
    NcclResult barrier_async(int device, cudaStream_t stream);

    /**
     * @brief Sync barrier (blocking)
     *
     * @param stream CUDA stream
     * @return NcclResult with status
     */
    NcclResult barrier(cudaStream_t stream);

    /**
     * @brief Sync barrier with device selection
     *
     * @param device Device index
     * @param stream CUDA stream
     * @return NcclResult with status
     */
    NcclResult barrier(int device, cudaStream_t stream);
};

}  // namespace cuda::nccl
