#pragma once

/**
 * @file nccl_broadcast.h
 * @brief NCCL-based broadcast collective operation
 *
 * Broadcasts data from one device (root) to all other devices in the mesh.
 * Used for weight synchronization in distributed training.
 *
 * @example
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 *
 * NcclBroadcast broadcast(ctx);
 * auto result = broadcast.broadcast_async(
 *     weight_data,  // Source on root device
 *     recv_data,
 *     count,
 *     ncclDataType_t::ncclFloat32,
 *     root_rank,    // Device 0 typically
 *     ctx.get_stream(device)
 * );
 * @endcode
 */

#include "cuda/nccl/nccl_collective.h"

#include <cuda_runtime.h>

namespace cuda::nccl {

/**
 * @class NcclBroadcast
 * @brief NCCL broadcast for weight/data distribution
 *
 * Broadcasts data from one root device to all other devices.
 * Uses ncclBroadcast internally with safe_nccl_call for error handling.
 */
class NcclBroadcast : public NcclCollective {
public:
    /**
     * @brief Construct broadcast operator
     * @param ctx Initialized NCCL context
     */
    explicit NcclBroadcast(NcclContext& ctx);

    // Non-copyable
    NcclBroadcast(const NcclBroadcast&) = delete;
    NcclBroadcast& operator=(const NcclBroadcast&) = delete;

    // Movable
    NcclBroadcast(NcclBroadcast&&) = default;
    NcclBroadcast& operator=(NcclBroadcast&&) = default;

    /**
     * @brief Async broadcast from root device to all others
     *
     * @param data Send buffer on root device
     * @param recv_data Receive buffer on all devices
     * @param count Number of elements
     * @param dtype NCCL data type
     * @param root_rank Root device rank (typically 0)
     * @param stream CUDA stream for ordering
     * @return NcclResult with status
     */
    NcclResult broadcast_async(
        const void* data,
        void* recv_data,
        size_t count,
        ncclDataType_t dtype,
        int root_rank,
        cudaStream_t stream);

    /**
     * @brief Sync broadcast (blocking)
     */
    NcclResult broadcast(
        const void* data,
        void* recv_data,
        size_t count,
        ncclDataType_t dtype,
        int root_rank,
        cudaStream_t stream);

    /**
     * @brief Convert CUDA dtype to NCCL dtype
     */
    static ncclDataType_t to_nccl_dtype(cudaDataType dtype);
};

}  // namespace cuda::nccl
