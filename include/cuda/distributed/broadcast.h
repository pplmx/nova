#pragma once

/**
 * @file broadcast.h
 * @brief Multi-GPU broadcast operation
 *
 * Broadcasts data from a root GPU to all other GPUs in the mesh.
 * Each non-root GPU receives a copy of the root's data.
 *
 * @example
 * @code
 * // Broadcast from GPU 0 to all other GPUs
 * DistributedBroadcast::broadcast(data, count, root=0);
 * @endcode
 */

#include "common.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::distributed {

/**
 * @class DistributedBroadcast
 * @brief Multi-GPU broadcast collective operation
 *
 * Distributes data from a root GPU to all other GPUs in the mesh.
 * Uses P2P peer copy with event-based synchronization.
 *
 * @note Single-GPU fallback: No-op, data is already in place.
 * @note Non-root GPUs must call this function but data pointer is ignored.
 */
class DistributedBroadcast {
public:
    /**
     * @brief Broadcast data from root GPU to all other GPUs
     *
     * @param data Data buffer (source on root, receives copy on others)
     * @param count Number of elements
     * @param root Source GPU rank (default: 0)
     * @param dtype CUDA data type (default: CUDA_R_32F)
     *
     * @note All GPUs must call this function with the same root and count
     */
    static void broadcast(
        void* data,
        size_t count,
        int root = 0,
        cudaDataType dtype = cudaDataType::CUDA_R_32F);

    /**
     * @brief Async broadcast with explicit stream
     *
     * @param data Data buffer
     * @param count Number of elements
     * @param root Source GPU rank
     * @param stream CUDA stream for the operation
     * @param dtype CUDA data type
     *
     * @note All GPUs must call this function with the same parameters
     */
    static void broadcast_async(
        void* data,
        size_t count,
        int root,
        cudaStream_t stream,
        cudaDataType dtype = cudaDataType::CUDA_R_32F);
};

}  // namespace cuda::distributed
