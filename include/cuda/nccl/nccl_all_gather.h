#pragma once

/**
 * @file nccl_all_gather.h
 * @brief NCCL-based all-gather collective operation
 *
 * Provides stream-based all-gather using NCCL when available,
 * falling back to P2P implementation when not.
 */

#include "cuda/nccl/nccl_collective.h"
#include "cuda/nccl/nccl_types.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::nccl {

    /**
     * @class NcclAllGather
     * @brief NCCL all-gather with stream ordering and error handling
     *
     * Gathers data from all ranks into a contiguous output buffer.
     * Each rank's contribution is placed at its corresponding offset.
     *
     * @example
     * @code
     * NcclContext ctx;
     * ctx.initialize();
     *
     * NcclAllGather gather(ctx);
     * size_t recv_count = 1024;
     * size_t recv_size = NcclAllGather::required_recv_buffer_size(recv_count, ctx.device_count());
     * cuda::memory::Buffer<float> recv_buffer(recv_size);
     *
     * auto result = gather.all_gather_async(
     *     send_buffer.data(), recv_buffer.data(), recv_count,
     *     ncclFloat32, ctx.get_stream(0));
     * @endcode
     */
    class NcclAllGather : public NcclCollective {
    public:
        /**
         * @brief Construct all-gather operator
         * @param ctx Initialized NCCL context
         */
        explicit NcclAllGather(NcclContext& ctx);

        // Non-copyable
        NcclAllGather(const NcclAllGather&) = delete;
        NcclAllGather& operator=(const NcclAllGather&) = delete;

        // Not movable. These read `= default` under a `// Movable` comment, which claimed a
        // capability the type does not have: NcclCollective declares a destructor and deleted
        // copies but no moves, so it has no move operation and any derived `= default` move is
        // defined as deleted. Nothing in the tree moves one; declaring `= delete` puts the fact
        // and its reason at the declaration instead of leaving it to a call site's error.
        NcclAllGather(NcclAllGather&&) = delete;
        NcclAllGather& operator=(NcclAllGather&&) = delete;

        /**
         * @brief Async all-gather across all devices
         *
         * @param send_data Input buffer (device pointer, send_count elements)
         * @param recv_data Output buffer (device pointer, send_count * device_count elements)
         * @param send_count Number of elements each rank contributes
         * @param dtype NCCL data type
         * @param stream CUDA stream for ordering
         * @return NcclResult with status
         */
        NcclResult all_gather_async(const void* send_data, void* recv_data, size_t send_count, ncclDataType_t dtype, cudaStream_t stream);

        /**
         * @brief Sync all-gather (blocking)
         *
         * @param send_data Input buffer (device pointer)
         * @param recv_data Output buffer (device pointer)
         * @param send_count Element count per rank
         * @param dtype NCCL data type
         * @param stream CUDA stream
         * @return NcclResult with status
         */
        NcclResult all_gather(const void* send_data, void* recv_data, size_t send_count, ncclDataType_t dtype, cudaStream_t stream);

        /**
         * @brief Calculate required receive buffer size
         * @param send_count Elements per rank
         * @param device_count Number of devices
         * @return Total buffer size in elements
         */
        static constexpr size_t required_recv_buffer_size(size_t send_count, int device_count) { return send_count * device_count; }
    };

}  // namespace cuda::nccl
