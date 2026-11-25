#pragma once

/**
 * @file nccl_reduce_scatter.h
 * @brief NCCL-based reduce-scatter collective operation
 *
 * Provides stream-based reduce-scatter using NCCL when available.
 */

#include "cuda/nccl/nccl_collective.h"
#include "cuda/nccl/nccl_types.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::nccl {

    /**
     * @class NcclReduceScatter
     * @brief NCCL reduce-scatter with stream ordering and error handling
     *
     * Reduces data across all ranks and scatters the result, with each rank
     * receiving a portion of the reduced output.
     *
     * @example
     * @code
     * NcclContext ctx;
     * ctx.initialize();
     *
     * NcclReduceScatter reduce_scatter(ctx);
     * size_t recv_count = 1024;
     * cuda::memory::Buffer<float> recv_buffer(recv_count);
     *
     * auto result = reduce_scatter.reduce_scatter_async(
     *     send_buffer.data(), recv_buffer.data(), recv_count,
     *     ncclFloat32, ncclSum, ctx.get_stream(0));
     * @endcode
     */
    class NcclReduceScatter : public NcclCollective {
    public:
        /**
         * @brief Construct reduce-scatter operator
         * @param ctx Initialized NCCL context
         */
        explicit NcclReduceScatter(NcclContext& ctx);

        // Non-copyable
        NcclReduceScatter(const NcclReduceScatter&) = delete;
        NcclReduceScatter& operator=(const NcclReduceScatter&) = delete;

        // Not movable. These read `= default` under a `// Movable` comment, which claimed a
        // capability the type does not have: NcclCollective declares a destructor and deleted
        // copies but no moves, so it has no move operation and any derived `= default` move is
        // defined as deleted. Nothing in the tree moves one; declaring `= delete` puts the fact
        // and its reason at the declaration instead of leaving it to a call site's error.
        NcclReduceScatter(NcclReduceScatter&&) = delete;
        NcclReduceScatter& operator=(NcclReduceScatter&&) = delete;

        /**
         * @brief Async reduce-scatter across all devices
         *
         * @param send_data Input buffer (device pointer, recv_count * device_count elements)
         * @param recv_data Output buffer (device pointer, recv_count elements)
         * @param recv_count Number of elements to receive (per rank)
         * @param dtype NCCL data type
         * @param op NCCL reduction operation
         * @param stream CUDA stream for ordering
         * @return NcclResult with status
         */
        NcclResult reduce_scatter_async(const void* send_data, void* recv_data, size_t recv_count, ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t stream);

        /**
         * @brief Sync reduce-scatter (blocking)
         *
         * @param send_data Input buffer (device pointer)
         * @param recv_data Output buffer (device pointer)
         * @param recv_count Element count per rank
         * @param dtype NCCL data type
         * @param op Reduction operation
         * @param stream CUDA stream
         * @return NcclResult with status
         */
        NcclResult reduce_scatter(const void* send_data, void* recv_data, size_t recv_count, ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t stream);

        /**
         * @brief Calculate required send buffer size
         * @param recv_count Elements per rank after scatter
         * @param device_count Number of devices
         * @return Total buffer size in elements
         */
        static constexpr size_t required_send_buffer_size(size_t recv_count, int device_count) { return recv_count * device_count; }
    };

}  // namespace cuda::nccl
