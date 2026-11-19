#pragma once

/**
 * @file nccl_ops.h
 * @brief Unified NCCL operations API
 *
 * Provides a unified interface for NCCL collective operations with
 * automatic fallback to P2P implementations when NCCL is unavailable.
 */

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_reduce_scatter.h"
#include "cuda/nccl/nccl_group.h"

namespace cuda::nccl {

    /**
     * @class NcclOps
     * @brief Unified NCCL operations interface
     *
     * Provides a single entry point for all NCCL collective operations.
     * Automatically handles NCCL availability and fallback paths.
     *
     * @example
     * @code
     * NcclOps ops(ctx);
     *
     * // All-gather
     * auto result1 = ops.all_gather(send.data(), recv.data(), 1024, ncclFloat32, stream);
     *
     * // Reduce-scatter
     * auto result2 = ops.reduce_scatter(send.data(), recv.data(), 1024, ncclFloat32, ncclSum, stream);
     * @endcode
     */
    class NcclOps {
    public:
        /**
         * @brief Construct with NCCL context
         * @param ctx Initialized NCCL context
         */
        explicit NcclOps(NcclContext& ctx);

        // Non-copyable
        NcclOps(const NcclOps&) = delete;
        NcclOps& operator=(const NcclOps&) = delete;

        /**
         * @brief Check if NCCL is available
         */
        [[nodiscard]] bool has_nccl() const;

        /**
         * @brief All-gather operation
         */
        NcclResult all_gather(const void* send_data, void* recv_data, size_t send_count, ncclDataType_t dtype, cudaStream_t stream);

        /**
         * @brief Reduce-scatter operation
         */
        NcclResult reduce_scatter(const void* send_data, void* recv_data, size_t recv_count, ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t stream);

        /**
         * @brief Get device count
         */
        [[nodiscard]] int device_count() const;

    private:
        NcclContext& ctx_;
        std::unique_ptr<NcclAllGather> all_gather_;
        std::unique_ptr<NcclReduceScatter> reduce_scatter_;
    };

}  // namespace cuda::nccl
