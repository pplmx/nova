#pragma once

/**
 * @file nccl_group.h
 * @brief NCCL group operations for batching multiple collectives
 *
 * Provides NcclGroupHandle for batching NCCL operations to improve
 * performance and prevent cross-collective deadlocks.
 */

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_error.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_reduce_scatter.h"

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <vector>

namespace cuda::nccl {

    /**
     * @class NcclGroupHandle
     * @brief RAII handle for batching multiple NCCL collective operations
     *
     * Batches multiple NCCL calls using ncclGroupStart/ncclGroupEnd
     * to improve performance and prevent deadlocks.
     *
     * @note Uses RAII pattern: operations are executed when the handle
     *       is destroyed, unless execute() is called explicitly.
     *
     * @example
     * @code
     * NcclContext ctx;
     * ctx.initialize();
     *
     * {
     *     NcclGroupHandle group(ctx, stream);
     *     group.add_all_reduce(send.data(), recv.data(), 1024, ncclFloat32, ncclSum);
     *     group.add_broadcast(root.data(), recv2.data(), 1024, ncclFloat32, 0);
     *     group.execute();  // Optional: explicit execution
     * }
     * @endcode
     */
    class NcclGroupHandle {
    public:
        /**
         * @brief Construct group handle and begin NCCL group
         * @param ctx Initialized NCCL context
         * @param stream CUDA stream for all operations
         */
        explicit NcclGroupHandle(NcclContext& ctx, cudaStream_t stream);

        /**
         * @brief Destructor - executes pending operations if not already done
         */
        ~NcclGroupHandle();

        // Non-copyable
        NcclGroupHandle(const NcclGroupHandle&) = delete;
        NcclGroupHandle& operator=(const NcclGroupHandle&) = delete;

        // Movable
        NcclGroupHandle(NcclGroupHandle&&) = default;
        NcclGroupHandle& operator=(NcclGroupHandle&&) = default;

        /**
         * @brief Add all-reduce to the group
         */
        void add_all_reduce(const void* send_data, void* recv_data, size_t count, ncclDataType_t dtype, ncclRedOp_t op);

        /**
         * @brief Add broadcast to the group
         */
        void add_broadcast(const void* send_data, void* recv_data, size_t count, ncclDataType_t dtype, int root);

        /**
         * @brief Add all-gather to the group
         */
        void add_all_gather(const void* send_data, void* recv_data, size_t send_count, ncclDataType_t dtype);

        /**
         * @brief Add reduce-scatter to the group
         */
        void add_reduce_scatter(const void* send_data, void* recv_data, size_t recv_count, ncclDataType_t dtype, ncclRedOp_t op);

        /**
         * @brief Execute all queued operations
         * @return NcclResult with status
         */
        NcclResult execute();

        /**
         * @brief Check if operations are pending
         */
        [[nodiscard]] bool has_operations() const noexcept;

        /**
         * @brief Get number of queued operations
         */
        [[nodiscard]] size_t operation_count() const noexcept;

    private:
        NcclContext& ctx_;
        cudaStream_t stream_;
        std::vector<std::function<NcclResult()>> operations_;
        bool executed_ = false;

        NcclResult execute_internal();
    };

}  // namespace cuda::nccl
