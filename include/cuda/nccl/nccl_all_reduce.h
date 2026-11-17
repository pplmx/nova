#pragma once

/**
 * @file nccl_all_reduce.h
 * @brief NCCL-based all-reduce collective operation
 *
 * Provides stream-based all-reduce using NCCL when available,
 * falling back to the legacy P2P implementation when not.
 */

#include "cuda/nccl/nccl_collective.h"
#include "cuda/nccl/nccl_types.h"

#include <cuda_runtime.h>

#include <functional>
#include <optional>

namespace cuda::nccl {

/**
 * @class NcclAllReduce
 * @brief NCCL all-reduce with stream ordering and error handling
 *
 * Implements all-reduce across all GPUs in the mesh using NCCL.
 * Uses safe_nccl_call for proper async error detection.
 *
 * @example
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 *
 * NcclAllReduce reduce(ctx);
 * auto result = reduce.all_reduce_async(
 *     send_data, recv_data, count,
 *     ncclRedOp_t::ncclSum,
 *     ncclDataType_t::ncclFloat32,
 *     ctx.get_stream(0)
 * );
 * @endcode
 */
class NcclAllReduce : public NcclCollective {
public:
    /**
     * @brief Construct all-reduce operator
     * @param ctx Initialized NCCL context
     */
    explicit NcclAllReduce(NcclContext& ctx);

    // Non-copyable
    NcclAllReduce(const NcclAllReduce&) = delete;
    NcclAllReduce& operator=(const NcclAllReduce&) = delete;

    // Movable
    NcclAllReduce(NcclAllReduce&&) = default;
    NcclAllReduce& operator=(NcclAllReduce&&) = default;

    /**
     * @brief Async all-reduce across all devices
     *
     * @param send_data Input buffer (device-local)
     * @param recv_data Output buffer (identical on all devices after call)
     * @param count Number of elements
     * @param dtype NCCL data type
     * @param op NCCL reduction operation
     * @param stream CUDA stream for ordering
     * @return NcclResult with status (use result.ok() to check)
     */
    NcclResult all_reduce_async(
        const void* send_data,
        void* recv_data,
        size_t count,
        ncclDataType_t dtype,
        ncclRedOp_t op,
        cudaStream_t stream);

    /**
     * @brief Sync all-reduce (blocking)
     *
     * Convenience wrapper that waits for completion.
     *
     * @param send_data Input buffer (device pointer)
     * @param recv_data Output buffer (device pointer)
     * @param count Element count
     * @param dtype NCCL data type
     * @param op Reduction operation
     * @param stream CUDA stream
     * @return NcclResult with status
     */
    NcclResult all_reduce(
        const void* send_data,
        void* recv_data,
        size_t count,
        ncclDataType_t dtype,
        ncclRedOp_t op,
        cudaStream_t stream);

    /**
     * @brief Convert ReductionOp to NCCL op
     * @param op cuda::distributed ReductionOp
     * @return ncclRedOp_t
     */
    static ncclRedOp_t to_nccl_op(::cuda::distributed::ReductionOp op);

    /**
     * @brief Convert CUDA dtype to NCCL dtype
     * @param dtype CUDA data type
     * @return ncclDataType_t
     */
    static ncclDataType_t to_nccl_dtype(cudaDataType dtype);
};

}  // namespace cuda::nccl
