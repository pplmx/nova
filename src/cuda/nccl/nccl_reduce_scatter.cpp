/**
 * @file nccl_reduce_scatter.cpp
 * @brief NCCL reduce-scatter implementation
 *
 * Implements stream-based reduce-scatter using NCCL when available.
 * Uses safe_nccl_call wrapper for proper async error detection.
 */

#include "cuda/nccl/nccl_reduce_scatter.h"
#include "cuda/nccl/nccl_error.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclReduceScatter::NcclReduceScatter(NcclContext& ctx)
    : NcclCollective(ctx) {}

NcclResult NcclReduceScatter::reduce_scatter_async(
    const void* send_data,
    void* recv_data,
    size_t recv_count,
    ncclDataType_t dtype,
    ncclRedOp_t op,
    cudaStream_t stream) {

#ifndef NOVA_NCCL_ENABLED
    return NcclResult{.code = ncclInternalError,
                      .error_message = "NCCL not enabled"};
#else
    if (!has_nccl()) {
        return NcclResult{.code = ncclInternalError,
                          .error_message = "NCCL context not initialized"};
    }

    // Per-rank operation: use the current device's communicator. Resolve it up
    // front and keep the "returns NcclResult, never throws" contract.
    ncclComm_t comm;
    try {
        comm = current_comm();
    } catch (const std::exception& e) {
        return NcclResult{.code = ncclInvalidArgument, .error_message = e.what()};
    }

    return safe_nccl_call(
        [&]() {
            return ncclReduceScatter(
                send_data, recv_data, recv_count,
                dtype, op, comm, stream);
        },
        comm,
        30000,
        &ctx_);
#endif
}

NcclResult NcclReduceScatter::reduce_scatter(
    const void* send_data,
    void* recv_data,
    size_t recv_count,
    ncclDataType_t dtype,
    ncclRedOp_t op,
    cudaStream_t stream) {
    auto result = reduce_scatter_async(send_data, recv_data, recv_count, dtype, op, stream);
    if (result.ok()) {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }
    return result;
}

}  // namespace cuda::nccl
