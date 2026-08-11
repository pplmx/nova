/**
 * @file nccl_all_gather.cpp
 * @brief NCCL all-gather implementation
 *
 * Implements stream-based all-gather using NCCL when available.
 * Uses safe_nccl_call wrapper for proper async error detection.
 */

#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_error.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclAllGather::NcclAllGather(NcclContext& ctx)
    : NcclCollective(ctx) {}

NcclResult NcclAllGather::all_gather_async(
    const void* send_data,
    void* recv_data,
    size_t send_count,
    ncclDataType_t dtype,
    cudaStream_t stream) {

#ifndef NOVA_NCCL_ENABLED
    return NcclResult{.code = ncclInternalError,
                      .error_message = "NCCL not enabled"};
#else
    if (!has_nccl()) {
        return NcclResult{.code = ncclInternalError,
                          .error_message = "NCCL context not initialized"};
    }

    return safe_nccl_call(
        [&]() {
            // Per-rank operation: use the current device's communicator.
            ncclComm_t comm = current_comm();
            return ncclAllGather(
                send_data, recv_data, send_count,
                dtype, comm, stream);
        },
        current_comm(),
        30000);
#endif
}

NcclResult NcclAllGather::all_gather(
    const void* send_data,
    void* recv_data,
    size_t send_count,
    ncclDataType_t dtype,
    cudaStream_t stream) {
    auto result = all_gather_async(send_data, recv_data, send_count, dtype, stream);
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
