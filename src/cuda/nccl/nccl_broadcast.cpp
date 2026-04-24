/**
 * @file nccl_broadcast.cpp
 * @brief NCCL broadcast implementation
 */

#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_error.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclBroadcast::NcclBroadcast(NcclContext& ctx)
    : NcclCollective(ctx) {}

NcclResult NcclBroadcast::broadcast_async(
    const void* data,
    void* recv_data,
    size_t count,
    ncclDataType_t dtype,
    int root_rank,
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
            ncclComm_t comm = get_comm(root_rank);
            return ncclBroadcast(
                data, recv_data, count,
                dtype, root_rank, comm, stream);
        },
        get_comm(root_rank),
        30000);
#endif
}

NcclResult NcclBroadcast::broadcast(
    const void* data,
    void* recv_data,
    size_t count,
    ncclDataType_t dtype,
    int root_rank,
    cudaStream_t stream) {
    auto result = broadcast_async(data, recv_data, count, dtype, root_rank, stream);
    if (result.ok()) {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }
    return result;
}

ncclDataType_t NcclBroadcast::to_nccl_dtype(cudaDataType dtype) {
    switch (dtype) {
        case CUDA_R_32F:
            return ncclFloat32;
        case CUDA_R_64F:
            return ncclFloat64;
        case CUDA_R_16F:
            return ncclFloat16;
        case CUDA_R_8I:
            return ncclInt8;
        case CUDA_R_8U:
            return ncclUint8;
        case CUDA_R_32I:
            return ncclInt32;
        case CUDA_R_32U:
            return ncclUint32;
        case CUDA_R_64I:
            return ncclInt64;
        case CUDA_R_64U:
            return ncclUint64;
        default:
            return ncclFloat32;
    }
}

}  // namespace cuda::nccl
