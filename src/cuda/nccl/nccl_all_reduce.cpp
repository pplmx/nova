/**
 * @file nccl_all_reduce.cpp
 * @brief NCCL all-reduce implementation
 *
 * Implements stream-based all-reduce using NCCL when available.
 * Uses safe_nccl_call wrapper for proper async error detection.
 */

#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_error.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclAllReduce::NcclAllReduce(NcclContext& ctx)
    : NcclCollective(ctx) {}

NcclResult NcclAllReduce::all_reduce_async(
    const void* send_data,
    void* recv_data,
    size_t count,
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

    // Launch all-reduce across all devices in the communicator
    // Each GPU provides its local send_data, NCCL handles the ring algorithm
    // Result is written to recv_data on each GPU
    return safe_nccl_call(
        [&]() {
            // Use device 0's communicator (valid for all devices in the group)
            ncclComm_t comm = get_comm(0);
            return ncclAllReduce(
                send_data, recv_data, count,
                dtype, op, comm, stream);
        },
        get_comm(0),
        30000);
#endif
}

NcclResult NcclAllReduce::all_reduce(
    const void* send_data,
    void* recv_data,
    size_t count,
    ncclDataType_t dtype,
    ncclRedOp_t op,
    cudaStream_t stream) {
    auto result = all_reduce_async(send_data, recv_data, count, dtype, op, stream);
    if (result.ok()) {
        // Wait for completion
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }
    return result;
}

ncclRedOp_t NcclAllReduce::to_nccl_op(::cuda::distributed::ReductionOp op) {
    switch (op) {
        case ::cuda::distributed::ReductionOp::Sum:
            return ncclSum;
        case ::cuda::distributed::ReductionOp::Min:
            return ncclMin;
        case ::cuda::distributed::ReductionOp::Max:
            return ncclMax;
        case ::cuda::distributed::ReductionOp::Product:
            return ncclProd;
        default:
            return ncclSum;
    }
}

ncclDataType_t NcclAllReduce::to_nccl_dtype(cudaDataType dtype) {
    // Map CUDA dtype to NCCL dtype
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
