/**
 * @file nccl_barrier.cpp
 * @brief NCCL barrier implementation
 */

#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_error.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclBarrier::NcclBarrier(NcclContext& ctx)
    : NcclCollective(ctx) {}

NcclResult NcclBarrier::barrier_async(cudaStream_t stream) {
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
            // NCCL requires device-resident pointers; a host stack `int` here
            // silently corrupts the kernel (illegal memory access) — use a
            // device buffer and reduce the min of 0 to nothing across ranks.
            cuda::memory::Buffer<int> dummy(1);
            int* dummy_ptr = dummy.data();
            return ncclAllReduce(dummy_ptr, dummy_ptr, 1, ncclInt, ncclMin, comm, stream);
        },
        current_comm(),
        30000);
#endif
}

NcclResult NcclBarrier::barrier_async(int device, cudaStream_t stream) {
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
            ncclComm_t comm = get_comm(device);
            int dummy = 0;
            return ncclAllReduce(&dummy, &dummy, 1, ncclInt, ncclMin, comm, stream);
        },
        get_comm(device),
        30000);
#endif
}

NcclResult NcclBarrier::barrier(cudaStream_t stream) {
    auto result = barrier_async(stream);
    if (result.ok()) {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }
    return result;
}

NcclResult NcclBarrier::barrier(int device, cudaStream_t stream) {
    auto result = barrier_async(device, stream);
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
