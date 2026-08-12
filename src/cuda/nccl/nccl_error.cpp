/**
 * @file nccl_error.cpp
 * @brief NCCL error handling implementation
 */

#include "cuda/nccl/nccl_error.h"
#include "cuda/nccl/nccl_context.h"

#include <thread>

namespace cuda::nccl {

NcclResult safe_stream_wait(ncclComm_t comm, cudaStream_t stream,
                             int timeout_ms, NcclContext* ctx) {
    NcclResult result;

#if NOVA_NCCL_ENABLED
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        cudaError_t cuda_err = cudaStreamQuery(stream);

        if (cuda_err == cudaSuccess) {
            // Stream completed successfully
            result.code = ncclSuccess;
            return result;
        }

        if (cuda_err != cudaErrorNotReady) {
            // Actual CUDA error
            result.code = ncclInternalError;
            result.error_message = cudaGetErrorString(cuda_err);
            return result;
        }

        // Check for async errors on the communicator
        ncclResult_t async_err;
        if (ncclCommGetAsyncError(comm, &async_err) != ncclSuccess) {
            result.code = ncclInternalError;
            result.error_message = "ncclCommGetAsyncError failed";
            return result;
        }

        if (async_err != ncclSuccess && async_err != ncclInProgress) {
            // Async error detected
            result.async_error = true;
            result.code = async_err;
            result.error_message = ncclGetErrorString(async_err);

            // Abort communicator to prevent further hangs; flag the dead comm
            // on its owning context so later collectives fail fast instead of
            // silently reusing it.
            ncclCommAbort(comm);
            if (ctx != nullptr) {
                ctx->mark_comm_aborted(comm);
            }
            return result;
        }

        // Still in progress, yield to other threads
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Timeout
    result.timed_out = true;
    result.code = static_cast<ncclResult_t>(-2);
    result.error_message = "Stream wait timed out after " +
                           std::to_string(timeout_ms) + "ms";

    // Abort communicator on timeout; same never-reuse-dead-comm flagging.
    ncclCommAbort(comm);
    if (ctx != nullptr) {
        ctx->mark_comm_aborted(comm);
    }

#else
    result.code = static_cast<ncclResult_t>(-1);
    result.error_message = "NCCL not enabled";
#endif

    return result;
}

}  // namespace cuda::nccl
