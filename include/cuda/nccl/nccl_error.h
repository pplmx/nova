#pragma once

/**
 * @file nccl_error.h
 * @brief NCCL error handling and async polling infrastructure
 *
 * Provides safe NCCL operation wrappers that poll for async errors
 * before blocking on cudaStreamSynchronize. This prevents indefinite
 * hangs from asynchronous NCCL errors (per D-02).
 *
 * @example
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 *
 * auto result = safe_nccl_call(
 *     [&]() {
 *         return ncclAllReduce(send, recv, count, dtype, op, comm, stream);
 *     },
 *     ctx.get_comm(0),
 *     30000  // 30 second timeout
 * );
 *
 * if (!result) {
 *     std::cerr << "NCCL error: " << result.error_message << "\n";
 * }
 * @endcode
 */

#include "cuda/nccl/nccl_types.h"

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <thread>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

// Included for the complete NcclContext type: safe_nccl_call (template)
// calls ctx->mark_comm_aborted() and a pointer-to-incomplete member call does
// not resolve at template definition time. nccl_context.h does not include
// nccl_error.h, so there is no include cycle.
#include "cuda/nccl/nccl_context.h"

namespace cuda::nccl {

/**
 * @brief Result of a safe NCCL operation
 *
 * Provides detailed status about NCCL operations including:
 * - Return code
 * - Timeout status
 * - Async error detection
 * - Human-readable error message
 */
struct NcclResult {
    /** NCCL result code */
    ncclResult_t code = ncclSuccess;

    /** Whether operation timed out */
    bool timed_out = false;

    /** Whether an async error was detected */
    bool async_error = false;

    /** Human-readable error message */
    std::string error_message;

    /**
     * @brief Check if operation was successful
     * @return true if no errors occurred
     */
    [[nodiscard]] bool ok() const {
        return code == ncclSuccess && !timed_out && !async_error;
    }

    /**
     * @brief Explicit bool conversion for convenience
     */
    explicit operator bool() const { return ok(); }
};

/**
 * @brief Wrapper for NCCL calls that polls for async errors
 *
 * Executes the NCCL call, then polls ncclCommGetAsyncError() until
 * completion or timeout. This prevents hangs from async NCCL errors.
 *
 * Unlike bare cudaStreamSynchronize(), this function detects errors
 * that occur asynchronously on the GPU before blocking indefinitely.
 *
 * @tparam Fn Callable type (typically lambda or function pointer)
 * @param fn NCCL call to execute (must return ncclResult_t)
 * @param comm NCCL communicator for async error polling
 * @param timeout_ms Maximum time to wait for completion (default: 30s)
 * @return NcclResult with status and error details
 *
 * @note Per D-02: Automatic polling via ncclCommGetAsyncError()
 *
 * @example
 * @code
 * auto result = safe_nccl_call(
 *     [&]() {
 *         return ncclAllReduce(send, recv, count, dtype, op, comm, stream);
 *     },
 *     comm,
 *     30000
 * );
 *
 * if (!result) {
 *     if (result.timed_out) {
 *         std::cerr << "Operation timed out\n";
 *     } else if (result.async_error) {
 *         std::cerr << "Async error: " << result.error_message << "\n";
 *     }
 * }
 * @endcode
 */
template<typename Fn>
NcclResult safe_nccl_call(Fn&& fn, ncclComm_t comm, int timeout_ms = 30000,
                          NcclContext* ctx = nullptr) {
    NcclResult result;

#if NOVA_NCCL_ENABLED
    // Execute the NCCL call
    result.code = fn();

    // If call failed immediately, return. The communicator is NOT aborted
    // here — this is a synchronous caller error (e.g. ncclInvalidArgument),
    // so the comm remains usable and the shared context must NOT be marked
    // broken (a deliberate negative test would otherwise poison every later
    // NCCL user in the process).
    if (result.code != ncclSuccess && result.code != ncclInProgress) {
        result.error_message = ncclGetErrorString(result.code);
        return result;
    }

    // Poll for completion with async error detection
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        // Check for async errors on the communicator
        ncclResult_t async_err;
        ncclResult_t poll_result = ncclCommGetAsyncError(comm, &async_err);

        if (poll_result != ncclSuccess) {
            result.code = poll_result;
            result.error_message = "ncclCommGetAsyncError failed";
            return result;
        }

        if (async_err != ncclSuccess && async_err != ncclInProgress) {
            result.async_error = true;
            result.code = async_err;
            result.error_message = ncclGetErrorString(async_err);

            // Abort the communicator to prevent further hangs; a dead
            // communicator must never be silently reused, so flag it on the
            // owning context (has_nccl() goes false, later collectives fail
            // fast instead of hanging).
            ncclCommAbort(comm);
            if (ctx != nullptr) {
                ctx->mark_comm_aborted(comm);
            }
            return result;
        }

        if (async_err == ncclSuccess) {
            // Operation completed successfully
            result.code = ncclSuccess;
            return result;
        }

        // Still in progress, yield to other threads
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Timeout
    result.timed_out = true;
    result.code = static_cast<ncclResult_t>(-2);
    result.error_message = "NCCL operation timed out after " +
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

/**
 * @brief Wait for NCCL operation on a specific stream with polling
 *
 * Similar to safe_nccl_call but focuses on stream synchronization
 * with async error detection. Use this when the NCCL operation
 * has already been launched.
 *
 * @param stream CUDA stream to query
 * @param comm NCCL communicator
 * @param timeout_ms Timeout in milliseconds
 * @return NcclResult with status
 */
NcclResult safe_stream_wait(ncclComm_t comm, cudaStream_t stream,
                             int timeout_ms = 30000,
                             NcclContext* ctx = nullptr);

}  // namespace cuda::nccl
