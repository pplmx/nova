#pragma once

/**
 * @file nccl_collective.h
 * @brief Base class for NCCL collective operations
 *
 * Provides common infrastructure for NCCL collectives:
 * - NcclContext reference for communicator access
 * - safe_nccl_call wrapper for error handling
 * - Stream ordering for async operations
 *
 * @example
 * @code
 * class MyCollective : public NcclCollective {
 * public:
 *     MyCollective(NcclContext& ctx) : NcclCollective(ctx) {}
 *
 *     NcclResult do_operation(cudaStream_t stream) {
 *         return safe_nccl_call(
 *             [&]() { return ncclOp(comm, stream); },
 *             get_comm(0),
 *             30000);
 *     }
 * };
 * @endcode
 */

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_error.h"

#include <cuda_runtime.h>

namespace cuda::nccl {

/**
 * @class NcclCollective
 * @brief Base class for NCCL collective operations
 *
 * Provides common infrastructure for NCCL collectives:
 * - NcclContext reference for communicator access
 * - Helper methods for communicator and stream access
 * - Integration with safe_nccl_call error handling
 *
 * @note Derived classes must ensure NcclContext is initialized
 *       before use. Call has_nccl() to verify availability.
 */
class NcclCollective {
public:
    /**
     * @brief Construct with NCCL context
     * @param ctx Reference to initialized NcclContext
     */
    explicit NcclCollective(NcclContext& ctx);

    /**
     * @brief Destructor
     */
    virtual ~NcclCollective() = default;

    // Non-copyable
    NcclCollective(const NcclCollective&) = delete;
    NcclCollective& operator=(const NcclCollective&) = delete;

    /**
     * @brief Check if NCCL is available
     * @return true if context has valid NCCL
     */
    [[nodiscard]] bool has_nccl() const;

    /**
     * @brief Get communicator for a device
     * @param device Device index
     * @return NCCL communicator
     */
    [[nodiscard]] ncclComm_t get_comm(int device) const;

    /**
     * @brief Get communicator for the currently-active device
     *
     * NCCL collects are per-rank: each rank must issue the operation with its
     * own communicator (the one owned by the device the caller's buffers and
     * stream live on). Use this instead of hard-coding rank 0's communicator.
     *
     * @return NCCL communicator for the current device
     */
    [[nodiscard]] ncclComm_t current_comm() const;

    /**
     * @brief Get stream for a device
     * @param device Device index
     * @return CUDA stream
     */
    [[nodiscard]] cudaStream_t get_stream(int device) const;

    /**
     * @brief Get device count in the NCCL group
     * @return Number of devices
     */
    [[nodiscard]] int device_count() const;

protected:
    /** Reference to NCCL context (non-owning) */
    NcclContext& ctx_;
};

}  // namespace cuda::nccl
