#pragma once

/**
 * @file barrier.h
 * @brief Multi-GPU barrier synchronization
 *
 * Synchronizes all GPUs in the mesh such that no GPU proceeds past the
 * barrier until all GPUs have reached it.
 *
 * @example
 * @code
 * MeshBarrier barrier;
 *
 * // All GPUs must call synchronize (one thread per rank)
 * barrier.synchronize();
 *
 * // Safe to proceed - all GPUs are synchronized
 * @endcode
 */

#include "common.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <vector>

namespace cuda::distributed {

/**
 * @class MeshBarrier
 * @brief Multi-GPU barrier synchronization primitive
 *
 * Implements a barrier that synchronizes all GPUs. Converged onto the verified
 * NCCL layer in v2.18: the multi-GPU path is a genuine per-rank
 * `NcclBarrier` collective (see decision-v18-meshbarrier-nccl), so every rank
 * contributing to the barrier MUST enter it from its own thread pinned with
 * `cudaSetDevice` — the same collective contract as DistributedReduce /
 * DistributedAllGather / DistributedBroadcast. The pre-v2.18 per-instance host
 * event-poll never enforced cross-rank arrival (events on empty internal
 * streams fire immediately), so ranks were released before all had reached the
 * barrier — see HYP-001 / EV-001.
 *
 * Thread safety: Safe for concurrent use from one thread per rank; each rank
 * thread enters the collective on its current device.
 *
 * @note Single-GPU fallback: Calls cudaDeviceSynchronize
 * @note Multi-GPU: Genuine per-rank NCCL barrier
 */
class MeshBarrier {
public:
    /**
     * @brief Construct a barrier for all GPUs in the mesh
     */
    MeshBarrier();

    /**
     * @brief Destructor
     */
    ~MeshBarrier();

    // Non-copyable, non-movable
    MeshBarrier(const MeshBarrier&) = delete;
    MeshBarrier& operator=(const MeshBarrier&) = delete;
    MeshBarrier(MeshBarrier&&) = delete;
    MeshBarrier& operator=(MeshBarrier&&) = delete;

    /**
     * @brief Synchronize all GPUs: all must arrive before any proceeds
     *
     * Blocking call - returns only when all GPUs have reached the barrier.
     * Every rank must call this (one thread per device, pinned with
     * cudaSetDevice) for the collective to complete.
     */
    void synchronize();

    /**
     * @brief Async barrier: NCCL barrier ordered on the given stream
     *
     * Every rank must call this for the collective to complete. The barrier
     * is ordered on @p stream; the caller synchronizes that stream before
     * proceeding with dependent operations.
     *
     * @param stream CUDA stream on which the barrier is ordered
     */
    void synchronize_async(cudaStream_t stream);

    /**
     * @brief Synchronize a specific subset of GPUs
     *
     * @param devices Vector of device indices to synchronize.
     *
     * Only the full NCCL group is supported: a proper subset would require a
     * sub-communicator that NcclContext does not expose, and any rank not
     * entering the collective deadlocks the group communicator. A request for
     * a proper subset throws NcclException with that explanation.
     */
    void synchronize_devices(const std::vector<int>& devices);

    /**
     * @brief Check if barrier is currently engaged
     * @return true if a rank thread is waiting at the barrier
     */
    bool is_barriering() const { return barriering_.load(std::memory_order_acquire); }

private:
    int device_count_;
    std::atomic<bool> barriering_;
    bool initialized_;
};

}  // namespace cuda::distributed
