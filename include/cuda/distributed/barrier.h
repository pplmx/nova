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
 * // All GPUs must call synchronize
 * barrier.synchronize();
 *
 * // Safe to proceed - all GPUs are synchronized
 * @endcode
 */

#include "common.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace cuda::distributed {

/**
 * @class MeshBarrier
 * @brief Multi-GPU barrier synchronization primitive
 *
 * Implements a barrier that synchronizes all GPUs. Uses event-based
 * synchronization with a host thread as coordinator.
 *
 * Thread safety: Safe for concurrent use from multiple host threads,
 * each operating on different GPUs.
 *
 * @note Single-GPU fallback: Calls cudaStreamSynchronize
 * @note Multi-GPU: Uses host polling on events recorded by each GPU
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
     * Uses host thread to poll for GPU events and broadcast proceed signal.
     */
    void synchronize();

    /**
     * @brief Async barrier: records event on each device's stream
     *
     * Non-blocking variant. Caller must wait on all device events before
     * proceeding with dependent operations.
     *
     * @param stream Base CUDA stream (used to record events)
     *
     * @note After calling this, wait on each device's event from MeshStreams
     */
    void synchronize_async(cudaStream_t stream);

    /**
     * @brief Synchronize a specific subset of GPUs
     *
     * @param devices Vector of device indices to synchronize
     */
    void synchronize_devices(const std::vector<int>& devices);

    /**
     * @brief Check if barrier is currently engaged
     * @return true if another thread is waiting at the barrier
     */
    bool is_barriering() const { return barriering_.load(std::memory_order_acquire); }

private:
    int device_count_;
    std::vector<cudaEvent_t> events_;
    std::vector<cudaStream_t> streams_;
    std::atomic<bool> barriering_;
    bool initialized_;
};

}  // namespace cuda::distributed
