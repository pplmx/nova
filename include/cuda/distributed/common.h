#pragma once

/**
 * @file common.h
 * @brief Common infrastructure for distributed CUDA operations
 *
 * Provides shared types, utilities, and stream management for multi-GPU
 * collective operations including all-reduce, broadcast, all-gather, and barrier.
 *
 * @example
 * @code
 * // Access per-device streams for async operations
 * auto& streams = MeshStreams::instance();
 * cudaStream_t stream = streams.get_stream(device_id);
 * @endcode
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace cuda::distributed {

/**
 * @enum ReductionOp
 * @brief Supported reduction operations for collective primitives
 */
enum class ReductionOp {
    Sum,
    Min,
    Max,
    Product
};

/**
 * @class MeshStreams
 * @brief Manages per-device CUDA streams and events for collective operations
 *
 * CUDA streams are device-local, so each GPU requires its own stream.
 * This class maintains a vector of streams, one per device, along with
 * pre-allocated events for synchronization.
 *
 * @note Thread-safe singleton pattern using Meyer's singleton.
 */
class MeshStreams {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to MeshStreams
     */
    static MeshStreams& instance();

    /**
     * @brief Destructor cleans up all streams and events
     */
    ~MeshStreams();

    // Non-copyable, non-movable
    MeshStreams(const MeshStreams&) = delete;
    MeshStreams& operator=(const MeshStreams&) = delete;
    MeshStreams(MeshStreams&&) = delete;
    MeshStreams& operator=(MeshStreams&&) = delete;

    /**
     * @brief Get the CUDA stream for a specific device
     * @param device Device index
     * @return CUDA stream handle
     * @note Must call initialize() before first use
     */
    cudaStream_t get_stream(int device);

    /**
     * @brief Get an event on a specific device's stream
     * @param device Device index
     * @return CUDA event handle
     * @note Events are pre-allocated and reused
     */
    cudaEvent_t get_event(int device);

    /**
     * @brief Synchronize all device streams
     *
     * Calls cudaStreamSynchronize on each device's stream in sequence.
     * For true multi-device synchronization, use MeshBarrier instead.
     */
    void synchronize_all();

    /**
     * @brief Wait for event from src_device on dst_device's stream
     *
     * This is the core synchronization primitive for cross-GPU operations.
     * The event must have been recorded on src_device's stream.
     *
     * @param dst_device Destination device that will wait
     * @param src_device Source device that recorded the event
     * @param event Event to wait on
     */
    void wait_event(int dst_device, int src_device, cudaEvent_t event);

    /**
     * @brief Returns the number of devices
     * @return Device count
     */
    int device_count() const { return device_count_; }

    /**
     * @brief Initialize streams and events for all devices
     * @param device_count Number of devices to initialize
     */
    void initialize(int device_count);

    /**
     * @brief Check if initialized
     * @return true if streams have been created
     */
    bool initialized() const { return initialized_; }

private:
    MeshStreams() = default;

    int device_count_ = 0;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    bool initialized_ = false;
};

}  // namespace cuda::distributed
