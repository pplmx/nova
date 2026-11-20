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
#include <mutex>
#include <vector>

namespace cuda::distributed {

    /**
     * @enum ReductionOp
     * @brief Supported reduction operations for collective primitives
     */
    enum class ReductionOp { Sum, Min, Max, Product };

    /**
     * @class MeshStreams
     * @brief Manages per-device CUDA streams and events for collective operations
     *
     * CUDA streams are device-local, so each GPU requires its own stream.
     * This class maintains a vector of streams, one per device, along with
     * pre-allocated events for synchronization.
     *
     * @note The singleton itself is Meyer's, and every accessor that reads or
     *       writes the per-device arrays takes mutex_. Without that the note here
     *       claimed a thread-safety the class did not have: initialize() resizes the
     *       vectors and then fills them device by device, so a concurrent
     *       get_stream() could observe a resized-but-unfilled vector and return a
     *       null handle, or read initialized_ before the streams existed.
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
        int device_count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return device_count_;
        }

        /**
         * @brief Initialize streams and events for all devices
         * @param device_count Number of devices to initialize
         */
        void initialize(int device_count);

        /**
         * @brief Release every stream and event and return to the uninitialised state
         *
         * initialize() is a one-shot: a second call returns early, so a process that
         * has once built the mesh can never exercise the cold path again. That made the
         * class's claimed thread-safety untestable -- the window that matters is
         * initialize() filling the arrays, and it can only be entered once. reset()
         * exists so tests (and the singleton-reset registry) can return to that state.
         *
         * Not for concurrent use with readers: it destroys the handles they may hold.
         */
        void reset();

        /**
         * @brief Check if initialized
         * @return true if streams have been created
         */
        bool initialized() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return initialized_;
        }

    private:
        MeshStreams() = default;

        mutable std::mutex mutex_;
        int device_count_ = 0;
        std::vector<cudaStream_t> streams_;
        std::vector<cudaEvent_t> events_;
        bool initialized_ = false;
    };

}  // namespace cuda::distributed
