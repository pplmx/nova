#pragma once

/**
 * @file activation_buffer.h
 * @brief Activation buffer management with ping-pong double buffering
 *
 * Provides double-buffered activation storage for pipeline parallelism
 * to overlap computation and communication.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace cuda::pipeline {

    /**
     * @class ActivationBuffer
     * @brief Double-buffered activation storage
     *
     * Manages two buffers (ping and pong) that swap roles each microbatch.
     * Allows computation to overlap with communication.
     *
     * @example
     * @code
     * ActivationBuffer buf(device, capacity);
     *
     * // During microbatch N:
     * void* compute_buffer = buf.active_buffer();  // Get ping or pong
     * void* recv_buffer = buf.inactive_buffer();   // Receive into this
     *
     * buf.swap();  // Swap for next microbatch
     * @endcode
     */
    class ActivationBuffer {
    public:
        /**
         * @brief Construct activation buffer
         * @param device CUDA device index
         * @param capacity Buffer capacity in bytes
         */
        ActivationBuffer(int device, size_t capacity);

        ~ActivationBuffer();

        // Non-copyable
        ActivationBuffer(const ActivationBuffer&) = delete;
        ActivationBuffer& operator=(const ActivationBuffer&) = delete;

        /**
         * @brief Get pointer to currently active buffer
         * @return Active buffer pointer
         */
        [[nodiscard]] void* active_buffer() const;

        /**
         * @brief Get pointer to inactive buffer
         * @return Inactive buffer pointer (for receiving)
         */
        [[nodiscard]] void* inactive_buffer() const;

        /**
         * @brief Swap ping and pong buffers
         */
        void swap();

        /**
         * @brief Check if ping is currently active
         * @return true if ping is active, false if pong
         */
        [[nodiscard]] bool is_ping_active() const;

        /**
         * @brief Get buffer capacity
         * @return Capacity in bytes
         */
        [[nodiscard]] size_t capacity() const;

        /**
         * @brief Get device index
         * @return CUDA device
         */
        [[nodiscard]] int device() const;

    private:
        int device_;
        size_t capacity_;
        void* ping_;
        void* pong_;
        bool ping_is_active_;
    };

}  // namespace cuda::pipeline
