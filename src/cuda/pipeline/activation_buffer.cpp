/**
 * @file activation_buffer.cpp
 * @brief Activation buffer implementation
 */

#include "cuda/pipeline/activation_buffer.h"

#include "cuda/device/error.h"

namespace cuda::pipeline {

    ActivationBuffer::ActivationBuffer(int device, size_t capacity)
        : device_(device),
          capacity_(capacity),
          ping_is_active_(true) {
        // Selecting the device is a side effect on the CALLING thread, so restore
        // whatever was current before: a pipeline buffer must not silently migrate
        // the caller's subsequent kernels to another device.
        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        CUDA_CHECK(cudaSetDevice(device_));

        // The two allocations and the device restore form one step: if the second
        // cudaMalloc fails, the constructor throws and the destructor never runs,
        // so the first buffer would leak. Allocate through a guard that frees on
        // the way out.
        const cudaError_t ping_err = cudaMalloc(&ping_, capacity_);
        if (ping_err != cudaSuccess) {
            if (had_previous) {
                cudaSetDevice(previous_device);
            }
            CUDA_CHECK(ping_err);
        }

        const cudaError_t pong_err = cudaMalloc(&pong_, capacity_);
        if (pong_err != cudaSuccess) {
            cudaFree(ping_);
            ping_ = nullptr;
            if (had_previous) {
                cudaSetDevice(previous_device);
            }
            CUDA_CHECK(pong_err);
        }

        if (had_previous) {
            cudaSetDevice(previous_device);
        }
    }

    ActivationBuffer::~ActivationBuffer() {
        // Free on the owning device (cudaFree is device-bound), then put the
        // caller's device back.
        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        cudaSetDevice(device_);
        cudaFree(ping_);
        cudaFree(pong_);
        if (had_previous) {
            cudaSetDevice(previous_device);
        }
    }

    void* ActivationBuffer::active_buffer() const {
        return ping_is_active_ ? ping_ : pong_;
    }

    void* ActivationBuffer::inactive_buffer() const {
        return ping_is_active_ ? pong_ : ping_;
    }

    void ActivationBuffer::swap() {
        ping_is_active_ = !ping_is_active_;
    }

    bool ActivationBuffer::is_ping_active() const {
        return ping_is_active_;
    }

    size_t ActivationBuffer::capacity() const {
        return capacity_;
    }

    int ActivationBuffer::device() const {
        return device_;
    }

}  // namespace cuda::pipeline
