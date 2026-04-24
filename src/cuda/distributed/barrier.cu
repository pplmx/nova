/**
 * @file barrier.cu
 * @brief Barrier implementation using event-based synchronization
 */

#include "cuda/distributed/barrier.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <chrono>
#include <thread>

namespace cuda::distributed {

MeshBarrier::MeshBarrier()
    : device_count_(0),
      initialized_(false),
      barriering_(false) {

    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    device_count_ = mesh.device_count();

    if (device_count_ <= 1) {
        // Single GPU - no initialization needed
        initialized_ = false;
        return;
    }

    events_.resize(device_count_);
    streams_.resize(device_count_);

    for (int i = 0; i < device_count_; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming));
    }

    initialized_ = true;
}

MeshBarrier::~MeshBarrier() {
    if (!initialized_) return;

    for (int i = 0; i < device_count_; ++i) {
        if (streams_[i]) {
            cudaStreamDestroy(streams_[i]);
        }
        if (events_[i]) {
            cudaEventDestroy(events_[i]);
        }
    }
}

void MeshBarrier::synchronize() {
    auto& mesh = cuda::mesh::DeviceMesh::instance();

    // Single-GPU fallback
    if (device_count_ <= 1 || !initialized_) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // Signal that we're barriering
    barriering_.store(true, std::memory_order_release);

    // Each GPU records an event on its stream
    for (int i = 0; i < device_count_; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventRecord(events_[i], streams_[i]));
    }

    // Host polls for all events to complete
    // Once all are complete, all GPUs can proceed
    std::vector<bool> completed(device_count_, false);
    int completed_count = 0;

    while (completed_count < device_count_) {
        for (int i = 0; i < device_count_; ++i) {
            if (!completed[i]) {
                CUDA_CHECK(cudaSetDevice(i));
                cudaError_t err = cudaEventQuery(events_[i]);
                if (err == cudaSuccess) {
                    completed[i] = true;
                    ++completed_count;
                } else if (err != cudaErrorNotReady) {
                    // Unexpected error
                    throw cuda::device::CudaException(err, __FILE__, __LINE__);
                }
            }
        }

        // Yield to avoid busy-waiting
        if (completed_count < device_count_) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // All events completed - signal that we're done
    barriering_.store(false, std::memory_order_release);
}

void MeshBarrier::synchronize_async(cudaStream_t /*stream*/) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();

    // Single-GPU fallback
    if (device_count_ <= 1 || !initialized_) {
        return;
    }

    // Each GPU records an event on its stream
    for (int i = 0; i < device_count_; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventRecord(events_[i], streams_[i]));
    }

    barriering_.store(true, std::memory_order_release);
}

void MeshBarrier::synchronize_devices(const std::vector<int>& devices) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();

    int n = static_cast<int>(devices.size());

    if (n <= 1) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    barriering_.store(true, std::memory_order_release);

    // Record event on each specified device
    std::vector<cudaEvent_t> local_events(n);
    for (int i = 0; i < n; ++i) {
        int device = devices[i];
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaEventCreateWithFlags(&local_events[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(local_events[i], streams_[device]));
    }

    // Poll for completion
    std::vector<bool> completed(n, false);
    int completed_count = 0;

    while (completed_count < n) {
        for (int i = 0; i < n; ++i) {
            if (!completed[i]) {
                int device = devices[i];
                CUDA_CHECK(cudaSetDevice(device));
                cudaError_t err = cudaEventQuery(local_events[i]);
                if (err == cudaSuccess) {
                    completed[i] = true;
                    ++completed_count;
                } else if (err != cudaErrorNotReady) {
                    throw cuda::device::CudaException(err, __FILE__, __LINE__);
                }
            }
        }

        if (completed_count < n) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Cleanup local events
    for (int i = 0; i < n; ++i) {
        int device = devices[i];
        CUDA_CHECK(cudaSetDevice(device));
        cudaEventDestroy(local_events[i]);
    }

    barriering_.store(false, std::memory_order_release);
}

}  // namespace cuda::distributed
