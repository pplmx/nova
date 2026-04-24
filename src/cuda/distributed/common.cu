/**
 * @file common.cu
 * @brief MeshStreams implementation
 */

#include "cuda/distributed/common.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace cuda::distributed {

MeshStreams& MeshStreams::instance() {
    static MeshStreams instance;
    return instance;
}

MeshStreams::~MeshStreams() {
    // Destroy events first
    for (auto event : events_) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    events_.clear();

    // Then destroy streams
    for (auto stream : streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    streams_.clear();

    initialized_ = false;
}

void MeshStreams::initialize(int device_count) {
    if (initialized_) {
        return;
    }

    if (device_count <= 0) {
        throw std::invalid_argument("Device count must be positive");
    }

    device_count_ = device_count;
    streams_.resize(device_count);
    events_.resize(device_count);

    for (int i = 0; i < device_count; ++i) {
        // Set device before creating stream/event
        CUDA_CHECK(cudaSetDevice(i));

        // Create non-blocking stream for concurrency
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));

        // Create event with default flags
        CUDA_CHECK(cudaEventCreate(&events_[i]));
    }

    initialized_ = true;
}

cudaStream_t MeshStreams::get_stream(int device) {
    if (!initialized_) {
        throw std::runtime_error("MeshStreams not initialized");
    }
    if (device < 0 || device >= device_count_) {
        throw std::out_of_range("Device index out of range");
    }
    return streams_[device];
}

cudaEvent_t MeshStreams::get_event(int device) {
    if (!initialized_) {
        throw std::runtime_error("MeshStreams not initialized");
    }
    if (device < 0 || device >= device_count_) {
        throw std::out_of_range("Device index out of range");
    }
    return events_[device];
}

void MeshStreams::synchronize_all() {
    if (!initialized_) {
        return;
    }

    for (int i = 0; i < device_count_; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
    }
}

void MeshStreams::wait_event(int dst_device, int src_device, cudaEvent_t event) {
    if (!initialized_) {
        throw std::runtime_error("MeshStreams not initialized");
    }

    if (dst_device < 0 || dst_device >= device_count_ ||
        src_device < 0 || src_device >= device_count_) {
        throw std::out_of_range("Device index out of range");
    }

    // Set destination device and make it wait on source device's event
    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaStreamWaitEvent(streams_[dst_device], event, 0));
}

}  // namespace cuda::distributed
