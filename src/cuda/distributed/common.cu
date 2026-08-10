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
    // Heap-allocated singleton that is intentionally never destroyed.
    //
    // A function-local static's destructor runs at process exit, and the
    // destructor here calls cudaEventDestroy/cudaStreamDestroy. Driving libcuda
    // during __run_exit_handlers, after the per-thread driver context is already
    // being torn down, makes cuEventDestroy SEGV inside the driver — observed as
    // an "end-of-run SIGSEGV" (EXIT=139) after an otherwise fully-green suite,
    // exactly like the NcclContext singleton fixed in the same way.
    //
    // The CUDA driver reclaims all GPU resources when the process exits, so
    // orderly teardown at exit is unnecessary.
    static MeshStreams* instance = new MeshStreams();
    return *instance;
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
