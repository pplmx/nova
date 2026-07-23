#include "cuda/observability/bandwidth_tracker.h"

#include <cuda_runtime.h>
#include "cuda/device/error.h"

namespace cuda::observability {

BandwidthResult BandwidthTracker::measure_transfer(MemoryTransferType type,
                                                   uint64_t size_bytes,
                                                   cudaStream_t stream) {
    BandwidthResult result{};
    result.type = type;
    result.bytes_transferred = size_bytes;

    void* d_ptr = nullptr;
    void* h_ptr = nullptr;

    if (type == MemoryTransferType::DeviceToDevice) {
        if (cudaMalloc(&d_ptr, size_bytes) != cudaSuccess) {
            return result;
        }
    } else {
        if (cudaMallocHost(&h_ptr, size_bytes) != cudaSuccess) {
            return result;
        }
        if (type == MemoryTransferType::HostToDevice) {
            if (cudaMalloc(&d_ptr, size_bytes) != cudaSuccess) {
                CUDA_CHECK(cudaFreeHost(h_ptr));
                return result;
            }
        }
    }

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start, stream));

    if (type == MemoryTransferType::HostToDevice) {
        CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, size_bytes, cudaMemcpyHostToDevice, stream));
    } else if (type == MemoryTransferType::DeviceToHost) {
        CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, size_bytes, cudaMemcpyDeviceToHost, stream));
    } else {
        void* d_ptr2 = nullptr;
        if (cudaMalloc(&d_ptr2, size_bytes) == cudaSuccess) {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr2, d_ptr, size_bytes, cudaMemcpyDeviceToDevice, stream));
            cudaFree(d_ptr2);
        }
    }

    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));

    result.elapsed_ms = elapsed_ms;
    result.bandwidth_gbps = (size_bytes * 1000.0) / (elapsed_ms * 1e9);

    total_bytes_transferred_ += size_bytes;
    total_time_ns_ += static_cast<uint64_t>(elapsed_ms * 1e6);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_ptr));
    if (h_ptr) {
        CUDA_CHECK(cudaFreeHost(h_ptr));
    }

    return result;
}

BandwidthResult BandwidthTracker::measure_host_to_device(uint64_t size_bytes, cudaStream_t stream) {
    return measure_transfer(MemoryTransferType::HostToDevice, size_bytes, stream);
}

BandwidthResult BandwidthTracker::measure_device_to_host(uint64_t size_bytes, cudaStream_t stream) {
    return measure_transfer(MemoryTransferType::DeviceToHost, size_bytes, stream);
}

BandwidthResult BandwidthTracker::measure_device_to_device(uint64_t size_bytes, cudaStream_t stream) {
    return measure_transfer(MemoryTransferType::DeviceToDevice, size_bytes, stream);
}

DeviceMemoryBandwidth DeviceMemoryBandwidth::query(int device) {
    DeviceMemoryBandwidth result{};
    result.h2d_gbps = 0;
    result.d2h_gbps = 0;
    result.d2d_gbps = 0;

    int old_device;
    CUDA_CHECK(cudaGetDevice(&old_device));
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        int memory_clock_khz = 0;
        int memory_bus_width = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device));

        double memory_clock_ghz = memory_clock_khz / 1e6;
        double bus_width_bytes = memory_bus_width / 8.0;

        double peak_bandwidth = memory_clock_ghz * bus_width_bytes;

        result.h2d_gbps = peak_bandwidth * 0.9;
        result.d2h_gbps = peak_bandwidth * 0.9;
        result.d2d_gbps = peak_bandwidth * 0.95;
    }

    CUDA_CHECK(cudaSetDevice(old_device));
    return result;
}

double DeviceMemoryBandwidth::theoretical_bandwidth_gbps() const {
    return d2d_gbps;
}

}  // namespace cuda::observability
