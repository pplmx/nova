#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda::observability {

enum class MemoryTransferType {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

struct BandwidthResult {
    double bandwidth_gbps;
    uint64_t bytes_transferred;
    double elapsed_ms;
    MemoryTransferType type;
};

class BandwidthTracker {
public:
    BandwidthTracker() = default;

    BandwidthResult measure_transfer(MemoryTransferType type,
                                     uint64_t size_bytes,
                                     cudaStream_t stream = 0);

    BandwidthResult measure_host_to_device(uint64_t size_bytes, cudaStream_t stream = 0);
    BandwidthResult measure_device_to_host(uint64_t size_bytes, cudaStream_t stream = 0);
    BandwidthResult measure_device_to_device(uint64_t size_bytes, cudaStream_t stream = 0);

    void reset() {
        total_bytes_transferred_ = 0;
        total_time_ns_ = 0;
    }

    uint64_t total_bytes_transferred() const { return total_bytes_transferred_; }
    uint64_t total_time_ns() const { return total_time_ns_; }

private:
    uint64_t total_bytes_transferred_ = 0;
    uint64_t total_time_ns_ = 0;
};

struct DeviceMemoryBandwidth {
    double h2d_gbps;
    double d2h_gbps;
    double d2d_gbps;

    static DeviceMemoryBandwidth query(int device = 0);

    double peak_memory_clock_gbps() const;
    double memory_bus_width_bits() const;
    double theoretical_bandwidth_gbps() const;
};

}  // namespace cuda::observability
