#pragma once

/**
 * @file device_info.h
 * @brief Device capability queries and memory bandwidth detection
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <optional>

#include "cuda/device/error.h"

namespace cuda::performance {

struct DeviceProperties {
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    size_t global_memory_bytes = 0;
    size_t shared_memory_per_block = 0;
    int max_threads_per_block = 0;
    int multiprocessor_count = 0;
    size_t memory_bandwidth_gbps = 0;
    int device_id = -1;
    char name[256] = {0};
};

inline DeviceProperties get_device_properties(int device_id = 0) {
    DeviceProperties props;
    props.device_id = device_id;

    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

    props.compute_capability_major = device_prop.major;
    props.compute_capability_minor = device_prop.minor;
    props.global_memory_bytes = device_prop.totalGlobalMem;
    props.shared_memory_per_block = device_prop.sharedMemPerBlock;
    props.max_threads_per_block = device_prop.maxThreadsPerBlock;
    props.multiprocessor_count = device_prop.multiProcessorCount;

    int clock_rate_khz = device_prop.memoryClockRate;
    int memory_bus_width = device_prop.memoryBusWidth;
    size_t bandwidth_bps =
        static_cast<size_t>(2) * static_cast<size_t>(clock_rate_khz) * 1000 *
        static_cast<size_t>(memory_bus_width) / 8;
    props.memory_bandwidth_gbps = bandwidth_bps / (1024 * 1024 * 1024);

    for (int i = 0; i < 255 && device_prop.name[i] != '\0'; ++i) {
        props.name[i] = device_prop.name[i];
    }

    return props;
}

inline int get_optimal_block_size(int device_id = 0) {
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

    int cc = device_prop.major * 10 + device_prop.minor;

    if (cc >= 90) {
        return 1024;
    } else if (cc >= 80) {
        return 512;
    } else if (cc >= 70) {
        return 512;
    } else if (cc >= 60) {
        return 256;
    }

    return 256;
}

inline size_t get_memory_bandwidth_gbps(int device_id = 0) {
    return get_device_properties(device_id).memory_bandwidth_gbps;
}

inline size_t get_global_memory_bytes(int device_id = 0) {
    return get_device_properties(device_id).global_memory_bytes;
}

inline int get_compute_capability_major(int device_id = 0) {
    return get_device_properties(device_id).compute_capability_major;
}

inline int get_compute_capability_minor(int device_id = 0) {
    return get_device_properties(device_id).compute_capability_minor;
}

inline int get_multiprocessor_count(int device_id = 0) {
    return get_device_properties(device_id).multiprocessor_count;
}

inline int get_max_threads_per_block(int device_id = 0) {
    return get_device_properties(device_id).max_threads_per_block;
}

inline std::optional<int> get_current_device() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        return device;
    }
    return std::nullopt;
}

inline void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

inline int get_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

}  // namespace cuda::performance
