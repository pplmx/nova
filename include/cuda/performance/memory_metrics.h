#pragma once

/**
 * @file memory_metrics.h
 * @brief Memory usage query interface
 */

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda/device/error.h"

namespace cuda::memory {

struct MemoryMetrics {
    size_t used_bytes = 0;
    size_t available_bytes = 0;
    size_t total_bytes = 0;
    double utilization_percent = 0.0;
};

inline size_t used() {
    size_t available = 0;
    size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&available, &total));
    return total - available;
}

inline size_t available() {
    size_t available = 0;
    size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&available, &total));
    return available;
}

inline size_t total() {
    size_t available = 0;
    size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&available, &total));
    return total;
}

inline MemoryMetrics get_metrics() {
    MemoryMetrics metrics;
    CUDA_CHECK(cudaMemGetInfo(&metrics.available_bytes, &metrics.total_bytes));
    metrics.used_bytes = metrics.total_bytes - metrics.available_bytes;
    metrics.utilization_percent = static_cast<double>(metrics.used_bytes) /
                                  static_cast<double>(metrics.total_bytes) * 100.0;
    return metrics;
}

}  // namespace cuda::memory
