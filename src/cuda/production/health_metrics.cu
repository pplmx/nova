#include "cuda/production/health_metrics.h"

#include <chrono>
#include <sstream>

namespace cuda::production {

HealthMetrics HealthMonitor::get_health_snapshot() {
    HealthMetrics metrics{};
    metrics.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();

    int device_id = 0;
    cudaGetDevice(&device_id);
    metrics.device_id = static_cast<uint32_t>(device_id);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    metrics.memory_used_mb = static_cast<float>((total_mem - free_mem) / (1024 * 1024));
    metrics.memory_total_mb = static_cast<float>(total_mem / (1024 * 1024));
    metrics.memory_used_mb = metrics.memory_total_mb - metrics.memory_used_mb;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, metrics.device_id) == cudaSuccess) {
        metrics.utilization_percent = 0.0f;
    }

    metrics.error_count_24h = error_count_24h_;

    metrics.temperature_celsius = 0.0f;

    metrics.power_usage_watts = 0.0f;

    return metrics;
}

MemorySnapshot HealthMonitor::get_memory_snapshot() {
    MemorySnapshot snapshot{};

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    snapshot.reserved_bytes = total_mem;
    snapshot.allocated_bytes = total_mem - free_mem;

    return snapshot;
}

std::string HealthMonitor::to_json() const {
    auto metrics = const_cast<HealthMonitor*>(this)->get_health_snapshot();
    auto mem = const_cast<HealthMonitor*>(this)->get_memory_snapshot();

    std::ostringstream oss;
    oss << "{";
    oss << "\"device_id\":" << metrics.device_id << ",";
    oss << "\"utilization_percent\":" << metrics.utilization_percent << ",";
    oss << "\"memory_used_mb\":" << metrics.memory_used_mb << ",";
    oss << "\"memory_total_mb\":" << metrics.memory_total_mb << ",";
    oss << "\"memory_allocated_bytes\":" << mem.allocated_bytes << ",";
    oss << "\"error_count_24h\":" << metrics.error_count_24h << ",";
    oss << "\"temperature_celsius\":" << metrics.temperature_celsius << ",";
    oss << "\"power_usage_watts\":" << metrics.power_usage_watts << ",";
    oss << "\"timestamp_ns\":" << metrics.timestamp_ns;
    oss << "}";

    return oss.str();
}

std::string HealthMonitor::to_csv() const {
    auto metrics = const_cast<HealthMonitor*>(this)->get_health_snapshot();
    auto mem = const_cast<HealthMonitor*>(this)->get_memory_snapshot();

    std::ostringstream oss;
    oss << metrics.timestamp_ns << ","
        << metrics.device_id << ","
        << metrics.utilization_percent << ","
        << metrics.memory_used_mb << ","
        << metrics.memory_total_mb << ","
        << mem.allocated_bytes << ","
        << metrics.error_count_24h << ","
        << metrics.temperature_celsius << ","
        << metrics.power_usage_watts;

    return oss.str();
}

void HealthMonitor::record_error() {
    error_count_24h_++;
}

uint32_t HealthMonitor::error_count_24h() const {
    return error_count_24h_;
}

void HealthMonitor::reset_error_count() {
    error_count_24h_ = 0;
    last_reset_ns_ = std::chrono::steady_clock::now().time_since_epoch().count();
}

}  // namespace cuda::production
