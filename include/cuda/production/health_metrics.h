#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace cuda::production {

struct HealthMetrics {
    uint32_t device_id;
    float utilization_percent;
    float memory_used_mb;
    float memory_total_mb;
    uint32_t error_count_24h;
    float temperature_celsius;
    float power_usage_watts;
    uint64_t timestamp_ns;
};

struct MemorySnapshot {
    size_t allocated_bytes;
    size_t reserved_bytes;
    size_t peak_allocated_bytes;
    size_t pool_size_bytes;
    float fragmentation_percent;
};

class HealthMonitor {
public:
    HealthMonitor() = default;

    [[nodiscard]] HealthMetrics get_health_snapshot();
    [[nodiscard]] MemorySnapshot get_memory_snapshot();

    [[nodiscard]] std::string to_json() const;
    [[nodiscard]] std::string to_csv() const;

    void record_error();
    [[nodiscard]] uint32_t error_count_24h() const;

    void reset_error_count();

private:
    uint32_t error_count_24h_{0};
    uint64_t last_reset_ns_{0};
};

}  // namespace cuda::production
