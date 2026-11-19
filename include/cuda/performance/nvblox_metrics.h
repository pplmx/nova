#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuda::performance {

    enum class MetricType { Latency, Throughput, Occupancy, ArithmeticIntensity, MemoryBandwidth };

    struct KernelMetrics {
        std::string name;
        uint64_t latency_ns{0};
        double throughput_gflops{0.0};
        float sm_occupancy{0.0f};
        double arithmetic_intensity{0.0};
        double memory_bandwidth_gbs{0.0};
        uint64_t timestamp_ns{0};
        uint32_t block_x{0};
        uint32_t block_y{0};
        uint32_t block_z{0};
        uint32_t grid_x{0};
        uint32_t grid_y{0};
        uint32_t grid_z{0};
    };

    struct MetricSample {
        double value;
        uint64_t timestamp_ns;
    };

    class NVBloxMetricsCollector {
    public:
        NVBloxMetricsCollector();
        ~NVBloxMetricsCollector();

        NVBloxMetricsCollector(const NVBloxMetricsCollector&) = delete;
        NVBloxMetricsCollector& operator=(const NVBloxMetricsCollector&) = delete;
        NVBloxMetricsCollector(NVBloxMetricsCollector&&) noexcept;
        NVBloxMetricsCollector& operator=(NVBloxMetricsCollector&&) noexcept;

        void register_metric(const std::string& name, MetricType type);
        void add_sample(const std::string& name, double value);
        void record_kernel(const KernelMetrics& metrics);

        [[nodiscard]] std::vector<KernelMetrics> get_kernel_metrics() const;
        [[nodiscard]] std::vector<MetricSample> get_metric_samples(const std::string& name) const;
        [[nodiscard]] std::string to_json() const;
        [[nodiscard]] std::string to_csv() const;

        void reset();
        [[nodiscard]] size_t registered_metric_count() const;
        [[nodiscard]] size_t total_kernel_count() const;

        static NVBloxMetricsCollector& instance();

    private:
        mutable std::mutex mutex_;
        std::unordered_map<std::string, MetricType> registered_metrics_;
        std::unordered_map<std::string, std::vector<MetricSample>> metric_samples_;
        std::vector<KernelMetrics> kernel_metrics_;
        bool initialized_{false};
    };

    double calculate_arithmetic_intensity(uint64_t flops, size_t bytes_accessed);

    double calculate_throughput_gflops(uint64_t flops, uint64_t elapsed_ns);

    double calculate_memory_bandwidth_gbs(size_t bytes, uint64_t elapsed_ns);

}  // namespace cuda::performance
