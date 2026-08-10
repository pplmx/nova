#include <cuda/performance/nvblox_metrics.h>

#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <iomanip>

namespace cuda::performance {

NVBloxMetricsCollector::NVBloxMetricsCollector() : initialized_(true) {}

NVBloxMetricsCollector::~NVBloxMetricsCollector() = default;

NVBloxMetricsCollector::NVBloxMetricsCollector(NVBloxMetricsCollector&& other) noexcept
    : mutex_()
    , registered_metrics_(std::move(other.registered_metrics_))
    , metric_samples_(std::move(other.metric_samples_))
    , kernel_metrics_(std::move(other.kernel_metrics_))
    , initialized_(other.initialized_) {
    other.initialized_ = false;
}

NVBloxMetricsCollector& NVBloxMetricsCollector::operator=(NVBloxMetricsCollector&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> other_lock(other.mutex_);
        registered_metrics_ = std::move(other.registered_metrics_);
        metric_samples_ = std::move(other.metric_samples_);
        kernel_metrics_ = std::move(other.kernel_metrics_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

void NVBloxMetricsCollector::register_metric(const std::string& name, MetricType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    registered_metrics_[name] = type;
}

void NVBloxMetricsCollector::add_sample(const std::string& name, double value) {
    uint64_t timestamp = 0;
    cudaEvent_t start, stop;
    if (cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float dummy_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&dummy_time_ms, start, stop));
        timestamp = static_cast<uint64_t>(dummy_time_ms * 1e6);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    metric_samples_[name].push_back({value, timestamp});
}

void NVBloxMetricsCollector::record_kernel(const KernelMetrics& metrics) {
    std::lock_guard<std::mutex> lock(mutex_);
    kernel_metrics_.push_back(metrics);
}

std::vector<KernelMetrics> NVBloxMetricsCollector::get_kernel_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernel_metrics_;
}

std::vector<MetricSample> NVBloxMetricsCollector::get_metric_samples(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metric_samples_.find(name);
    if (it != metric_samples_.end()) {
        return it->second;
    }
    return {};
}

std::string NVBloxMetricsCollector::to_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"metrics\": {\n";
    bool first_metric = true;
    for (const auto& [name, samples] : metric_samples_) {
        if (!first_metric) oss << ",\n";
        first_metric = false;
        oss << "    \"" << name << "\": [";
        bool first_sample = true;
        for (const auto& sample : samples) {
            if (!first_sample) oss << ", ";
            first_sample = false;
            oss << "{\"value\": " << std::fixed << std::setprecision(6) << sample.value
                << ", \"timestamp\": " << sample.timestamp_ns << "}";
        }
        oss << "]";
    }
    oss << "\n  },\n";
    oss << "  \"kernels\": [\n";
    bool first_kernel = true;
    for (const auto& km : kernel_metrics_) {
        if (!first_kernel) oss << ",\n";
        first_kernel = false;
        oss << "    {\"name\": \"" << km.name << "\", "
            << "\"latency_ns\": " << km.latency_ns << ", "
            << "\"throughput_gflops\": " << std::fixed << std::setprecision(2) << km.throughput_gflops << ", "
            << "\"sm_occupancy\": " << km.sm_occupancy << ", "
            << "\"arithmetic_intensity\": " << km.arithmetic_intensity << ", "
            << "\"memory_bandwidth_gbs\": " << km.memory_bandwidth_gbs << "}";
    }
    oss << "\n  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string NVBloxMetricsCollector::to_csv() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "kernel_name,latency_ns,throughput_gflops,sm_occupancy,arithmetic_intensity,memory_bandwidth_gbs,timestamp_ns\n";
    for (const auto& km : kernel_metrics_) {
        oss << "\"" << km.name << "\","
            << km.latency_ns << ","
            << std::fixed << std::setprecision(2) << km.throughput_gflops << ","
            << km.sm_occupancy << ","
            << km.arithmetic_intensity << ","
            << km.memory_bandwidth_gbs << ","
            << km.timestamp_ns << "\n";
    }
    return oss.str();
}

void NVBloxMetricsCollector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    // reset() restores the collector to a fresh state: clear the registered
    // metric registry as well, otherwise registered_metric_count() keeps
    // accumulating across reset() calls (the collector is a process-wide
    // singleton, so stale registrations leak across test suites and sessions).
    registered_metrics_.clear();
    metric_samples_.clear();
    kernel_metrics_.clear();
}

size_t NVBloxMetricsCollector::registered_metric_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registered_metrics_.size();
}

size_t NVBloxMetricsCollector::total_kernel_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernel_metrics_.size();
}

NVBloxMetricsCollector& NVBloxMetricsCollector::instance() {
    static NVBloxMetricsCollector instance;
    return instance;
}

double calculate_arithmetic_intensity(uint64_t flops, size_t bytes_accessed) {
    if (bytes_accessed == 0) return 0.0;
    return static_cast<double>(flops) / static_cast<double>(bytes_accessed);
}

double calculate_throughput_gflops(uint64_t flops, uint64_t elapsed_ns) {
    if (elapsed_ns == 0) return 0.0;
    return (static_cast<double>(flops) / 1e9) / (static_cast<double>(elapsed_ns) / 1e9);
}

double calculate_memory_bandwidth_gbs(size_t bytes, uint64_t elapsed_ns) {
    if (elapsed_ns == 0) return 0.0;
    return (static_cast<double>(bytes) / 1e9) / (static_cast<double>(elapsed_ns) / 1e9);
}

}  // namespace cuda::performance
