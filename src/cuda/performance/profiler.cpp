#include "cuda/performance/profiler.h"

#include "cuda/device/error.h"

#include <fstream>
#include <mutex>
#include <iomanip>
#include <algorithm>

namespace cuda::performance {

ScopedTimer::ScopedTimer(const char* name, Profiler& profiler)
    : name_(name), profiler_(profiler), valid_(false) {

    if (!profiler_.is_enabled()) return;

    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUDA_CHECK(cudaEventRecord(start_, 0));
    valid_ = true;
}

ScopedTimer::~ScopedTimer() {
    if (!valid_) return;

    // Not using CUDA_CHECK here: throwing in a destructor is undefined behavior
    // (C++11 destructors default to noexcept). Event cleanup failures are
    // non-critical — the process is likely already in error state if these
    // fail.
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);

    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start_, stop_);

    profiler_.record_kernel(name_, time_ms);

    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::start_timer(const char* name) {
    if (!enabled_) return;

    std::lock_guard lock(mutex_);

    ActiveTimer timer;
    CUDA_CHECK(cudaEventCreate(&timer.start));
    CUDA_CHECK(cudaEventCreate(&timer.stop));
    CUDA_CHECK(cudaEventRecord(timer.start, 0));
    timer.cpu_start = std::chrono::steady_clock::now();

    active_timers_[name] = std::move(timer);
}

void Profiler::stop_timer(const char* name) {
    if (!enabled_) return;

    std::lock_guard lock(mutex_);

    auto it = active_timers_.find(name);
    if (it == active_timers_.end()) return;

    auto& timer = it->second;
    CUDA_CHECK(cudaEventRecord(timer.stop, 0));
    CUDA_CHECK(cudaEventSynchronize(timer.stop));

    float time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, timer.start, timer.stop));

    record_kernel(name, time_ms);

    CUDA_CHECK(cudaEventDestroy(timer.start));
    CUDA_CHECK(cudaEventDestroy(timer.stop));
    active_timers_.erase(it);
}

void Profiler::record_kernel(const char* name, float time_ms) {
    std::lock_guard lock(mutex_);
    kernel_metrics_.push_back({std::string(name), time_ms, 0, 0.0f});
}

void Profiler::record_memory_op(const char* name, size_t bytes, float time_ms) {
    std::lock_guard lock(mutex_);

    float bandwidth_gbps = 0.0f;
    if (time_ms > 0.0f) {
        bandwidth_gbps = (bytes / (1e9)) / (time_ms / 1e3);
    }

    kernel_metrics_.push_back({
        std::string(name),
        time_ms,
        bytes,
        bandwidth_gbps
    });
}

void Profiler::record_collective(const char* name, const char* op_type,
                                  float time_ms, int num_ranks) {
    std::lock_guard lock(mutex_);
    collective_metrics_.push_back({
        std::string(name),
        std::string(op_type),
        time_ms,
        num_ranks
    });
}

std::vector<Profiler::KernelMetrics> Profiler::get_kernel_metrics() const {
    std::lock_guard lock(mutex_);
    return kernel_metrics_;
}

std::vector<Profiler::CollectiveMetrics> Profiler::get_collective_metrics() const {
    std::lock_guard lock(mutex_);
    return collective_metrics_;
}

float Profiler::get_total_kernel_time() const {
    std::lock_guard lock(mutex_);
    float total = 0.0f;
    for (const auto& m : kernel_metrics_) {
        total += m.time_ms;
    }
    return total;
}

float Profiler::get_total_memory_bandwidth() const {
    std::lock_guard lock(mutex_);
    float total = 0.0f;
    for (const auto& m : kernel_metrics_) {
        total += m.bandwidth_gbps * (m.time_ms / 1000.0f);
    }
    return total;
}

float Profiler::get_total_collective_time() const {
    std::lock_guard lock(mutex_);
    float total = 0.0f;
    for (const auto& m : collective_metrics_) {
        total += m.time_ms;
    }
    return total;
}

void Profiler::export_json(const std::string& filepath) const {
    std::lock_guard lock(mutex_);

    float total_kernel = 0.0f;
    for (const auto& m : kernel_metrics_) {
        total_kernel += m.time_ms;
    }

    float total_collective = 0.0f;
    for (const auto& m : collective_metrics_) {
        total_collective += m.time_ms;
    }

    std::ofstream out(filepath);
    if (!out) return;

    out << "{\n";
    out << "  \"kernel_metrics\": [\n";
    for (size_t i = 0; i < kernel_metrics_.size(); ++i) {
        const auto& m = kernel_metrics_[i];
        out << "    {\n";
        out << "      \"name\": \"" << m.name << "\",\n";
        out << "      \"time_ms\": " << std::fixed << std::setprecision(3) << m.time_ms << ",\n";
        out << "      \"bytes_transferred\": " << m.bytes_transferred << ",\n";
        out << "      \"bandwidth_gbps\": " << m.bandwidth_gbps << "\n";
        out << "    }" << (i < kernel_metrics_.size() - 1 ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"collective_metrics\": [\n";
    for (size_t i = 0; i < collective_metrics_.size(); ++i) {
        const auto& m = collective_metrics_[i];
        out << "    {\n";
        out << "      \"name\": \"" << m.name << "\",\n";
        out << "      \"op_type\": \"" << m.op_type << "\",\n";
        out << "      \"time_ms\": " << m.time_ms << ",\n";
        out << "      \"num_ranks\": " << m.num_ranks << "\n";
        out << "    }" << (i < collective_metrics_.size() - 1 ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"summary\": {\n";
    out << "    \"total_kernel_time_ms\": " << total_kernel << ",\n";
    out << "    \"total_collective_time_ms\": " << total_collective << ",\n";
    out << "    \"kernel_count\": " << kernel_metrics_.size() << ",\n";
    out << "    \"collective_count\": " << collective_metrics_.size() << "\n";
    out << "  }\n";
    out << "}\n";

    out.close();
}

void Profiler::reset() {
    std::lock_guard lock(mutex_);
    kernel_metrics_.clear();
    collective_metrics_.clear();
    active_timers_.clear();
}

} // namespace cuda::performance
