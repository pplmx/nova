#include "cuda/tools/bank_conflict_analyzer.h"
#include "cuda/tools/timeline_visualizer.h"
#include "cuda/performance/device_info.h"
#include "cuda/device/error.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace cuda::tools {

BankConflictResult analyze_bank_conflicts(
    const void* shared_mem_ptr,
    size_t data_size,
    const BankConflictConfig& config
) {
    BankConflictResult result;

    int num_banks = 32;
    int bank_width = 4;

    int elements_per_bank = static_cast<int>(data_size / bank_width) / num_banks;
    int stride = config.num_threads > 0 ? config.num_threads : config.block_size;

    if (stride > 1 && elements_per_bank > 0) {
        result.potential_conflicts = (elements_per_bank / stride) * stride;
    }

    result.suggested_padding = bank_width;

    if (config.check_padding) {
        int optimal_stride = 1;
        for (int s = 1; s <= 16; ++s) {
            if (elements_per_bank % s != 0) {
                optimal_stride = s;
                break;
            }
        }
        if (optimal_stride > 1) {
            result.suggested_padding = optimal_stride * bank_width;
        }
    }

    std::ostringstream oss;
    oss << "Bank conflict analysis:\n";
    oss << "  Data size: " << data_size << " bytes\n";
    oss << "  Stride: " << stride << "\n";
    oss << "  Potential conflicts: " << result.potential_conflicts << "\n";
    oss << "  Suggested padding: " << result.suggested_padding << " bytes\n";
    result.analysis = oss.str();

    return result;
}

int detect_bank_conflicts(
    const void* shared_mem_ptr,
    int num_elements,
    int stride
) {
    int num_banks = 32;
    int bank_width = 4;

    int conflicts = 0;
    for (int i = 0; i < num_elements; i += stride) {
        int bank_idx = (i * bank_width) % (num_banks * bank_width) / bank_width;
        for (int j = i + 1; j < num_elements && j < i + stride; ++j) {
            int other_bank = (j * bank_width) % (num_banks * bank_width) / bank_width;
            if (bank_idx == other_bank) {
                conflicts++;
            }
        }
    }

    return conflicts;
}

SharedMemoryAnalyzer& SharedMemoryAnalyzer::instance() {
    static SharedMemoryAnalyzer analyzer;
    return analyzer;
}

void SharedMemoryAnalyzer::set_config(const BankConflictConfig& config) {
    config_ = config;
}

BankConflictConfig SharedMemoryAnalyzer::get_config() const {
    return config_;
}

BankConflictResult SharedMemoryAnalyzer::analyze(const void* ptr, size_t size) {
    return analyze_bank_conflicts(ptr, size, config_);
}

int SharedMemoryAnalyzer::suggest_padding(int data_type_size, int num_elements) {
    int num_banks = 32;
    int stride = num_banks;
    int padding = 0;

    while ((num_elements * data_type_size) % (num_banks * stride) == 0 && stride < 1024) {
        stride *= 2;
    }

    if (stride > num_banks) {
        padding = stride - num_banks;
    }

    return padding * data_type_size;
}

void SharedMemoryAnalyzer::enable_padding_hints(bool enable) {
    padding_hints_enabled_ = enable;
}

TimelineVisualizer& TimelineVisualizer::instance() {
    static TimelineVisualizer visualizer;
    return visualizer;
}

void TimelineVisualizer::begin_event(const std::string& name) {
    if (!enabled_) return;

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    start_events_[name] = start;
    end_events_[name] = end;
    cpu_starts_[name] = std::chrono::steady_clock::now();

    if (base_timestamp_ == 0) {
        base_timestamp_ = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }
}

void TimelineVisualizer::end_event(const std::string& name) {
    if (!enabled_) return;

    auto it = start_events_.find(name);
    if (it == start_events_.end()) return;

    CUDA_CHECK(cudaEventRecord(end_events_[name]));
    CUDA_CHECK(cudaEventSynchronize(end_events_[name]));

    float duration;
    CUDA_CHECK(cudaEventElapsedTime(&duration, it->second, end_events_[name]));

    KernelEvent event;
    event.name = name;
    event.start = it->second;
    event.end = end_events_[name];
    event.cpu_start = cpu_starts_[name];
    event.cpu_end = std::chrono::steady_clock::now();
    event.gpu_duration_ms = duration;

    events_.push_back(event);

    start_events_.erase(it);
    end_events_.erase(name);
    cpu_starts_.erase(name);
}

void TimelineVisualizer::record_kernel(const std::string& name, float duration_ms) {
    if (!enabled_) return;

    KernelEvent event;
    event.name = name;
    event.gpu_duration_ms = duration_ms;
    event.cpu_start = std::chrono::steady_clock::now();
    events_.push_back(event);
}

void TimelineVisualizer::record_memory_op(const std::string& name, size_t bytes, float duration_ms) {
    if (!enabled_) return;

    KernelEvent event;
    event.name = name;
    event.bytes_transferred = bytes;
    event.gpu_duration_ms = duration_ms;
    event.bandwidth_gbps = bytes > 0 && duration_ms > 0
        ? (bytes / (1e9)) / (duration_ms / 1e3)
        : 0.0f;
    event.cpu_start = std::chrono::steady_clock::now();
    events_.push_back(event);
}

std::vector<TimelineVisualizer::TraceEvent> TimelineVisualizer::get_trace_events() const {
    std::vector<TraceEvent> traces;

    long long base_ts = base_timestamp_;
    if (base_ts == 0) {
        base_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    for (const auto& event : events_) {
        TraceEvent trace;
        trace.name = event.name;
        trace.cat = event.bytes_transferred > 0 ? "memory" : "kernel";
        trace.ph = "X";
        trace.pid = 1;
        trace.tid = 0;

        long long start_us = std::chrono::duration_cast<std::chrono::microseconds>(
            event.cpu_start.time_since_epoch()
        ).count();
        trace.ts = start_us - base_ts;
        trace.dur = static_cast<long long>(event.gpu_duration_ms * 1000);

        traces.push_back(trace);
    }

    return traces;
}

void TimelineVisualizer::export_chrome_trace(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return;

    file << "[\n";

    auto traces = get_trace_events();
    for (size_t i = 0; i < traces.size(); ++i) {
        const auto& t = traces[i];
        file << "  {\n";
        file << "    \"name\": \"" << t.name << "\",\n";
        file << "    \"cat\": \"" << t.cat << "\",\n";
        file << "    \"ph\": \"" << t.ph << "\",\n";
        file << "    \"pid\": " << t.pid << ",\n";
        file << "    \"tid\": " << t.tid << ",\n";
        file << "    \"ts\": " << t.ts << ",\n";
        file << "    \"dur\": " << t.dur << "\n";
        file << "  }";
        if (i < traces.size() - 1) file << ",";
        file << "\n";
    }

    file << "]\n";
}

void TimelineVisualizer::export_json(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return;

    file << "{\n";
    file << "  \"events\": [\n";

    auto traces = get_trace_events();
    for (size_t i = 0; i < traces.size(); ++i) {
        const auto& t = traces[i];
        file << "    {\n";
        file << "      \"name\": \"" << t.name << "\",\n";
        file << "      \"duration_ms\": " << std::fixed << std::setprecision(3)
             << (t.dur / 1000.0) << "\n";
        file << "    }";
        if (i < traces.size() - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
}

void TimelineVisualizer::clear() {
    for (auto& [name, event] : start_events_) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    for (auto& [name, event] : end_events_) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    start_events_.clear();
    end_events_.clear();
    cpu_starts_.clear();
    events_.clear();
}

void TimelineVisualizer::enable() {
    enabled_ = true;
}

BandwidthAnalyzer& BandwidthAnalyzer::instance() {
    static BandwidthAnalyzer analyzer;
    return analyzer;
}

void BandwidthAnalyzer::record_operation(const std::string& name, size_t bytes, float duration_ms) {
    BandwidthStats* stats = nullptr;

    for (auto& s : stats_) {
        if (s.name == name) {
            stats = &s;
            break;
        }
    }

    if (!stats) {
        BandwidthStats new_stats;
        new_stats.name = name;
        stats_.push_back(new_stats);
        stats = &stats_.back();
    }

    stats->total_bytes += bytes;
    stats->total_duration_ms += duration_ms;
    stats->operation_count++;

    if (stats->total_duration_ms > 0) {
        stats->bandwidth_gbps = (stats->total_bytes / (1e9)) / (stats->total_duration_ms / 1e3);
    }
}

std::vector<BandwidthAnalyzer::BandwidthStats> BandwidthAnalyzer::get_stats() const {
    return stats_;
}

void BandwidthAnalyzer::export_report(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return;

    file << "# Memory Bandwidth Report\n\n";
    file << "| Operation | Bytes | Duration (ms) | Bandwidth (GB/s) | Count |\n";
    file << "|-----------|-------|--------------|------------------|-------|\n";

    for (const auto& s : stats_) {
        file << "| " << s.name << " | " << s.total_bytes
             << " | " << std::fixed << std::setprecision(3) << s.total_duration_ms
             << " | " << s.bandwidth_gbps
             << " | " << s.operation_count << " |\n";
    }
}

void BandwidthAnalyzer::clear() {
    stats_.clear();
}

float BandwidthAnalyzer::get_device_bandwidth_gbps(int device_id) {
    return cuda::performance::get_memory_bandwidth_gbps(device_id);
}

float BandwidthAnalyzer::get_theoretical_bandwidth_gbps(int device_id) {
    return get_device_bandwidth_gbps(device_id);
}

float BandwidthAnalyzer::get_utilization_percentage(int device_id) {
    float theoretical = get_theoretical_bandwidth_gbps(device_id);
    if (theoretical <= 0) return 0;

    float max_observed = 0;
    for (const auto& s : stats_) {
        max_observed = std::max(max_observed, s.bandwidth_gbps);
    }

    return (max_observed / theoretical) * 100.0f;
}

}  // namespace cuda::tools
