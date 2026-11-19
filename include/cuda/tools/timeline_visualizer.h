#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>

namespace cuda::tools {

    struct KernelEvent {
        std::string name;
        cudaEvent_t start;
        cudaEvent_t end;
        std::chrono::steady_clock::time_point cpu_start;
        std::chrono::steady_clock::time_point cpu_end;
        float gpu_duration_ms = 0.0f;
        size_t bytes_transferred = 0;
        float bandwidth_gbps = 0.0f;
    };

    class TimelineVisualizer {
    public:
        static TimelineVisualizer& instance();

        void begin_event(const std::string& name);
        void end_event(const std::string& name);

        void record_kernel(const std::string& name, float duration_ms);
        void record_memory_op(const std::string& name, size_t bytes, float duration_ms);

        void export_chrome_trace(const std::string& filepath);
        void export_json(const std::string& filepath);

        void clear();
        void enable();

        int get_event_count() const { return static_cast<int>(events_.size()); }

        struct TraceEvent {
            std::string name;
            std::string cat;
            std::string ph;
            int pid = 0;
            int tid = 0;
            long long ts;
            long long dur;
        };

        std::vector<TraceEvent> get_trace_events() const;

    private:
        TimelineVisualizer() = default;

        std::vector<KernelEvent> events_;
        std::unordered_map<std::string, cudaEvent_t> start_events_;
        std::unordered_map<std::string, cudaEvent_t> end_events_;
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> cpu_starts_;
        bool enabled_ = true;
        long long base_timestamp_ = 0;
    };

    class BandwidthAnalyzer {
    public:
        static BandwidthAnalyzer& instance();

        void record_operation(const std::string& name, size_t bytes, float duration_ms);

        struct BandwidthStats {
            std::string name;
            size_t total_bytes = 0;
            float total_duration_ms = 0;
            float bandwidth_gbps = 0;
            int operation_count = 0;
        };

        std::vector<BandwidthStats> get_stats() const;
        void export_report(const std::string& filepath);

        void clear();

        float get_device_bandwidth_gbps(int device_id = 0);
        float get_theoretical_bandwidth_gbps(int device_id = 0);
        float get_utilization_percentage(int device_id = 0);

    private:
        BandwidthAnalyzer() = default;

        std::vector<BandwidthStats> stats_;
    };

}  // namespace cuda::tools
