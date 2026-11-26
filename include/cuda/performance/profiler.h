#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <cuda_runtime.h>

namespace cuda::performance {

    class Profiler;

    class ScopedTimer {
    public:
        ScopedTimer(const char* name, Profiler& profiler);
        ~ScopedTimer();

        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;

    private:
        const char* name_;
        Profiler& profiler_;
        cudaEvent_t start_ = nullptr, stop_ = nullptr;
        bool valid_;
    };

    class Profiler {
    public:
        static Profiler& instance();

        void enable() { enabled_ = true; }
        void disable() { enabled_ = false; }
        bool is_enabled() const { return enabled_; }

        void start_timer(const char* name);
        void stop_timer(const char* name);

        struct KernelMetrics {
            std::string name;
            float time_ms;
            size_t bytes_transferred;
            float bandwidth_gbps;
        };

        struct CollectiveMetrics {
            std::string name;
            std::string op_type;
            float time_ms;
            int num_ranks;
        };

        void record_kernel(const char* name, float time_ms);
        void record_memory_op(const char* name, size_t bytes, float time_ms);
        void record_collective(const char* name, const char* op_type, float time_ms, int num_ranks = 1);

        std::vector<KernelMetrics> get_kernel_metrics() const;
        std::vector<CollectiveMetrics> get_collective_metrics() const;
        void export_json(const std::string& filepath) const;
        void reset();

        float get_total_kernel_time() const;
        float get_total_memory_bandwidth() const;
        float get_total_collective_time() const;

    private:
        Profiler() = default;

        bool enabled_ = false;
        mutable std::mutex mutex_;
        std::vector<KernelMetrics> kernel_metrics_;
        std::vector<CollectiveMetrics> collective_metrics_;

        // The two handles start null so a partly-constructed timer is recognisable: start_timer()
        // creates them one at a time and either create can fail.
        struct ActiveTimer {
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            std::chrono::steady_clock::time_point cpu_start;
        };
        std::unordered_map<std::string, ActiveTimer> active_timers_;
    };

    class MemoryBandwidthTracker {
    public:
        MemoryBandwidthTracker() = default;

        void record_copy(size_t bytes, float time_ms) {
            bandwidth_gbps_ += (bytes / (1e9)) / (time_ms / 1e3);
            num_copies_++;
        }

        float get_average_bandwidth() const { return num_copies_ > 0 ? bandwidth_gbps_ / num_copies_ : 0.0f; }

        void reset() {
            bandwidth_gbps_ = 0.0f;
            num_copies_ = 0;
        }

    private:
        float bandwidth_gbps_ = 0.0f;
        int num_copies_ = 0;
    };

#ifdef NOVA_PROFILING_ENABLED
#define PROFILE_SCOPED(name) cuda::performance::ScopedTimer scoped_timer_##__LINE__(name, cuda::performance::Profiler::instance())
#else
#define PROFILE_SCOPED(name) \
    while (false)            \
    (void)0
#endif

}  // namespace cuda::performance
