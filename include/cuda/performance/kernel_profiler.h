#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>
#include <map>
#include <mutex>

namespace cuda::performance {

    struct KernelRecord;

    class KernelProfiler {
    public:
        using ProfilerCallback = std::function<void(const std::string&, uint64_t)>;

        static KernelProfiler& instance();

        void set_callback(ProfilerCallback callback);

        void record_start(const std::string& kernel_name, cudaStream_t stream = 0);
        void record_end(const std::string& kernel_name, cudaStream_t stream = 0);

        [[nodiscard]] uint64_t get_kernel_latency_ns(const std::string& kernel_name) const;
        [[nodiscard]] float estimate_occupancy(int block_size, int max_blocks_per_sm) const;

        void enable();
        void disable();
        [[nodiscard]] bool is_enabled() const;

        void reset();

    private:
        KernelProfiler() = default;
        ~KernelProfiler() = default;

        KernelProfiler(const KernelProfiler&) = delete;
        KernelProfiler& operator=(const KernelProfiler&) = delete;

        KernelRecord* get_or_create_record(const std::string& kernel_name);

        ProfilerCallback callback_;
        bool enabled_ {true};
        mutable std::mutex mutex_;
        std::map<std::string, std::unique_ptr<KernelRecord>> records_;
    };

    struct KernelRecord {
        cudaEvent_t start_event {nullptr};
        cudaEvent_t stop_event {nullptr};
        uint64_t latency_ns {0};
        bool completed {false};
    };

    class ScopedKernelProfile {
    public:
        explicit ScopedKernelProfile(const std::string& kernel_name, cudaStream_t stream = 0);
        /// Records the end of the measured region.
        ///
        /// Declared noexcept because a destructor is; record_end() calls CUDA_CHECK three times and
        /// then invokes an arbitrary user callback, all of which can throw. An exception escaping a
        /// destructor calls std::terminate, so the failure is dropped here instead of being turned
        /// into a process abort. Use KernelProfiler::record_end() directly when the error matters.
        ~ScopedKernelProfile() noexcept;

        ScopedKernelProfile(const ScopedKernelProfile&) = delete;
        ScopedKernelProfile& operator=(const ScopedKernelProfile&) = delete;
        ScopedKernelProfile(ScopedKernelProfile&&) noexcept = default;
        ScopedKernelProfile& operator=(ScopedKernelProfile&&) noexcept = default;

    private:
        std::string kernel_name_;
        cudaStream_t stream_ {0};
    };

    class OccupancyCalculator {
    public:
        [[nodiscard]] static float calculate_theoretical_occupancy(int threads_per_block, int registers_per_thread, int shared_mem_bytes, int device_id = 0);

        [[nodiscard]] static int recommended_block_size(int threads_per_block, int registers_per_thread, int shared_mem_bytes, int device_id = 0);
    };

}  // namespace cuda::performance
