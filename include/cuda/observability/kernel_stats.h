#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <string>

namespace cuda::observability {

    struct KernelStats {
        std::string kernel_name;
        uint64_t invocations = 0;
        double total_time_us = 0.0;
        double min_time_us = std::numeric_limits<double>::max();
        double max_time_us = 0.0;
        double avg_time_us = 0.0;
        double achieved_occupancy = 0.0;
        size_t blocks_launched = 0;
        size_t threads_per_block = 0;
    };

    class KernelStatsCollector {
    public:
        KernelStatsCollector() = default;

        void record_kernel(const char* name, cudaEvent_t start, cudaEvent_t end, size_t blocks, size_t threads_per_block);

        void record_launch(const char* name, size_t blocks, size_t threads_per_block);

        void set_stream(cudaStream_t stream) { stream_ = stream; }
        cudaStream_t stream() const { return stream_; }

        const std::vector<KernelStats>& stats() const { return stats_; }
        KernelStats get_stats(const char* name) const;

        void reset() { stats_.clear(); }
        void clear() { stats_.clear(); }

    private:
        cudaStream_t stream_ = 0;
        std::vector<KernelStats> stats_;
    };

    class ScopedKernelTiming {
    public:
        ScopedKernelTiming(const char* name, KernelStatsCollector& collector, size_t blocks = 0, size_t threads_per_block = 0);

        /// Records the timing on destruction.
        ///
        /// Declared noexcept because a destructor is. The body avoids CUDA_CHECK on its own event
        /// calls, but record_kernel() throws when cudaEventElapsedTime fails, and that exception
        /// would otherwise escape into std::terminate.
        ~ScopedKernelTiming() noexcept;

        /// Exclusive to its scope: the constructor creates two events and the destructor destroys
        /// them, so a copy would have two objects destroying the same handles. Moving is refused for
        /// a different reason -- the timing is recorded when the object dies, so relocating it would
        /// move the record boundary somewhere the caller did not write.
        ScopedKernelTiming(const ScopedKernelTiming&) = delete;
        ScopedKernelTiming& operator=(const ScopedKernelTiming&) = delete;
        ScopedKernelTiming(ScopedKernelTiming&&) = delete;
        ScopedKernelTiming& operator=(ScopedKernelTiming&&) = delete;

        void record_blocks(size_t blocks, size_t threads_per_block) {
            blocks_ = blocks;
            threads_per_block_ = threads_per_block;
        }

    private:
        const char* name_;
        KernelStatsCollector& collector_;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t end_ = nullptr;
        size_t blocks_ = 0;
        size_t threads_per_block_ = 0;
    };

    class OccupancyMetrics {
    public:
        double theoretical_occupancy = 0.0;
        double achieved_occupancy = 0.0;
        size_t active_blocks_per_sm = 0;
        size_t max_blocks_per_sm = 0;
    };

    OccupancyMetrics measure_occupancy(const void* kernel_func, size_t dynamic_smem_bytes = 0, int device = 0);

}  // namespace cuda::observability
