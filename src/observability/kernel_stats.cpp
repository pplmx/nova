#include "cuda/observability/kernel_stats.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace cuda::observability {

void KernelStatsCollector::record_kernel(const char* name,
                                         cudaEvent_t start,
                                         cudaEvent_t end,
                                         size_t blocks,
                                         size_t threads_per_block) {
    float elapsed_us = 0;
    cudaEventElapsedTime(&elapsed_us, start, end);
    elapsed_us *= 1000.0;

    bool found = false;
    for (auto& stat : stats_) {
        if (stat.kernel_name == name) {
            found = true;
            stat.invocations++;
            stat.total_time_us += elapsed_us;
            stat.min_time_us = std::min(stat.min_time_us, elapsed_us);
            stat.max_time_us = std::max(stat.max_time_us, elapsed_us);
            stat.avg_time_us = stat.total_time_us / stat.invocations;
            stat.blocks_launched += blocks;
            stat.threads_per_block = threads_per_block;
            break;
        }
    }

    if (!found) {
        KernelStats stat;
        stat.kernel_name = name;
        stat.invocations = 1;
        stat.total_time_us = elapsed_us;
        stat.min_time_us = elapsed_us;
        stat.max_time_us = elapsed_us;
        stat.avg_time_us = elapsed_us;
        stat.blocks_launched = blocks;
        stat.threads_per_block = threads_per_block;
        stats_.push_back(stat);
    }
}

void KernelStatsCollector::record_launch(const char* name,
                                         size_t blocks,
                                         size_t threads_per_block) {
    bool found = false;
    for (auto& stat : stats_) {
        if (stat.kernel_name == name) {
            found = true;
            stat.blocks_launched += blocks;
            stat.threads_per_block = threads_per_block;
            break;
        }
    }

    if (!found) {
        KernelStats stat;
        stat.kernel_name = name;
        stat.invocations = 0;
        stat.blocks_launched = blocks;
        stat.threads_per_block = threads_per_block;
        stats_.push_back(stat);
    }
}

KernelStats KernelStatsCollector::get_stats(const char* name) const {
    for (const auto& stat : stats_) {
        if (stat.kernel_name == name) {
            return stat;
        }
    }
    return KernelStats{};
}

ScopedKernelTiming::ScopedKernelTiming(const char* name,
                                       KernelStatsCollector& collector,
                                       size_t blocks,
                                       size_t threads_per_block)
    : name_(name), collector_(collector), blocks_(blocks), threads_per_block_(threads_per_block) {
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
    cudaEventRecord(start_, collector.stream());
}

ScopedKernelTiming::~ScopedKernelTiming() {
    cudaEventRecord(end_, collector_.stream());
    cudaStreamSynchronize(collector_.stream());
    collector_.record_kernel(name_, start_, end_, blocks_, threads_per_block_);
    cudaEventDestroy(start_);
    cudaEventDestroy(end_);
}

OccupancyMetrics measure_occupancy(const void* kernel_func,
                                   size_t dynamic_smem_bytes,
                                   int device) {
    OccupancyMetrics metrics{};

    if (!kernel_func) {
        return metrics;
    }

    int old_device;
    cudaGetDevice(&old_device);
    if (device >= 0) {
        cudaSetDevice(device);
    }

    int num_blocks = 0;
    int active_blocks_per_sm = 0;
    size_t dynamic_smem = dynamic_smem_bytes;

    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        kernel_func,
        256,
        dynamic_smem);

    if (err == cudaSuccess) {
        metrics.active_blocks_per_sm = static_cast<size_t>(active_blocks_per_sm);
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        int block_size = 256;
        err = cudaOccupancyMaxPotentialBlockSize(
            &block_size,
            &active_blocks_per_sm,
            kernel_func,
            dynamic_smem,
            block_size);

        if (err == cudaSuccess) {
            metrics.max_blocks_per_sm = static_cast<size_t>(active_blocks_per_sm);
            int threads_per_block = block_size;
            int max_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks,
                kernel_func,
                threads_per_block,
                dynamic_smem);

            int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
            int threads_per_block_for_occupancy = threads_per_block * max_blocks;
            metrics.theoretical_occupancy =
                static_cast<double>(threads_per_block_for_occupancy) / max_threads_per_sm;
        }
    }

    cudaSetDevice(old_device);
    return metrics;
}

}  // namespace cuda::observability
