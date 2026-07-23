#include <cuda/performance/kernel_profiler.h>

#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace cuda::performance {

KernelProfiler& KernelProfiler::instance() {
    static KernelProfiler profiler;
    return profiler;
}

void KernelProfiler::set_callback(ProfilerCallback callback) {
    callback_ = std::move(callback);
}

void KernelProfiler::record_start(const std::string& kernel_name, cudaStream_t stream) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto record = get_or_create_record(kernel_name);
    if (record && !record->start_event) {
        CUDA_CHECK(cudaEventCreate(&record->start_event));
        CUDA_CHECK(cudaEventCreate(&record->stop_event));
        CUDA_CHECK(cudaEventRecord(record->start_event, stream));
    }
}

void KernelProfiler::record_end(const std::string& kernel_name, cudaStream_t stream) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(kernel_name);
    if (it != records_.end() && it->second) {
        auto& record = it->second;
        if (record->start_event && !record->completed) {
            CUDA_CHECK(cudaEventRecord(record->stop_event, stream));
            CUDA_CHECK(cudaEventSynchronize(record->stop_event));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, record->start_event, record->stop_event));
            record->latency_ns = static_cast<uint64_t>(elapsed_ms * 1e6);
            record->completed = true;

            if (callback_) {
                callback_(kernel_name, record->latency_ns);
            }
        }
    }
}

uint64_t KernelProfiler::get_kernel_latency_ns(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(kernel_name);
    if (it != records_.end() && it->second) {
        return it->second->latency_ns;
    }
    return 0;
}

float KernelProfiler::estimate_occupancy(int block_size, int max_blocks_per_sm) const {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return 0.0f;
    }

    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int threads_per_block = std::min(block_size, prop.maxThreadsPerBlock);
    int blocks_needed = (threads_per_block + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock;
    int active_blocks = std::min(max_blocks_per_sm, blocks_needed);

    return static_cast<float>(active_blocks * threads_per_block) / static_cast<float>(max_threads_per_sm);
}

void KernelProfiler::enable() {
    enabled_ = true;
}

void KernelProfiler::disable() {
    enabled_ = false;
}

bool KernelProfiler::is_enabled() const {
    return enabled_;
}

void KernelProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, record] : records_) {
        if (record->start_event) CUDA_CHECK(cudaEventDestroy(record->start_event));
        if (record->stop_event) CUDA_CHECK(cudaEventDestroy(record->stop_event));
    }
    records_.clear();
}

std::unique_ptr<KernelRecord> KernelProfiler::get_or_create_record(const std::string& kernel_name) {
    auto it = records_.find(kernel_name);
    if (it == records_.end()) {
        records_[kernel_name] = std::make_unique<KernelRecord>();
    }
    return std::move(records_[kernel_name]);
}

ScopedKernelProfile::ScopedKernelProfile(const std::string& kernel_name, cudaStream_t stream)
    : kernel_name_(kernel_name), stream_(stream) {
    KernelProfiler::instance().record_start(kernel_name_, stream_);
}

ScopedKernelProfile::~ScopedKernelProfile() {
    KernelProfiler::instance().record_end(kernel_name_, stream_);
}

float OccupancyCalculator::calculate_theoretical_occupancy(
    int threads_per_block,
    int registers_per_thread,
    int shared_mem_bytes,
    int device_id
) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return 0.0f;
    }

    int threads_per_block_clamped = std::min(threads_per_block, prop.maxThreadsPerBlock);
    int max_blocks_per_sm = std::min(
        prop.maxThreadsPerMultiProcessor / threads_per_block_clamped,
        prop.maxBlocksPerMultiProcessor
    );

    int registers_per_block = registers_per_thread * threads_per_block_clamped;
    int max_blocks_by_registers = prop.regsPerBlock > 0
        ? prop.regsPerBlock / registers_per_block
        : prop.maxBlocksPerMultiProcessor;
    max_blocks_per_sm = std::min(max_blocks_per_sm, max_blocks_by_registers);

    if (prop.sharedMemPerBlock > 0) {
        int max_blocks_by_shared = prop.sharedMemPerBlock > 0
            ? static_cast<int>(prop.sharedMemPerBlock / std::max(shared_mem_bytes, 1))
            : prop.maxBlocksPerMultiProcessor;
        max_blocks_per_sm = std::min(max_blocks_per_sm, max_blocks_by_shared);
    }

    int active_threads = max_blocks_per_sm * threads_per_block_clamped;
    return static_cast<float>(active_threads) / static_cast<float>(prop.maxThreadsPerMultiProcessor);
}

int OccupancyCalculator::recommended_block_size(
    int threads_per_block,
    int registers_per_thread,
    int shared_mem_bytes,
    int device_id
) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return 256;
    }

    int best_block_size = 256;
    float best_occupancy = 0.0f;

    for (int bs = 32; bs <= prop.maxThreadsPerBlock; bs += 32) {
        float occ = calculate_theoretical_occupancy(bs, registers_per_thread, shared_mem_bytes, device_id);
        if (occ > best_occupancy) {
            best_occupancy = occ;
            best_block_size = bs;
        }
    }

    return best_block_size;
}

}  // namespace cuda::performance
