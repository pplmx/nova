#include <cuda/performance/kernel_profiler.h>
#include <stdexcept>
#include <string>

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
        if (!enabled_) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto record = get_or_create_record(kernel_name);
        if (record && !record->start_event) {
            CUDA_CHECK(cudaEventCreate(&record->start_event));
            CUDA_CHECK(cudaEventCreate(&record->stop_event));
            CUDA_CHECK(cudaEventRecord(record->start_event, stream));
        }
    }

    void KernelProfiler::record_end(const std::string& kernel_name, cudaStream_t stream) {
        if (!enabled_) {
            return;
        }

        // The callback is copied out and invoked after the lock is released.
        //
        // It used to be called straight from inside the lock_guard, and every other member of this
        // class takes the same non-recursive mutex_ -- get_kernel_latency_ns, reset, record_start.
        // A callback that asks the profiler anything therefore re-entered the lock it was called
        // under. Measured with a callback calling get_kernel_latency_ns(): record_start followed by
        // record_end never returned (timeout, exit 124). Same defect and same remedy as
        // Profiler::stop_timer, which deadlocked through record_kernel for exactly this reason, and
        // as timeout_manager's callbacks.
        bool measured = false;
        uint64_t latency_ns = 0;
        ProfilerCallback callback;
        {
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

                    measured = true;
                    latency_ns = record->latency_ns;
                    callback = callback_;
                }
            }
        }

        if (measured && callback) {
            callback(kernel_name, latency_ns);
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

        // Occupancy is resident threads over the SM's capacity.
        //
        // The old body computed `blocks_needed = ceil(threads_per_block / maxThreadsPerBlock)`,
        // which is 1 for every valid block size because a block can never be larger than
        // maxThreadsPerBlock -- so `active_blocks = min(max_blocks_per_sm, 1)` was always 1 and
        // the caller's max_blocks_per_sm argument had no effect whatsoever. Measured:
        // estimate_occupancy(1024, 1) and estimate_occupancy(1024, 64) both returned 0.5 on a
        // device with 2048 threads per SM, where 2 blocks of 1024 can be resident and the
        // answer is 1.0. The function reported the occupancy of exactly one block per SM and
        // called the result a percentage.
        //
        // The number of blocks that can be resident is bounded by both the caller's
        // max_blocks_per_sm and what the SM can hold, and the result is clamped to 1.0 because
        // a request for more blocks than fit cannot exceed full occupancy.
        if (block_size <= 0 || max_blocks_per_sm <= 0) {
            return 0.0f;
        }

        const int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        if (max_threads_per_sm <= 0) {
            return 0.0f;
        }

        // A block larger than the device allows cannot be launched; report nothing rather
        // than silently shrinking the caller's request and answering about a different block.
        if (block_size > prop.maxThreadsPerBlock) {
            return 0.0f;
        }

        const int blocks_that_fit = max_threads_per_sm / block_size;
        const int resident_blocks = std::min(max_blocks_per_sm, blocks_that_fit);
        const float occupancy = static_cast<float>(resident_blocks * block_size) / static_cast<float>(max_threads_per_sm);
        return std::min(occupancy, 1.0f);
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
            if (record->start_event) {
                CUDA_CHECK(cudaEventDestroy(record->start_event));
            }
            if (record->stop_event) {
                CUDA_CHECK(cudaEventDestroy(record->stop_event));
            }
        }
        records_.clear();
    }

    KernelRecord* KernelProfiler::get_or_create_record(const std::string& kernel_name) {
        // Returning std::move(records_[kernel_name]) handed ownership of the record to the caller
        // and left a null unique_ptr in the map. record_start() then worked on a record the profiler
        // no longer held, and record_end() -- which looks the name up again -- found the null and
        // did nothing. Measured: record_start("x") followed by record_end("x") left
        // get_kernel_latency_ns("x") at 0, so the profiler never reported a latency and never
        // invoked its callback. The map owns the record; the caller borrows it.
        // operator[] default-constructs on a missing key but leaves an EXISTING null entry alone, so
        // a record that was emptied earlier stays null forever and every later lookup for that name
        // returns nullptr.
        auto& slot = records_[kernel_name];
        if (!slot) {
            slot = std::make_unique<KernelRecord>();
        }
        return slot.get();
    }

    ScopedKernelProfile::ScopedKernelProfile(const std::string& kernel_name, cudaStream_t stream)
        : kernel_name_(kernel_name),
          stream_(stream) {
        KernelProfiler::instance().record_start(kernel_name_, stream_);
    }

    ScopedKernelProfile::~ScopedKernelProfile() noexcept {
        // record_end() reports failure by throwing (CUDA_CHECK on three event calls, plus whatever
        // a registered callback does). Letting that out of a destructor reaches std::terminate, so
        // a profiling scope can kill the process it is only supposed to be measuring.
        try {
            KernelProfiler::instance().record_end(kernel_name_, stream_);
        } catch (...) {
            // A lost sample is preferable to a terminated process; the timer is scoped to one
            // kernel launch and the profiler keeps whatever it already recorded.
        }
    }

    float OccupancyCalculator::calculate_theoretical_occupancy(int threads_per_block, int registers_per_thread, int shared_mem_bytes, int device_id) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
            return 0.0f;
        }

        // Both divisions below use a caller-supplied value as the divisor. A zero block
        // size or zero registers-per-thread performed integer division by zero, which
        // aborts the process; a negative block size slipped through the clamp (min of a
        // negative and the device maximum keeps the negative) and produced a nonsense
        // occupancy of 1.0. Reject the inputs instead.
        if (threads_per_block <= 0) {
            throw std::invalid_argument(
                "calculate_theoretical_occupancy: threads_per_block must be positive, "
                "got "
                + std::to_string(threads_per_block));
        }

        int threads_per_block_clamped = std::min(threads_per_block, prop.maxThreadsPerBlock);
        int max_blocks_per_sm = std::min(prop.maxThreadsPerMultiProcessor / threads_per_block_clamped, prop.maxBlocksPerMultiProcessor);

        // A kernel compiled without register pressure reports 0; treat that as "no
        // register limit" rather than dividing by zero.
        int registers_per_block = registers_per_thread * threads_per_block_clamped;
        int max_blocks_by_registers = (prop.regsPerBlock > 0 && registers_per_block > 0) ? prop.regsPerBlock / registers_per_block : prop.maxBlocksPerMultiProcessor;
        max_blocks_per_sm = std::min(max_blocks_per_sm, max_blocks_by_registers);

        if (prop.sharedMemPerBlock > 0) {
            int max_blocks_by_shared = prop.sharedMemPerBlock > 0 ? static_cast<int>(prop.sharedMemPerBlock / std::max(shared_mem_bytes, 1)) : prop.maxBlocksPerMultiProcessor;
            max_blocks_per_sm = std::min(max_blocks_per_sm, max_blocks_by_shared);
        }

        int active_threads = max_blocks_per_sm * threads_per_block_clamped;
        return static_cast<float>(active_threads) / static_cast<float>(prop.maxThreadsPerMultiProcessor);
    }

    int OccupancyCalculator::recommended_block_size(int registers_per_thread, int shared_mem_bytes, int device_id) {
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
