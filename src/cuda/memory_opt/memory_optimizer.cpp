#include "cuda/memory_opt/memory_optimizer.h"

#include <algorithm>
#include <cstring>
#include <numeric>

#if NOVA_USE_ZSTD
#include <zstd.h>
#elif NOVA_USE_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

namespace cuda::memory_opt {

CheckpointCompressor::CheckpointCompressor()
    : config_(), compression_ratio_(1.0f), total_original_bytes_(0), total_compressed_bytes_(0) {}

CheckpointCompressor& CheckpointCompressor::instance() {
    static CheckpointCompressor compressor;
    return compressor;
}

void CheckpointCompressor::set_config(const CompressionConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

CompressionConfig CheckpointCompressor::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

size_t CheckpointCompressor::compress(
    const void* input,
    size_t input_size,
    void* output,
    size_t output_capacity
) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        total_original_bytes_ += input_size;
    }

    if (!config_.enable_compression || input_size < config_.min_size_for_compression) {
        if (input_size <= output_capacity) {
            memcpy(output, input, input_size);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                total_compressed_bytes_ += input_size;
                compression_ratio_ = total_original_bytes_ > 0
                    ? static_cast<float>(total_original_bytes_) / total_compressed_bytes_
                    : 1.0f;
            }
            return input_size;
        }
        return 0;
    }

    size_t compressed_size = 0;

#if NOVA_USE_ZSTD
    compressed_size = ZSTD_compress(
        output, output_capacity, input, input_size, config_.compression_level);
    if (ZSTD_isError(compressed_size)) {
        memcpy(output, input, std::min(input_size, output_capacity));
        compressed_size = std::min(input_size, output_capacity);
    }
#elif NOVA_USE_LZ4
    int lz4_size = LZ4_compress_default(
        static_cast<const char*>(input),
        static_cast<char*>(output),
        static_cast<int>(input_size),
        static_cast<int>(output_capacity));
    if (lz4_size == 0) {
        memcpy(output, input, std::min(input_size, output_capacity));
        compressed_size = std::min(input_size, output_capacity);
    } else {
        compressed_size = static_cast<size_t>(lz4_size);
    }
#else
    memcpy(output, input, std::min(input_size, output_capacity));
    compressed_size = std::min(input_size, output_capacity);
#endif

    {
        std::lock_guard<std::mutex> lock(mutex_);
        total_compressed_bytes_ += compressed_size;
        compression_ratio_ = total_original_bytes_ > 0
            ? static_cast<float>(total_original_bytes_) / total_compressed_bytes_
            : 1.0f;
    }

    return compressed_size;
}

size_t CheckpointCompressor::decompress(
    const void* input,
    size_t input_size,
    void* output,
    size_t output_capacity
) {
#if NOVA_USE_ZSTD
    size_t decompressed_size = ZSTD_decompress(
        output, output_capacity, input, input_size);
    if (ZSTD_isError(decompressed_size)) {
        memcpy(output, input, std::min(input_size, output_capacity));
        return std::min(input_size, output_capacity);
    }
    return decompressed_size;
#elif NOVA_USE_LZ4
    int decompressed_size = LZ4_decompress_safe(
        static_cast<const char*>(input),
        static_cast<char*>(output),
        static_cast<int>(input_size),
        static_cast<int>(output_capacity));
    if (decompressed_size < 0) {
        memcpy(output, input, std::min(input_size, output_capacity));
        return std::min(input_size, output_capacity);
    }
    return decompressed_size;
#else
    memcpy(output, input, std::min(input_size, output_capacity));
    return std::min(input_size, output_capacity);
#endif
}

float CheckpointCompressor::get_average_compression_ratio() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return compression_ratio_;
}

GradientAccumulator::GradientAccumulator(int max_accumulation_steps)
    : max_accumulation_steps_(max_accumulation_steps),
      accumulated_gradients_(0) {}

GradientAccumulator::~GradientAccumulator() {}

void GradientAccumulator::add_gradient(int step, const float* gradient, size_t size) {
    if (step != current_step_) {
        reset();
        current_step_ = step;
        accumulated_gradients_.resize(size, 0.0f);
    }

    if (accumulated_gradients_.size() != size) {
        accumulated_gradients_.resize(size, 0.0f);
    }

    for (size_t i = 0; i < size; ++i) {
        accumulated_gradients_[i] += gradient[i];
    }

    // Gradient data is present from this call onwards. This must be set
    // regardless of whether step==current_step_ (e.g. the very first call
    // with step 0), otherwise the first accumulated microbatch would never
    // mark the accumulator as holding data.
    has_gradients_ = true;
}

void GradientAccumulator::get_accumulated_gradient(float* output) {
    if (!has_gradients_) return;
    memcpy(output, accumulated_gradients_.data(), accumulated_gradients_.size() * sizeof(float));
}

void GradientAccumulator::reset() {
    current_step_ = 0;
    has_gradients_ = false;
    accumulated_gradients_.clear();
}

bool GradientAccumulator::is_ready_to_apply() const {
    return has_gradients_ && current_step_ >= max_accumulation_steps_ - 1;
}

MemoryDefragmenter::MemoryDefragmenter(int device_id)
    : device_id_(device_id) {}

void MemoryDefragmenter::register_allocation(void* ptr, size_t size) {
    MemoryBlock block;
    block.ptr = ptr;
    block.size = size;
    block.in_use = true;
    block.device_id = device_id_;
    blocks_.push_back(block);
}

void MemoryDefragmenter::unregister_allocation(void* ptr) {
    for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
        if (it->ptr == ptr) {
            blocks_.erase(it);
            break;
        }
    }
}

void MemoryDefragmenter::defragment() {
    std::sort(blocks_.begin(), blocks_.end(),
        [](const MemoryBlock& a, const MemoryBlock& b) {
            return reinterpret_cast<size_t>(a.ptr) < reinterpret_cast<size_t>(b.ptr);
        });

    for (auto& block : blocks_) {
        if (!block.in_use || !reallocate_callback_) continue;

        void* new_ptr = block.ptr;
        size_t new_size = block.size;
        reallocate_callback_(block.ptr, block.size, new_ptr, new_size);

        if (new_ptr != block.ptr) {
            block.ptr = new_ptr;
        }
        block.size = new_size;
    }
}

size_t MemoryDefragmenter::get_total_fragmentation() const {
    if (blocks_.empty()) return 0;

    size_t total_free = 0;
    size_t prev_end = 0;

    for (const auto& block : blocks_) {
        size_t block_start = reinterpret_cast<size_t>(block.ptr);
        if (block_start > prev_end) {
            total_free += block_start - prev_end;
        }
        prev_end = block_start + block.size;
    }

    return total_free;
}

size_t MemoryDefragmenter::get_largest_free_block() const {
    if (blocks_.empty()) return 0;

    size_t max_free = 0;
    size_t prev_end = 0;

    for (const auto& block : blocks_) {
        size_t block_start = reinterpret_cast<size_t>(block.ptr);
        if (block_start > prev_end) {
            max_free = std::max(max_free, block_start - prev_end);
        }
        prev_end = block_start + block.size;
    }

    return max_free;
}

int MemoryDefragmenter::get_fragment_count() const {
    if (blocks_.empty()) return 0;

    int count = 0;
    size_t prev_end = 0;

    for (const auto& block : blocks_) {
        size_t block_start = reinterpret_cast<size_t>(block.ptr);
        if (block_start > prev_end) {
            count++;
        }
        prev_end = block_start + block.size;
    }

    return count;
}

void MemoryDefragmenter::set_reallocate_callback(ReallocateCallback callback) {
    reallocate_callback_ = callback;
}

MemoryOptimizationManager& MemoryOptimizationManager::instance() {
    static MemoryOptimizationManager manager;
    return manager;
}

void MemoryOptimizationManager::enable_checkpoint_compression(bool enable) {
    compression_enabled_ = enable;
}

void MemoryOptimizationManager::set_gradient_accumulation_steps(int steps) {
    gradient_accumulation_steps_ = steps;
}

void MemoryOptimizationManager::enable_defragmentation(bool enable) {
    defragmentation_enabled_ = enable;
}

void MemoryOptimizationManager::record_checkpoint_size(size_t original, size_t compressed) {
    total_original_bytes_ += original;
    total_compressed_bytes_ += compressed;
}

void MemoryOptimizationManager::record_defragmentation() {
    num_defragmentations_++;
}

MemoryOptimizationStats MemoryOptimizationManager::get_stats() const {
    MemoryOptimizationStats stats;
    stats.compressed_bytes = total_compressed_bytes_;
    stats.original_bytes = total_original_bytes_;
    stats.compression_ratio = total_original_bytes_ > 0
        ? static_cast<float>(total_original_bytes_) / total_compressed_bytes_
        : 1.0f;
    stats.gradient_accumulation_buffers = gradient_accumulation_steps_;
    stats.total_fragmentation = 0;
    stats.num_allocations = 0;
    stats.num_defragmentations = num_defragmentations_;
    return stats;
}

void MemoryOptimizationManager::reset_stats() {
    total_compressed_bytes_ = 0;
    total_original_bytes_ = 0;
    num_defragmentations_ = 0;
}

AdaptiveMemoryPoolTuner::AdaptiveMemoryPoolTuner()
    : config_(), adaptive_enabled_(true),
      current_usage_(0), peak_usage_(0), num_failures_(0),
      total_allocations_(0), total_deallocations_(0) {}

AdaptiveMemoryPoolTuner& AdaptiveMemoryPoolTuner::instance() {
    static AdaptiveMemoryPoolTuner tuner;
    return tuner;
}

void AdaptiveMemoryPoolTuner::set_config(const PoolTuningConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

PoolTuningConfig AdaptiveMemoryPoolTuner::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void AdaptiveMemoryPoolTuner::record_allocation(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_allocations_++;
    current_usage_ += bytes;
    peak_usage_ = std::max(peak_usage_, current_usage_);
    allocation_samples_.push_back(bytes);

    if (allocation_samples_.size() > static_cast<size_t>(config_.samples_for_adaptation)) {
        allocation_samples_.erase(allocation_samples_.begin());
    }
}

void AdaptiveMemoryPoolTuner::record_deallocation(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_deallocations_++;
    current_usage_ = current_usage_ > bytes ? current_usage_ - bytes : 0;
}

void AdaptiveMemoryPoolTuner::record_allocation_failure() {
    std::lock_guard<std::mutex> lock(mutex_);
    num_failures_++;
}

size_t AdaptiveMemoryPoolTuner::suggest_pool_size() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!adaptive_enabled_) {
        return config_.initial_pool_size;
    }

    size_t suggested = config_.initial_pool_size;
    float avg_size = 0.0f;

    if (!allocation_samples_.empty()) {
        avg_size = static_cast<float>(
            std::accumulate(allocation_samples_.begin(), allocation_samples_.end(), 0ULL)
        ) / allocation_samples_.size();

        suggested = static_cast<size_t>(avg_size * config_.samples_for_adaptation * 1.2f);
    }

    suggested = std::max(suggested, peak_usage_ * 2);
    suggested = std::min(suggested, config_.max_pool_size);

    return suggested;
}

bool AdaptiveMemoryPoolTuner::should_grow() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!adaptive_enabled_) return false;
    if (num_failures_ > 0) return true;

    float utilization = static_cast<float>(current_usage_) / config_.initial_pool_size;
    return utilization > 0.8f;
}

bool AdaptiveMemoryPoolTuner::should_shrink() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!adaptive_enabled_) return false;
    if (total_allocations_ < 10) return false;

    float utilization = static_cast<float>(current_usage_) / config_.initial_pool_size;
    return utilization < config_.shrink_threshold;
}

void AdaptiveMemoryPoolTuner::enable_adaptive_tuning() {
    std::lock_guard<std::mutex> lock(mutex_);
    adaptive_enabled_ = true;
}

void AdaptiveMemoryPoolTuner::disable_adaptive_tuning() {
    std::lock_guard<std::mutex> lock(mutex_);
    adaptive_enabled_ = false;
}

bool AdaptiveMemoryPoolTuner::is_adaptive_enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return adaptive_enabled_;
}

WorkloadProfile AdaptiveMemoryPoolTuner::detect_workload_profile() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return detect_workload_profile_locked();
}

// Caller must hold mutex_. Kept separate from detect_workload_profile() so
// get_stats() can reuse the logic without recursively locking the same
// non-recursive std::mutex (which would self-deadlock).
WorkloadProfile AdaptiveMemoryPoolTuner::detect_workload_profile_locked() const {
    if (allocation_samples_.empty()) {
        return WorkloadProfile::Training;
    }

    float avg_size = static_cast<float>(
        std::accumulate(allocation_samples_.begin(), allocation_samples_.end(), 0ULL)
    ) / allocation_samples_.size();

    if (avg_size > 100 * 1024 * 1024) {
        return WorkloadProfile::LargeBatch;
    } else if (avg_size < 10 * 1024 * 1024) {
        return WorkloadProfile::Inference;
    } else {
        return WorkloadProfile::SmallBatch;
    }
}

void AdaptiveMemoryPoolTuner::set_workload_profile(WorkloadProfile profile) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.profile = profile;

    switch (profile) {
        case WorkloadProfile::SmallBatch:
            config_.initial_pool_size = 256 * 1024 * 1024;
            break;
        case WorkloadProfile::LargeBatch:
            config_.initial_pool_size = 1024 * 1024 * 1024;
            break;
        case WorkloadProfile::Inference:
            config_.initial_pool_size = 128 * 1024 * 1024;
            break;
        case WorkloadProfile::Training:
        default:
            config_.initial_pool_size = 512 * 1024 * 1024;
            break;
    }
}

AdaptiveMemoryPoolTuner::TuningStats AdaptiveMemoryPoolTuner::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    TuningStats stats;
    stats.total_allocations = total_allocations_;
    stats.total_deallocations = total_deallocations_;
    stats.peak_usage = peak_usage_;
    stats.current_usage = current_usage_;
    stats.num_failures = num_failures_;
    stats.detected_profile = detect_workload_profile_locked();

    if (!allocation_samples_.empty()) {
        stats.average_allocation_size = static_cast<float>(
            std::accumulate(allocation_samples_.begin(), allocation_samples_.end(), 0ULL)
        ) / allocation_samples_.size();
    }

    return stats;
}

void AdaptiveMemoryPoolTuner::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    allocation_samples_.clear();
    deallocation_samples_.clear();
    current_usage_ = 0;
    peak_usage_ = 0;
    num_failures_ = 0;
    total_allocations_ = 0;
    total_deallocations_ = 0;
}

} // namespace cuda::memory_opt
