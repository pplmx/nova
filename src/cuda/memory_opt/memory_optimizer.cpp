#include <stdexcept>
#include <string>
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
        : config_(),
          compression_ratio_(1.0f),
          total_original_bytes_(0),
          total_compressed_bytes_(0) {}

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

    size_t CheckpointCompressor::compress(const void* input, size_t input_size, void* output, size_t output_capacity) {
        // Every path reports bytes WRITTEN or 0 for "did not fit". Reporting success for a
        // buffer it could only partly fill is the one answer a caller cannot recover from,
        // because the returned size is indistinguishable from a real compression result.
        const auto record = [this](size_t written) {
            std::lock_guard<std::mutex> lock(mutex_);
            total_compressed_bytes_ += written;
            compression_ratio_ = total_compressed_bytes_ > 0 ? static_cast<float>(total_original_bytes_) / total_compressed_bytes_ : 1.0f;
        };

        {
            std::lock_guard<std::mutex> lock(mutex_);
            total_original_bytes_ += input_size;
        }

        if (!config_.enable_compression || input_size < config_.min_size_for_compression) {
            if (input_size > output_capacity) {
                return 0;
            }
            memcpy(output, input, input_size);
            record(input_size);
            return input_size;
        }

        size_t compressed_size = 0;

#if NOVA_USE_ZSTD
        compressed_size = ZSTD_compress(output, output_capacity, input, input_size, config_.compression_level);
        if (ZSTD_isError(compressed_size)) {
            // The codec could not fit the result. Fall back to a copy when it fits, and
            // refuse when it does not -- a truncated copy is not recoverable data.
            if (input_size > output_capacity) {
                return 0;
            }
            memcpy(output, input, input_size);
            compressed_size = input_size;
        }
#elif NOVA_USE_LZ4
        const int lz4_size = LZ4_compress_default(static_cast<const char*>(input), static_cast<char*>(output), static_cast<int>(input_size), static_cast<int>(output_capacity));
        if (lz4_size == 0) {
            if (input_size > output_capacity) {
                return 0;
            }
            memcpy(output, input, input_size);
            compressed_size = input_size;
        } else {
            compressed_size = static_cast<size_t>(lz4_size);
        }
#else
        // No codec is compiled into this build, so the input is stored verbatim. The path used
        // to memcpy min(input_size, output_capacity) bytes and report that count, so with a
        // buffer smaller than the input it silently truncated and returned a success-looking
        // size:
        //
        //     in=4096 cap=100 -> returned 100
        //
        // The other three paths already refuse in that situation; this one now does too,
        // which also makes the two builds behave the same way from the caller's side.
        if (input_size > output_capacity) {
            return 0;
        }
        memcpy(output, input, input_size);
        compressed_size = input_size;
#endif

        record(compressed_size);
        return compressed_size;
    }

    size_t CheckpointCompressor::decompress(const void* input, size_t input_size, void* output, size_t output_capacity) {
#if NOVA_USE_ZSTD
        size_t decompressed_size = ZSTD_decompress(output, output_capacity, input, input_size);
        if (ZSTD_isError(decompressed_size)) {
            memcpy(output, input, std::min(input_size, output_capacity));
            return std::min(input_size, output_capacity);
        }
        return decompressed_size;
#elif NOVA_USE_LZ4
        int decompressed_size = LZ4_decompress_safe(static_cast<const char*>(input), static_cast<char*>(output), static_cast<int>(input_size), static_cast<int>(output_capacity));
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
          accumulated_gradients_(0) {
        // A non-positive accumulation count is not a smaller accumulation, it is a different
        // behaviour: is_ready_to_apply() compares `current_step_ >= max_accumulation_steps_ - 1`,
        // and with max == 0 that bound is -1, so the very first add_gradient() makes the
        // accumulator report ready. Measured at step 0 with a single 2-element gradient:
        // max=-3 and max=0 both returned ready=true, meaning gradient accumulation is silently
        // disabled and every microbatch is applied on its own. Rejecting it here keeps the
        // comparison away from a bound the caller cannot have meant.
        if (max_accumulation_steps < 1) {
            throw ::std::invalid_argument("GradientAccumulator: max_accumulation_steps must be at least 1, got " + ::std::to_string(max_accumulation_steps));
        }
    }

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
        if (!has_gradients_) {
            return;
        }
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
        std::sort(blocks_.begin(), blocks_.end(), [](const MemoryBlock& a, const MemoryBlock& b) { return reinterpret_cast<size_t>(a.ptr) < reinterpret_cast<size_t>(b.ptr); });

        for (auto& block : blocks_) {
            if (!block.in_use || !reallocate_callback_) {
                continue;
            }

            void* new_ptr = block.ptr;
            size_t new_size = block.size;
            reallocate_callback_(block.ptr, block.size, new_ptr, new_size);

            if (new_ptr != block.ptr) {
                block.ptr = new_ptr;
            }
            block.size = new_size;
        }
    }

    /// Gaps between the registered blocks, in address order.
    ///
    /// Only the space BETWEEN blocks is fragmentation. Both accessors below used to seed the walk
    /// with prev_end = 0, so the first block's own address counted as free space: with two live
    /// 1000-byte allocations and nothing free between them, get_total_fragmentation() returned
    /// 93866660274872 and get_largest_free_block() 93866660274864 -- the process's virtual address
    /// of the first allocation, read as free bytes. Measured on this host.
    ///
    /// Shared because the two accessors had the same bug in the same shape and would otherwise drift.
    std::vector<size_t> MemoryDefragmenter::free_gaps() const {
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(blocks_.size());
        for (const auto& block : blocks_) {
            const auto start = reinterpret_cast<size_t>(block.ptr);
            ranges.emplace_back(start, start + block.size);
        }
        std::sort(ranges.begin(), ranges.end());

        std::vector<size_t> gaps;
        if (ranges.empty()) {
            return gaps;
        }
        size_t prev_end = ranges.front().second;
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first > prev_end) {
                gaps.push_back(ranges[i].first - prev_end);
            }
            prev_end = std::max(prev_end, ranges[i].second);
        }
        return gaps;
    }

    size_t MemoryDefragmenter::get_total_fragmentation() const {
        size_t total = 0;
        for (const size_t gap : free_gaps()) {
            total += gap;
        }
        return total;
    }

    size_t MemoryDefragmenter::get_largest_free_block() const {
        size_t largest = 0;
        for (const size_t gap : free_gaps()) {
            largest = std::max(largest, gap);
        }
        return largest;
    }

    int MemoryDefragmenter::get_fragment_count() const {
        if (blocks_.empty()) {
            return 0;
        }

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
        // Guard the DENOMINATOR, not the numerator. The old condition asked whether
        // anything was recorded at all and then divided by total_compressed_bytes_, so
        // recording an original size with nothing compressed produced +inf.
        stats.compression_ratio = total_compressed_bytes_ > 0 ? static_cast<float>(total_original_bytes_) / static_cast<float>(total_compressed_bytes_) : 1.0f;
        stats.gradient_accumulation_buffers = gradient_accumulation_steps_;
        stats.total_fragmentation = 0;
        stats.num_allocations = 0;
        stats.num_defragmentations = num_defragmentations_;
        // These two flags were stored by their setters and read by nothing, so a caller
        // could not tell whether its configuration took, and the stats could not explain
        // a zero compression ratio. Reported now.
        stats.compression_enabled = compression_enabled_;
        stats.defragmentation_enabled = defragmentation_enabled_;
        return stats;
    }

    void MemoryOptimizationManager::reset_stats() {
        total_compressed_bytes_ = 0;
        total_original_bytes_ = 0;
        num_defragmentations_ = 0;
    }

    AdaptiveMemoryPoolTuner::AdaptiveMemoryPoolTuner()
        : config_(),
          adaptive_enabled_(true),
          current_usage_(0),
          peak_usage_(0),
          num_failures_(0),
          total_allocations_(0),
          total_deallocations_(0) {}

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
            avg_size = static_cast<float>(std::accumulate(allocation_samples_.begin(), allocation_samples_.end(), 0ULL)) / allocation_samples_.size();

            suggested = static_cast<size_t>(avg_size * config_.samples_for_adaptation * 1.2f);
        }

        suggested = std::max(suggested, peak_usage_ * 2);
        suggested = std::min(suggested, config_.max_pool_size);

        return suggested;
    }

    bool AdaptiveMemoryPoolTuner::should_grow() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!adaptive_enabled_) {
            return false;
        }
        if (num_failures_ > 0) {
            return true;
        }

        // A zero initial_pool_size makes utilization inf: should_grow() would then always answer
        // yes and should_shrink() never. PoolTuningConfig is public and the field can be set to
        // anything, so the divisor is checked rather than assumed.
        if (config_.initial_pool_size == 0) {
            return false;
        }
        float utilization = static_cast<float>(current_usage_) / config_.initial_pool_size;
        return utilization > 0.8f;
    }

    bool AdaptiveMemoryPoolTuner::should_shrink() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!adaptive_enabled_) {
            return false;
        }
        if (total_allocations_ < 10) {
            return false;
        }

        if (config_.initial_pool_size == 0) {
            return false;
        }
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

        float avg_size = static_cast<float>(std::accumulate(allocation_samples_.begin(), allocation_samples_.end(), 0ULL)) / allocation_samples_.size();

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
            stats.average_allocation_size = static_cast<float>(std::accumulate(allocation_samples_.begin(), allocation_samples_.end(), 0ULL)) / allocation_samples_.size();
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

}  // namespace cuda::memory_opt
