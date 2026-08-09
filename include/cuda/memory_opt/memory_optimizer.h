#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>

namespace cuda::memory_opt {

struct CompressionConfig {
    bool enable_compression = true;
    int compression_level = 3;
    size_t min_size_for_compression = 1024;
    float target_compression_ratio = 2.0f;
};

class CheckpointCompressor {
public:
    static CheckpointCompressor& instance();

    void set_config(const CompressionConfig& config);
    CompressionConfig get_config() const;

    size_t compress(
        const void* input,
        size_t input_size,
        void* output,
        size_t output_capacity
    );

    size_t decompress(
        const void* input,
        size_t input_size,
        void* output,
        size_t output_capacity
    );

    float get_compression_ratio() const { return compression_ratio_; }
    float get_average_compression_ratio() const;
    size_t get_total_original_bytes() const { return total_original_bytes_; }
    size_t get_total_compressed_bytes() const { return total_compressed_bytes_; }

private:
    CheckpointCompressor();

    CompressionConfig config_;
    float compression_ratio_ = 1.0f;
    size_t total_original_bytes_ = 0;
    size_t total_compressed_bytes_ = 0;
    mutable std::mutex mutex_;
};

class GradientAccumulator {
public:
    GradientAccumulator(int max_accumulation_steps);
    ~GradientAccumulator();

    void add_gradient(int step, const float* gradient, size_t size);
    void get_accumulated_gradient(float* output);
    void reset();
    bool is_ready_to_apply() const;

    int current_step() const { return current_step_; }
    int max_steps() const { return max_accumulation_steps_; }

private:
    int max_accumulation_steps_;
    int current_step_ = 0;
    std::vector<float> accumulated_gradients_;
    bool has_gradients_ = false;
};

struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    int device_id;
};

class MemoryDefragmenter {
public:
    MemoryDefragmenter(int device_id);

    void register_allocation(void* ptr, size_t size);
    void unregister_allocation(void* ptr);
    void defragment();

    size_t get_total_fragmentation() const;
    size_t get_largest_free_block() const;
    int get_fragment_count() const;

    using ReallocateCallback = std::function<void(void* old_ptr, size_t old_size, void*& new_ptr, size_t& new_size)>;
    void set_reallocate_callback(ReallocateCallback callback);

private:
    int device_id_;
    std::vector<MemoryBlock> blocks_;
    ReallocateCallback reallocate_callback_;
};

struct MemoryOptimizationStats {
    size_t compressed_bytes;
    size_t original_bytes;
    float compression_ratio;
    size_t gradient_accumulation_buffers;
    size_t total_fragmentation;
    int num_allocations;
    int num_defragmentations;
};

class MemoryOptimizationManager {
public:
    static MemoryOptimizationManager& instance();

    void enable_checkpoint_compression(bool enable);
    void set_gradient_accumulation_steps(int steps);
    void enable_defragmentation(bool enable);

    void record_checkpoint_size(size_t original, size_t compressed);
    void record_defragmentation();

    MemoryOptimizationStats get_stats() const;
    void reset_stats();

private:
    MemoryOptimizationManager() = default;

    bool compression_enabled_ = true;
    bool defragmentation_enabled_ = true;
    int gradient_accumulation_steps_ = 1;

    size_t total_compressed_bytes_ = 0;
    size_t total_original_bytes_ = 0;
    int num_defragmentations_ = 0;
};

enum class WorkloadProfile {
    SmallBatch,
    LargeBatch,
    Inference,
    Training
};

struct PoolTuningConfig {
    WorkloadProfile profile = WorkloadProfile::Training;
    size_t initial_pool_size = 256 * 1024 * 1024;
    size_t max_pool_size = 2 * 1024 * 1024 * 1024;
    float growth_factor = 1.5f;
    float shrink_threshold = 0.3f;
    int samples_for_adaptation = 100;
};

class AdaptiveMemoryPoolTuner {
public:
    static AdaptiveMemoryPoolTuner& instance();

    void set_config(const PoolTuningConfig& config);
    PoolTuningConfig get_config() const;

    void record_allocation(size_t bytes);
    void record_deallocation(size_t bytes);
    void record_allocation_failure();

    size_t suggest_pool_size() const;
    bool should_grow() const;
    bool should_shrink() const;

    void enable_adaptive_tuning();
    void disable_adaptive_tuning();
    bool is_adaptive_enabled() const;

    WorkloadProfile detect_workload_profile() const;
    void set_workload_profile(WorkloadProfile profile);

    struct TuningStats {
        size_t total_allocations;
        size_t total_deallocations;
        size_t peak_usage;
        size_t current_usage;
        float average_allocation_size;
        size_t num_failures;
        WorkloadProfile detected_profile;
    };

    TuningStats get_stats() const;
    void reset_stats();

    /** @brief Compute workload profile assuming mutex_ is already held. */
    WorkloadProfile detect_workload_profile_locked() const;

private:
    AdaptiveMemoryPoolTuner();

    PoolTuningConfig config_;
    bool adaptive_enabled_ = true;

    std::vector<size_t> allocation_samples_;
    std::vector<size_t> deallocation_samples_;
    mutable std::mutex mutex_;

    size_t current_usage_ = 0;
    size_t peak_usage_ = 0;
    size_t num_failures_ = 0;
    size_t total_allocations_ = 0;
    size_t total_deallocations_ = 0;
};

} // namespace cuda::memory_opt
