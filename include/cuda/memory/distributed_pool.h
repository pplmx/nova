#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "memory_pool.h"

namespace cuda::memory {

struct OwnershipRecord {
    void* ptr = nullptr;
    size_t bytes = 0;
    int owning_device = -1;
    int requesting_device = -1;
};

struct DevicePoolMetrics {
    int device_id = -1;
    size_t allocated_bytes = 0;
    size_t available_bytes = 0;
    size_t total_memory = 0;
    size_t free_memory = 0;
    int num_allocations = 0;
    MemoryPool::PoolMetrics local_metrics;
};

class DistributedMemoryPool {
public:
    struct Config {
        Config() = default;
        size_t block_size = 1 << 20;
        size_t max_blocks_per_device = 16;
        bool enable_auto_allocation = true;
    };

    DistributedMemoryPool();
    explicit DistributedMemoryPool(const Config& config);
    ~DistributedMemoryPool();

    DistributedMemoryPool(const DistributedMemoryPool&) = delete;
    DistributedMemoryPool& operator=(const DistributedMemoryPool&) = delete;
    DistributedMemoryPool(DistributedMemoryPool&& other) noexcept;
    DistributedMemoryPool& operator=(DistributedMemoryPool&& other) noexcept;

    // MGPU-09: Per-device allocation
    void* allocate(size_t bytes, int device_id, int stream_id = -1);

    // MGPU-10: Auto-allocation (selects device with most free memory)
    void* allocate_auto(size_t bytes, int stream_id = -1);

    // MGPU-11: Ownership tracking - deallocate handles cross-device correctly
    void deallocate(void* ptr);

    // Get pointer ownership info
    OwnershipRecord get_ownership(void* ptr) const;

    // Check if pointer is tracked
    bool owns_pointer(void* ptr) const;

    // Per-device and aggregate statistics
    DevicePoolMetrics get_device_metrics(int device_id) const;
    std::vector<DevicePoolMetrics> get_all_metrics() const;

    // Device count
    int device_count() const { return static_cast<int>(pools_.size()); }

    // Get best device for allocation (most free memory)
    int get_best_device() const;

    // Single-GPU fallback
    bool is_single_gpu() const { return pools_.size() == 1; }

    // Clear all pools
    void clear();

private:
    Config config_;
    std::vector<MemoryPool> pools_;
    std::vector<size_t> device_total_memory_;
    mutable std::mutex ownership_mutex_;
    std::unordered_map<void*, OwnershipRecord> ownership_map_;

    void initialize_pools();
    size_t query_device_free_memory(int device_id) const;
};

}  // namespace cuda::memory
