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
        /// Bytes this pool has handed out on the device.
        size_t allocated_bytes = 0;
        /// Room this pool could still hand out: the device's free memory plus what this pool
        /// already holds itself. It cannot reclaim another owner's blocks, so this is not the
        /// device's free memory.
        size_t available_bytes = 0;
        /// Device memory total, as reported by the driver.
        size_t total_memory = 0;
        /// Device memory free, as reported by the driver. Every figure derived from this
        /// struct used to come from total minus this pool's own allocations, which overstates
        /// free memory by whatever other tenants on the device are holding.
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

        /// Auto-allocation on the device with the most FREE memory, as the driver reports it.
        /// Falls back to device 0 when Config::enable_auto_allocation is false.
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

        /// Device with the most free memory, as reported by cudaMemGetInfo.
        ///
        /// Queried live: free memory changes as every process on the machine allocates. The
        /// previous implementation ranked devices by total minus this pool's own allocations,
        /// which ignores other tenants and sent every auto-allocation to the busiest device.
        int get_best_device() const;

        // Single-GPU fallback
        bool is_single_gpu() const { return pools_.size() == 1; }

        // Clear all pools
        void clear();

    private:
        Config config_;
        std::vector<MemoryPool> pools_;
        mutable std::mutex ownership_mutex_;
        std::unordered_map<void*, OwnershipRecord> ownership_map_;

        /// One device's memory picture, both figures from the driver.
        struct DeviceMemory {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
        };

        void initialize_pools();
        /// Queries one device, restoring the calling thread's device afterwards.
        DeviceMemory query_device_memory(int device_id) const;
    };

}  // namespace cuda::memory
