#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "buffer.h"

namespace cuda::memory {

    class MemoryPool {
    public:
        struct Config {
            Config() = default;
            size_t block_size = 1 << 20;
            size_t max_blocks = 16;
            bool preallocate = true;
        };

        struct PoolMetrics {
            size_t hits = 0;
            size_t misses = 0;
            size_t fragmentation_bytes = 0;
            double fragmentation_percent = 0.0;
            size_t peak_allocated_bytes = 0;
            int num_active_streams = 0;
        };

        explicit MemoryPool();
        explicit MemoryPool(const Config& config);
        ~MemoryPool();

        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

        MemoryPool(MemoryPool&& other) noexcept;
        MemoryPool& operator=(MemoryPool&& other) noexcept;

        void* allocate(size_t bytes, int stream_id = -1);
        void deallocate(void* ptr, size_t bytes);

        size_t total_allocated() const { return total_allocated_; }
        size_t total_available() const { return total_available_; }
        size_t num_blocks() const { return blocks_.size(); }
        size_t num_allocations() const { return allocation_map_.size(); }

        PoolMetrics get_metrics() const;
        void defragment();

        std::unordered_map<int, size_t> get_allocations_by_stream() const;
        void synchronize_stream(int stream_id);

        void set_throw_on_failure(bool throw_on_failure);
        bool get_throw_on_failure() const { return throw_on_failure_; }

        void clear();

    private:
        struct Block {
            void* ptr = nullptr;
            size_t size = 0;
            size_t offset = 0;
            int stream_id = -1;
        };

        struct AllocationInfo {
            size_t block_idx;
            int stream_id;
        };

        Config config_;
        std::vector<Block> blocks_;
        std::unordered_map<void*, AllocationInfo> allocation_map_;
        mutable std::mutex mutex_;
        mutable std::mutex metrics_mutex_;
        size_t total_allocated_ = 0;
        size_t total_available_ = 0;
        size_t peak_allocated_bytes_ = 0;
        size_t hits_ = 0;
        size_t misses_ = 0;
        bool throw_on_failure_ = true;

        Block* allocate_block();
        void free_block(Block& block);
        Block* find_block_for_size(size_t bytes);
    };

    class ScopedMemoryPool {
    public:
        ScopedMemoryPool() = default;
        explicit ScopedMemoryPool(const MemoryPool::Config& config)
            : pool_(config) {}

        ~ScopedMemoryPool() = default;

        ScopedMemoryPool(const ScopedMemoryPool&) = delete;
        ScopedMemoryPool& operator=(const ScopedMemoryPool&) = delete;

        ScopedMemoryPool(ScopedMemoryPool&& other) noexcept = default;
        ScopedMemoryPool& operator=(ScopedMemoryPool&& other) noexcept = default;

        void* allocate(size_t bytes) { return pool_.allocate(bytes); }

        void deallocate(void* ptr, size_t bytes) { pool_.deallocate(ptr, bytes); }

        template <typename T>
        T* allocate(size_t count) {
            return static_cast<T*>(pool_.allocate(count * sizeof(T)));
        }

        template <typename T>
        void deallocate(T* ptr, size_t count) {
            pool_.deallocate(ptr, count * sizeof(T));
        }

        MemoryPool& get() { return pool_; }
        const MemoryPool& get() const { return pool_; }

    private:
        MemoryPool pool_;
    };

}  // namespace cuda::memory
