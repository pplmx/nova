#pragma once

#include "buffer.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace cuda::memory {

class MemoryPool {
public:
    struct Config {
        Config() = default;
        size_t block_size = 1 << 20;
        size_t max_blocks = 16;
        bool preallocate = true;
    };

    explicit MemoryPool();
    explicit MemoryPool(const Config& config);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    MemoryPool(MemoryPool&& other) noexcept;
    MemoryPool& operator=(MemoryPool&& other) noexcept;

    void* allocate(size_t bytes);
    void deallocate(void* ptr, size_t bytes);

    size_t total_allocated() const { return total_allocated_; }
    size_t total_available() const { return total_available_; }
    size_t num_blocks() const { return blocks_.size(); }
    size_t num_allocations() const { return allocation_map_.size(); }

    void clear();

private:
    struct Block {
        void* ptr = nullptr;
        size_t size = 0;
        size_t offset = 0;
    };

    Config config_;
    std::vector<Block> blocks_;
    std::unordered_map<void*, size_t> allocation_map_;
    mutable std::mutex mutex_;
    size_t total_allocated_ = 0;
    size_t total_available_ = 0;

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

    void* allocate(size_t bytes) {
        return pool_.allocate(bytes);
    }

    void deallocate(void* ptr, size_t bytes) {
        pool_.deallocate(ptr, bytes);
    }

    template<typename T>
    T* allocate(size_t count) {
        return static_cast<T*>(pool_.allocate(count * sizeof(T)));
    }

    template<typename T>
    void deallocate(T* ptr, size_t count) {
        pool_.deallocate(ptr, count * sizeof(T));
    }

    MemoryPool& get() { return pool_; }
    const MemoryPool& get() const { return pool_; }

private:
    MemoryPool pool_;
};

} // namespace cuda::memory
