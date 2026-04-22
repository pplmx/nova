#pragma once

#include "buffer.h"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <memory>

namespace cuda::memory {

class MemoryPool {
public:
    struct Config {
        Config() = default;
        size_t block_size = 1 << 20;
        size_t max_blocks = 16;
        bool preallocate = false;
    };

    explicit MemoryPool();
    explicit MemoryPool(const Config& config);
    ~MemoryPool();

    Buffer<void> allocate(size_t bytes);
    void deallocate(Buffer<void> buffer);

    size_t total_allocated() const { return total_allocated_; }
    size_t total_available() const { return total_available_; }
    size_t num_blocks() const { return blocks_.size(); }

    void clear();

private:
    struct Block {
        void* ptr;
        size_t size;
        size_t offset;
        bool in_use;
    };

    Config config_;
    std::vector<Block> blocks_;
    mutable std::mutex mutex_;
    size_t total_allocated_ = 0;
    size_t total_available_ = 0;

    Block* allocate_block();
    void free_block(Block* block);
};

class ScopedMemoryPool {
public:
    ScopedMemoryPool();
    explicit ScopedMemoryPool(const MemoryPool::Config& config)
        : pool_(config) {}

    ~ScopedMemoryPool() = default;

    template<typename T>
    Buffer<T> allocate(size_t count) {
        auto buf = pool_.allocate(count * sizeof(T));
        return Buffer<T>(buf.release(), count);
    }

    MemoryPool& get() { return pool_; }

private:
    MemoryPool pool_;
};

} // namespace cuda::memory
