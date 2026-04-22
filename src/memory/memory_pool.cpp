#include "cuda/memory/memory_pool.h"
#include "cuda/device/error.h"
#include <algorithm>

namespace cuda::memory {

MemoryPool::MemoryPool() : MemoryPool(Config()) {}

MemoryPool::MemoryPool(const Config& config) : config_(config) {
    if (config_.preallocate) {
        for (size_t i = 0; i < config_.max_blocks; ++i) {
            allocate_block();
        }
    }
}

MemoryPool::~MemoryPool() {
    clear();
}

MemoryPool::MemoryPool(MemoryPool&& other) noexcept
    : config_(other.config_)
    , blocks_(std::move(other.blocks_))
    , allocation_map_(std::move(other.allocation_map_))
    , total_allocated_(other.total_allocated_)
    , total_available_(other.total_available_) {
    other.blocks_.clear();
    other.allocation_map_.clear();
    other.total_allocated_ = 0;
    other.total_available_ = 0;
}

MemoryPool& MemoryPool::operator=(MemoryPool&& other) noexcept {
    if (this != &other) {
        clear();
        config_ = other.config_;
        blocks_ = std::move(other.blocks_);
        allocation_map_ = std::move(other.allocation_map_);
        total_allocated_ = other.total_allocated_;
        total_available_ = other.total_available_;
        other.blocks_.clear();
        other.allocation_map_.clear();
        other.total_allocated_ = 0;
        other.total_available_ = 0;
    }
    return *this;
}

MemoryPool::Block* MemoryPool::allocate_block() {
    if (blocks_.size() >= config_.max_blocks) {
        return nullptr;
    }

    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, config_.block_size));

    Block block;
    block.ptr = ptr;
    block.size = config_.block_size;
    block.offset = 0;

    blocks_.push_back(block);
    total_available_ += config_.block_size;

    return &blocks_.back();
}

void MemoryPool::free_block(Block& block) {
    if (block.ptr) {
        CUDA_CHECK(cudaFree(block.ptr));
        total_available_ -= (block.size - block.offset);
        block.ptr = nullptr;
        block.size = 0;
        block.offset = 0;
    }
}

MemoryPool::Block* MemoryPool::find_block_for_size(size_t bytes) {
    for (auto& block : blocks_) {
        if (block.ptr && (block.size - block.offset) >= bytes) {
            return &block;
        }
    }
    return nullptr;
}

void* MemoryPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (bytes == 0) {
        return nullptr;
    }

    Block* block = find_block_for_size(bytes);

    if (!block) {
        block = allocate_block();
        if (!block) {
            throw std::runtime_error("MemoryPool: failed to allocate block");
        }
    }

    void* result = static_cast<char*>(block->ptr) + block->offset;
    allocation_map_[result] = block - blocks_.data();
    block->offset += bytes;
    total_allocated_ += bytes;
    total_available_ -= bytes;

    return result;
}

void MemoryPool::deallocate(void* ptr, size_t bytes) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) {
        return;
    }

    size_t block_idx = it->second;
    if (block_idx >= blocks_.size()) {
        return;
    }

    Block& block = blocks_[block_idx];
    if (ptr >= block.ptr &&
        ptr < static_cast<char*>(block.ptr) + block.size) {
        block.offset -= bytes;
        total_allocated_ -= bytes;
        total_available_ += bytes;

        if (block.offset == 0) {
            block.ptr = nullptr;
        }
    }

    allocation_map_.erase(it);
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& block : blocks_) {
        if (block.ptr) {
            CUDA_CHECK(cudaFree(block.ptr));
        }
    }

    blocks_.clear();
    allocation_map_.clear();
    total_allocated_ = 0;
    total_available_ = 0;
}

} // namespace cuda::memory
