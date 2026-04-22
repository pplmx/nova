#include "cuda/memory/memory_pool.h"
#include "cuda/kernel/cuda_utils.h"
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
    block.in_use = false;

    blocks_.push_back(block);
    total_available_ += config_.block_size;

    return &blocks_.back();
}

void MemoryPool::free_block(Block* block) {
    if (block && block->ptr) {
        CUDA_CHECK(cudaFree(block->ptr));
        total_available_ -= block->size;

        auto it = std::find_if(blocks_.begin(), blocks_.end(),
            [block](const Block& b) { return &b == block; });
        if (it != blocks_.end()) {
            blocks_.erase(it);
        }
    }
}

Buffer<void> MemoryPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (bytes == 0) {
        return Buffer<void>(0);
    }

    for (auto& block : blocks_) {
        if (!block.in_use && (block.size - block.offset) >= bytes) {
            void* alloc_ptr = static_cast<char*>(block.ptr) + block.offset;
            block.offset += bytes;
            total_allocated_ += bytes;
            total_available_ -= bytes;

            void* result = nullptr;
            CUDA_CHECK(cudaMalloc(&result, bytes));
            if (bytes > 0) {
                CUDA_CHECK(cudaMemcpy(result, alloc_ptr, bytes, cudaMemcpyDeviceToDevice));
            }
            return Buffer<void>(bytes);
        }
    }

    Block* new_block = allocate_block();
    if (new_block && (new_block->size - new_block->offset) >= bytes) {
        void* alloc_ptr = static_cast<char*>(new_block->ptr) + new_block->offset;
        new_block->offset = bytes;
        total_allocated_ += bytes;
        total_available_ -= bytes;

        void* result = nullptr;
        CUDA_CHECK(cudaMalloc(&result, bytes));
        if (bytes > 0) {
            CUDA_CHECK(cudaMemcpy(result, alloc_ptr, bytes, cudaMemcpyDeviceToDevice));
        }
        return Buffer<void>(bytes);
    }

    void* result = nullptr;
    CUDA_CHECK(cudaMalloc(&result, bytes));
    total_allocated_ += bytes;
    return Buffer<void>(bytes);
}

void MemoryPool::deallocate(Buffer<void> buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer.size() > 0) {
        total_allocated_ -= buffer.size();
    }
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& block : blocks_) {
        if (block.ptr) {
            CUDA_CHECK(cudaFree(block.ptr));
        }
    }

    blocks_.clear();
    total_allocated_ = 0;
    total_available_ = 0;
}

} // namespace cuda::memory
