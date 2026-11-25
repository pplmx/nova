#include "cuda/memory/memory_pool.h"

#include <algorithm>
#include <unordered_set>

#include "cuda/device/error.h"

namespace cuda::memory {

    MemoryPool::MemoryPool()
        : MemoryPool(Config()) {}

    MemoryPool::MemoryPool(const Config& config)
        : config_(config) {
        if (config_.preallocate) {
            for (size_t i = 0; i < config_.max_blocks; ++i) {
                if (allocate_block()) {
                    ++misses_;
                }
            }
        }
    }

    MemoryPool::~MemoryPool() {
        release_noexcept();
    }

    MemoryPool::MemoryPool(MemoryPool&& other) noexcept
        : config_(other.config_),
          blocks_(std::move(other.blocks_)),
          allocation_map_(std::move(other.allocation_map_)),
          total_allocated_(other.total_allocated_),
          total_available_(other.total_available_) {
        other.blocks_.clear();
        other.allocation_map_.clear();
        other.total_allocated_ = 0;
        other.total_available_ = 0;
    }

    MemoryPool& MemoryPool::operator=(MemoryPool&& other) noexcept {
        if (this != &other) {
            release_noexcept();
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

        // push_back throws bad_alloc when the vector grows. Block is plain data, so that is
        // the only way it fails, but if it does the block just allocated has no owner yet and
        // would never be freed — the same shape as the constructor leaks fixed earlier.
        try {
            blocks_.push_back(block);
        } catch (...) {
            (void)cudaFree(ptr);
            throw;
        }
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

    void* MemoryPool::allocate(size_t bytes, int stream_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (bytes == 0) {
            return nullptr;
        }

        // Check the size before touching the block list. allocate_block() adds to total_available_,
        // so refusing after it ran would leave a block allocated for a request that is never served.
        if (bytes > config_.block_size) {
            if (!throw_on_failure_) {
                return nullptr;
            }
            throw std::runtime_error("MemoryPool: request of " + std::to_string(bytes) + " bytes exceeds the block size of " + std::to_string(config_.block_size) + " bytes");
        }

        Block* block = find_block_for_size(bytes);

        if (block) {
            ++hits_;
        } else {
            ++misses_;
            block = allocate_block();
            if (!block) {
                if (!throw_on_failure_) {
                    return nullptr;
                }
                throw std::runtime_error("MemoryPool: failed to allocate block");
            }
        }

        void* result = static_cast<char*>(block->ptr) + block->offset;
        allocation_map_[result] = {static_cast<size_t>(block - blocks_.data()), stream_id, bytes};
        block->offset += bytes;
        block->stream_id = stream_id;
        total_allocated_ += bytes;
        total_available_ -= bytes;

        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, total_allocated_);

        return result;
    }

    void MemoryPool::deallocate(void* ptr, size_t bytes) {
        if (!ptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocation_map_.find(ptr);
        if (it == allocation_map_.end()) {
            return;
        }

        size_t block_idx = it->second.block_idx;
        if (block_idx >= blocks_.size()) {
            return;
        }

        Block& block = blocks_[block_idx];

        // Release the size that was actually handed out, not the one reported back. The pool records
        // it per allocation so an overstated `bytes` cannot corrupt the accounting; `bytes` is kept
        // in the signature for source compatibility only.
        (void)bytes;
        const size_t released = it->second.bytes;

        if (ptr >= block.ptr && ptr < static_cast<char*>(block.ptr) + block.size) {
            if (total_allocated_ >= released) {
                total_allocated_ -= released;
            } else {
                total_allocated_ = 0;
            }
            total_available_ += released;

            allocation_map_.erase(it);

            // offset is rewound only when this release empties the block.
            //
            // It used to be rewound unconditionally (block.offset -= released), which made the bump
            // pointer the block's only notion of free space. That is wrong for any release that is
            // not the most recent allocation. Measured with a 4096-byte block:
            //
            //   a = allocate(256) -> 0x...000
            //   b = allocate(256) -> 0x...100
            //   c = allocate(256) -> 0x...200
            //   deallocate(a)                        // offset stepped back to 0x200
            //   d = allocate(256) -> 0x...200        // the address c is still using
            //
            // Two live allocations then shared memory, and the next write through one corrupted the
            // other. Counting the block's surviving allocations fixes that without giving up reuse:
            // while any allocation inside the block lives, the offset stays put and later requests
            // go to another block; once none survive the block is empty and its offset returns to
            // zero, which is the pooling behaviour callers already rely on.
            //
            // The block keeps its cudaMalloc. The pointer must NOT be cleared here: clear() and the
            // destructor release blocks by checking block.ptr, so nulling it leaked the whole block.
            size_t live_in_block = 0;
            for (const auto& entry : allocation_map_) {
                if (entry.second.block_idx == block_idx) {
                    ++live_in_block;
                }
            }
            if (live_in_block == 0) {
                block.offset = 0;
                block.stream_id = -1;
            }
        } else {
            allocation_map_.erase(it);
        }
    }

    void MemoryPool::clear() {
        std::lock_guard<std::mutex> lock(mutex_);

        // The first failing cudaFree used to abort the loop, so every block after it was leaked
        // while the pool went on reporting itself as holding them. The error is recorded and
        // rethrown once the list has been walked, so a single bad pointer cannot strand the rest.
        cudaError_t first_error = cudaSuccess;
        for (auto& block : blocks_) {
            if (block.ptr) {
                const cudaError_t status = cudaFree(block.ptr);
                if (status != cudaSuccess && first_error == cudaSuccess) {
                    first_error = status;
                }
            }
        }

        blocks_.clear();
        allocation_map_.clear();
        total_allocated_ = 0;
        total_available_ = 0;
        hits_ = 0;
        misses_ = 0;

        if (first_error != cudaSuccess) {
            throw ::cuda::device::CudaException(first_error, __FILE__, __LINE__);
        }
    }

    void MemoryPool::release_noexcept() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& block : blocks_) {
            if (block.ptr) {
                // Deliberately not CUDA_CHECK: see the header. A context that has already been
                // destroyed makes this return cudaErrorInvalidValue, and throwing here runs into
                // std::terminate.
                (void)cudaFree(block.ptr);
            }
        }

        blocks_.clear();
        allocation_map_.clear();
        total_allocated_ = 0;
        total_available_ = 0;
        hits_ = 0;
        misses_ = 0;
    }

    MemoryPool::PoolMetrics MemoryPool::get_metrics() const {
        // mutex_, not metrics_mutex_. This walks blocks_, and allocate_block() grows that vector
        // under mutex_; taking a different lock here meant the two could run at the same time, so a
        // concurrent get_metrics() could iterate a vector mid-reallocation. metrics_mutex_ had
        // exactly one user -- this function -- so it protected nothing; it is gone.
        std::lock_guard<std::mutex> lock(mutex_);

        PoolMetrics metrics;
        metrics.hits = hits_;
        metrics.misses = misses_;
        metrics.peak_allocated_bytes = peak_allocated_bytes_;

        std::unordered_set<int> active_streams;
        size_t total_block_space = 0;
        size_t total_free_space = 0;
        for (const auto& block : blocks_) {
            if (block.ptr) {
                total_block_space += block.size;
                total_free_space += (block.size - block.offset);
            }
        }

        // Count the streams that hold live allocations, from allocation_map_, not the streams the
        // blocks happen to be tagged with. Block::stream_id only records the stream of the most
        // recent allocate() into that block, so a block holding allocations from three streams
        // reports one of them. Measured: four 256-byte allocations on streams 0, 1, 2 and 1 all fit
        // one 4096-byte block; get_allocations_by_stream() returned 3 entries while
        // num_active_streams reported 1. The two accessors describe the same state and have to agree.
        for (const auto& [ptr, info] : allocation_map_) {
            (void)ptr;
            if (info.stream_id >= 0) {
                active_streams.insert(info.stream_id);
            }
        }
        metrics.num_active_streams = static_cast<int>(active_streams.size());

        if (total_block_space > 0) {
            metrics.fragmentation_bytes = total_free_space;
            metrics.fragmentation_percent = static_cast<double>(total_free_space) / static_cast<double>(total_block_space) * 100.0;
        }

        return metrics;
    }

    void MemoryPool::defragment() {
        std::lock_guard<std::mutex> lock(mutex_);

        // Refuse while any allocation is live.
        //
        // The body frees each used block and mallocs a replacement, which moves the memory. Every
        // pointer this pool handed out becomes dangling, and allocation_map_ still records the old
        // addresses, so deallocate() would no longer find them. Measured before this guard, with one
        // block holding 1000 used bytes: defragment() moved it, and cudaMemcpy from the caller's
        // pointer returned cudaErrorInvalidValue. It also left Block::size at the old block_size while
        // the allocation was shrunk to used bytes, so total_available() rose from 3096 to 4096 and a
        // later allocate(3000) was served out of memory that no longer existed.
        //
        // Making the move safe needs a way to tell callers their pointers changed, which the API has
        // no room for: allocate() returns a raw void*. So the operation is defined for the case where
        // it can be correct -- no live allocations, nothing to invalidate -- and rejected otherwise.
        if (!allocation_map_.empty()) {
            return;
        }

        for (auto& block : blocks_) {
            if (block.ptr && block.offset > 0) {
                size_t used_size = block.offset;
                void* new_ptr = nullptr;
                CUDA_CHECK(cudaMalloc(&new_ptr, used_size));
                CUDA_CHECK(cudaMemcpy(new_ptr, block.ptr, used_size, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaFree(block.ptr));
                block.ptr = new_ptr;
                block.offset = 0;
            }
        }

        total_available_ = 0;
        for (const auto& block : blocks_) {
            if (block.ptr) {
                total_available_ += (block.size - block.offset);
            }
        }
    }

    std::unordered_map<int, size_t> MemoryPool::get_allocations_by_stream() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::unordered_map<int, size_t> result;
        for (const auto& [ptr, info] : allocation_map_) {
            if (info.block_idx < blocks_.size()) {
                const Block& block = blocks_[info.block_idx];
                result[info.stream_id] += block.offset;
            }
        }
        return result;
    }

    void MemoryPool::synchronize_stream(int stream_id) {
        // stream_id is unused by design limitation, not by oversight: the pool stores the integer a
        // caller passed to allocate(), never the cudaStream_t it refers to, so there is no handle to
        // hand to cudaStreamSynchronize. Synchronising stream 0 is the whole behaviour this method
        // has ever had -- but a caller reading `synchronize_stream(my_stream_id)` would reasonably
        // expect otherwise, which is why the header now says so.
        (void)stream_id;
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void MemoryPool::set_throw_on_failure(bool throw_on_failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        throw_on_failure_ = throw_on_failure;
    }

}  // namespace cuda::memory
