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

        // Release the size that was actually handed out, not the one reported back. The pool
        // frees by walking the block's bump offset backwards, so an overstated `bytes`
        // underflowed it into a huge value: measured with a 100-byte allocation released as
        // 200 bytes, total_allocated became 18446744073709551516 and the NEXT allocate()
        // computed (block.size - block.offset) on wrapped arithmetic, believed the block had
        // thousands of free bytes, and returned a pointer 100 bytes BEFORE the block start --
        // outside the cudaMalloc'd region.
        //
        // `bytes` is kept in the signature for source compatibility; the recorded value is what
        // the accounting uses, so a caller that confuses element count with byte count can no
        // longer corrupt the pool.
        (void)bytes;
        const size_t released = it->second.bytes;

        if (ptr >= block.ptr && ptr < static_cast<char*>(block.ptr) + block.size) {
            // A block whose offset is already below the recorded size means the accounting no
            // longer matches (a non-LIFO release); clamp instead of wrapping.
            block.offset = (block.offset >= released) ? (block.offset - released) : 0;
            if (total_allocated_ >= released) {
                total_allocated_ -= released;
            } else {
                total_allocated_ = 0;
            }
            total_available_ += released;

            // A fully released block keeps its allocation so it can serve the next
            // request; that is the point of pooling. The pointer must NOT be cleared
            // here: clear() and the destructor release blocks by checking block.ptr,
            // so nulling it leaked the whole block. Measured with
            // compute-sanitizer --tool memcheck --leak-check full: a pool that had one
            // block allocated and fully released leaked 4096 bytes, while a pool that
            // only preallocated without ever allocating released nothing.
            if (block.offset == 0) {
                block.stream_id = -1;
            }
        }

        allocation_map_.erase(it);
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
        std::lock_guard<std::mutex> lock(metrics_mutex_);

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
                if (block.stream_id >= 0) {
                    active_streams.insert(block.stream_id);
                }
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
