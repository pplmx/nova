#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <cstddef>

namespace cuda::memory {

class MemoryPool;

template<typename T>
concept DeviceAllocator = requires(T alloc, size_t n, size_t size) {
    { alloc.allocate(n * size) } -> std::same_as<void*>;
    { alloc.deallocate(nullptr, n * size) } -> std::same_as<void>;
    { alloc.max_size() } -> std::same_as<size_t>;
};

template<typename T>
concept TrackedAllocator = DeviceAllocator<T> && requires(T alloc) {
    { alloc.total_allocated() } -> std::same_as<size_t>;
    { alloc.available() } -> std::same_as<size_t>;
};

struct DefaultAllocator {
    void* allocate(size_t bytes) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void deallocate(void* ptr, size_t) {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    size_t max_size() const { return SIZE_MAX; }
};

struct PooledAllocator {
    PooledAllocator() = default;
    explicit PooledAllocator(MemoryPool& pool) : pool_(&pool) {}

    void* allocate(size_t bytes) {
        if (pool_) {
            auto buf = pool_->allocate(bytes);
            return buf.data();
        }
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void deallocate(void* ptr, size_t bytes) {
        if (pool_) {
            pool_->deallocate(Buffer<void>(ptr, bytes));
        } else if (ptr) {
            cudaFree(ptr);
        }
    }

    size_t max_size() const { return SIZE_MAX; }

    size_t total_allocated() const {
        return pool_ ? pool_->total_allocated() : 0;
    }

    size_t available() const {
        return pool_ ? pool_->total_available() : 0;
    }

    void set_pool(MemoryPool* pool) { pool_ = pool; }

private:
    MemoryPool* pool_ = nullptr;
};

template<typename Allocator>
class allocator_traits {
public:
    using allocator_type = Allocator;
    using value_type = void;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static constexpr bool propagate_on_container_copy_assignment = false;
    static constexpr bool propagate_on_container_move_assignment = true;
    static constexpr bool propagate_on_container_swap = true;

    static void* allocate(Allocator& a, size_t n) {
        return a.allocate(n);
    }

    static void deallocate(Allocator& a, void* p, size_t n) {
        a.deallocate(p, n);
    }
};

} // namespace cuda::memory
