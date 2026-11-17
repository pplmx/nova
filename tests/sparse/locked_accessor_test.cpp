/**
 * @file locked_accessor_test.cpp
 * @brief Accessors that read state another thread mutates must take the class's lock
 *
 * Three classes held a mutex and used it on every mutation path, but had accessors that
 * read the same state without it:
 *
 *   KVCacheAllocator::get_num_free_blocks()  read free_list_.size() unlocked, while
 *                                            allocate() takes the mutex before
 *                                            push_back/pop_back on it, and get_stats()
 *                                            in the same class takes the lock.
 *   MemoryPool::num_allocations()            read allocation_map_.size() unlocked, while
 *                                            allocate()/deallocate() write it under the
 *                                            mutex and get_metrics() takes a lock.
 *   NcclContext::initialized(), has_nccl()   read initialized_ unlocked, while
 *                                            initialize() writes it under init_mutex_.
 *                                            broken_, declared beside it, is atomic —
 *                                            so the problem was understood for one member
 *                                            and not the other.
 *
 * A mutex on the writer side does not protect a reader: reading a vector's size while
 * another thread pushes to it is undefined behaviour, not merely a stale value.
 *
 * DistributedMemoryPool::device_count() and is_single_gpu() were reviewed and left alone:
 * pools_ is filled in the constructor and never mutated afterwards, so unlocked reads are
 * correct there.
 *
 * These are thread-safety properties. The cases below verify that the accessors still
 * return correct values with the lock in place; they cannot prove the absence of a race —
 * that needs a sanitizer. ThreadSanitizer is not usable against these kernels, so the fix
 * rests on the code comparison above rather than on a detection result.
 */

#include <cuda/memory/kv_cache_allocator.h>
#include <cuda/memory/memory_pool.h>
#include <cuda/nccl/nccl_context.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <thread>
#include <vector>

namespace locked_accessor_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

TEST(LockedAccessorTest, MemoryPoolAllocationCountFollowsAllocateAndRelease) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_allocations(), 0u);

    void* first = pool.allocate(64);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(pool.num_allocations(), 1u);

    void* second = pool.allocate(64);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(pool.num_allocations(), 2u);

    pool.deallocate(first, 64);
    EXPECT_EQ(pool.num_allocations(), 1u);

    pool.deallocate(second, 64);
    EXPECT_EQ(pool.num_allocations(), 0u);
}

TEST(LockedAccessorTest, MemoryPoolCountIsReadableWhileAnotherThreadAllocates) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The property that matters: the read is safe while a writer runs. Without the lock
    // this is a data race; with it the value is one of the states the pool really had.
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1 << 20;  // 1 MiB, so a few allocations fit
    config.max_blocks = 16;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);

    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::atomic<size_t> samples{0};

    std::thread reader([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!stop.load(std::memory_order_acquire)) {
            const size_t live = pool.num_allocations();
            EXPECT_LE(live, 16u) << "the count left the range the pool can reach";
            samples.fetch_add(1, std::memory_order_relaxed);
        }
    });

    start.store(true, std::memory_order_release);
    // The writer has to run long enough for the reader to interleave. My first version
    // did eight allocations and finished before the reader was scheduled, so the reader
    // sampled nothing and the assertion failed on samples == 0 — a test defect, not a
    // finding about the pool. Repeating the allocate/release cycle gives the reader
    // something to observe.
    while (samples.load(std::memory_order_relaxed) == 0) {
        std::this_thread::yield();
    }

    for (int round = 0; round < 200; ++round) {
        std::vector<void*> held;
        for (int i = 0; i < 4; ++i) {
            void* ptr = pool.allocate(1024);
            if (ptr != nullptr) {
                held.push_back(ptr);
            }
        }
        for (void* ptr : held) {
            pool.deallocate(ptr, 1024);
        }
    }
    stop.store(true, std::memory_order_release);
    reader.join();

    EXPECT_GT(samples.load(), 0u) << "the reader never observed a value";
    EXPECT_EQ(pool.num_allocations(), 0u);
}

TEST(LockedAccessorTest, KvCacheFreeBlockCountMatchesAllocation) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::memory::KVCacheAllocatorConfig config;
    config.num_blocks = 64;
    config.block_size_tokens = 16;

    cuda::memory::KVCacheAllocator allocator(config);
    const int before = allocator.get_num_free_blocks();
    EXPECT_GT(before, 0);

    std::vector<cuda::memory::KVCacheBlock*> blocks = allocator.allocate(1, 64);
    EXPECT_FALSE(blocks.empty()) << "allocating 64 tokens at 16 per block should succeed";
    EXPECT_LT(allocator.get_num_free_blocks(), before)
        << "handing out blocks must reduce the free count";
}

TEST(LockedAccessorTest, NcclContextReportsUninitialisedByDefault) {
    // A fresh context has not been initialised, and both accessors must agree with the
    // flag they read.
    cuda::nccl::NcclContext& context = cuda::nccl::NcclContext::instance();
    const bool initialised = context.initialized();
    const bool usable = context.has_nccl();

    if (!initialised) {
        EXPECT_FALSE(usable) << "an uninitialised context cannot be usable";
    }
}

}  // namespace locked_accessor_test
