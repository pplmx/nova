#include <gtest/gtest.h>
#include "cuda/memory/memory_pool.h"

class MemoryPoolStreamTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(MemoryPoolStreamTest, StreamIdTrackingInAllocations) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 4;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* ptr1 = pool.allocate(256, 0);
    void* ptr2 = pool.allocate(256, 1);
    void* ptr3 = pool.allocate(256, 0);

    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);

    auto by_stream = pool.get_allocations_by_stream();

    EXPECT_TRUE(by_stream.count(0) > 0);
    EXPECT_TRUE(by_stream.count(1) > 0);

    pool.deallocate(ptr1, 256);
    pool.deallocate(ptr2, 256);
    pool.deallocate(ptr3, 256);
}

TEST_F(MemoryPoolStreamTest, GetAllocationsByStreamReturnsCorrectMap) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    pool.allocate(128, 10);
    pool.allocate(128, 20);
    pool.allocate(128, 10);

    auto by_stream = pool.get_allocations_by_stream();

    EXPECT_EQ(by_stream.count(10), 1);
    EXPECT_EQ(by_stream.count(20), 1);
}

TEST_F(MemoryPoolStreamTest, SetThrowOnFailureFalseReturnsNullptr) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 512;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    pool.allocate(300);
    pool.allocate(300);

    pool.set_throw_on_failure(false);

    void* result = pool.allocate(1000);

    EXPECT_EQ(result, nullptr);
}

TEST_F(MemoryPoolStreamTest, SetThrowOnFailureTrueThrowsException) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 512;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    pool.allocate(300);
    pool.allocate(300);

    pool.set_throw_on_failure(true);

    EXPECT_THROW({
        pool.allocate(1000);
    }, std::exception);
}

TEST_F(MemoryPoolStreamTest, PeakAllocatedBytesTracked) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 4;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    pool.allocate(500);
    pool.allocate(300);

    auto metrics = pool.get_metrics();

    EXPECT_GE(metrics.peak_allocated_bytes, 800);
}

TEST_F(MemoryPoolStreamTest, NumActiveStreamsTracked) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 256;
    config.max_blocks = 3;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    pool.allocate(200, 1);
    pool.allocate(200, 2);
    pool.allocate(200, 3);

    auto metrics = pool.get_metrics();

    EXPECT_GE(metrics.num_active_streams, 1);
}

TEST_F(MemoryPoolStreamTest, DeallocateUpdatesStreamTracking) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* ptr1 = pool.allocate(256, 1);
    void* ptr2 = pool.allocate(256, 2);

    pool.deallocate(ptr1, 256);

    auto by_stream = pool.get_allocations_by_stream();

    EXPECT_EQ(by_stream.count(1), 0);
    EXPECT_GE(by_stream[2], 256);

    pool.deallocate(ptr2, 256);
}
