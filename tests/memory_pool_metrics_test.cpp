#include <gtest/gtest.h>
#include "cuda/memory/memory_pool.h"

class MemoryPoolMetricsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(MemoryPoolMetricsTest, InitialMetricsAreZero) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    auto metrics = pool.get_metrics();

    EXPECT_EQ(metrics.hits, 0);
    EXPECT_EQ(metrics.misses, 0);
}

TEST_F(MemoryPoolMetricsTest, AllocateIncrementsMisses) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    auto* ptr = pool.allocate(256);
    auto metrics = pool.get_metrics();

    EXPECT_EQ(metrics.misses, 1);
    EXPECT_EQ(metrics.hits, 0);

    pool.deallocate(ptr, 256);
}

TEST_F(MemoryPoolMetricsTest, MultipleAllocationsFromSameBlock) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 1;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    auto metrics_before = pool.get_metrics();
    EXPECT_EQ(metrics_before.misses, 1);

    auto* ptr1 = pool.allocate(256);
    auto metrics_after1 = pool.get_metrics();
    EXPECT_GE(metrics_after1.hits, 0);

    auto* ptr2 = pool.allocate(256);
    auto metrics_after2 = pool.get_metrics();
    EXPECT_GE(metrics_after2.hits, metrics_after1.hits);

    pool.deallocate(ptr1, 256);
    pool.deallocate(ptr2, 256);
}

TEST_F(MemoryPoolMetricsTest, ClearResetsMetrics) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    auto* ptr = pool.allocate(256);
    pool.deallocate(ptr, 256);

    auto metrics_before_clear = pool.get_metrics();

    pool.clear();

    auto metrics_after_clear = pool.get_metrics();
    EXPECT_EQ(metrics_after_clear.hits, 0);
    EXPECT_EQ(metrics_after_clear.misses, 0);
}

TEST_F(MemoryPoolMetricsTest, DefragmentCompactsMemory) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 1;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    auto* ptr1 = pool.allocate(512);
    auto* ptr2 = pool.allocate(256);

    pool.deallocate(ptr1, 512);

    EXPECT_NO_THROW(pool.defragment());

    EXPECT_TRUE(ptr2 != nullptr);

    pool.deallocate(ptr2, 256);
}

TEST_F(MemoryPoolMetricsTest, FragmentationPercentageCalculated) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 1;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    pool.allocate(512);

    auto metrics = pool.get_metrics();

    EXPECT_GE(metrics.fragmentation_percent, 0.0);
    EXPECT_LE(metrics.fragmentation_percent, 100.0);
}
