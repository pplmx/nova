#include <gtest/gtest.h>
#include "cuda/memory/memory_pool.h"

class MemoryPoolMetricsTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
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

// The destructor is implicitly noexcept, and clear() throws when cudaFree fails. A cudaFree can
// fail -- measured, freeing a block allocated before cudaDeviceReset() returns "invalid argument"
// -- so the old destructor could reach std::terminate during teardown. Destroying a pool whose
// blocks belong to a torn-down context must not kill the process.
TEST_F(MemoryPoolMetricsTest, DestructorDoesNotThrowWhenTheContextIsGone) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = false;

    auto* pool = new cuda::memory::MemoryPool(config);
    pool->allocate(1024);  // now owns a real block

    // Destroy the context the block belongs to, so the pool's cudaFree is bound to fail.
    (void)cudaDeviceReset();

    EXPECT_NO_THROW(delete pool) << "teardown must not throw out of a noexcept destructor";

    // Leave the process in a state the remaining cases can use.
    (void)cudaFree(nullptr);
}

// synchronize_stream(stream_id) ignores its argument: the pool records the integer a caller passed
// to allocate() but never holds the cudaStream_t it names, so there is no handle to synchronise.
// The body has always synchronised stream 0. This case pins that behaviour so a future change to it
// is deliberate, and states the thing a caller must not assume.
TEST_F(MemoryPoolMetricsTest, SynchronizeStreamIgnoresItsArgument) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = false;

    cuda::memory::MemoryPool pool(config);
    auto* p = pool.allocate(1024, /*stream_id=*/7);
    ASSERT_NE(p, nullptr);

    // Whatever id is passed, the call synchronises the device and returns; it must not throw and
    // must not depend on the id naming anything real.
    EXPECT_NO_THROW(pool.synchronize_stream(7));
    EXPECT_NO_THROW(pool.synchronize_stream(0));
    EXPECT_NO_THROW(pool.synchronize_stream(-1));

    // The id IS recorded, which is what makes the ignored argument confusing in the first place.
    const auto by_stream = pool.get_allocations_by_stream();
    ASSERT_TRUE(by_stream.count(7)) << "the allocation was made on stream 7 and must be recorded there";
    EXPECT_GT(by_stream.at(7), 0u);

    pool.deallocate(p, 1024);
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

// Preallocating is setup, not a request. The counters used to start at max_blocks because the
// constructor incremented misses_ for each block it created, so a caller reading get_metrics() before
// asking for anything saw a non-zero miss count, and the first real request did not change it.
TEST_F(MemoryPoolMetricsTest, PreallocationIsNotCountedAsAMiss) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    const auto before = pool.get_metrics();
    EXPECT_EQ(before.misses, 0) << "no request has been made; filling the pool is not a miss";
    EXPECT_EQ(before.hits, 0);

    auto* p = pool.allocate(1024);
    ASSERT_NE(p, nullptr);

    const auto after = pool.get_metrics();
    EXPECT_EQ(after.misses, 0) << "the request was served from a preallocated block";
    EXPECT_EQ(after.hits, 1) << "and it counts as a hit";

    pool.deallocate(p, 1024);
}

TEST_F(MemoryPoolMetricsTest, MultipleAllocationsFromSameBlock) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 1;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    // This used to expect 1: preallocate=true created the block in the constructor and counted that
    // as a miss. A miss is a request that found no block with room, and no request has been made yet,
    // so the counter starts at zero. The old expectation described the setup cost as a request
    // outcome.
    auto metrics_before = pool.get_metrics();
    EXPECT_EQ(metrics_before.misses, 0) << "nothing has been requested yet";
    EXPECT_EQ(metrics_before.hits, 0);

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
    // Without this the case cannot tell "clear() reset the counters" from "they were
    // never set": the value was collected and then discarded, which the compiler reports
    // as -Wunused-but-set-variable.
    EXPECT_GT(metrics_before_clear.misses, 0u) << "a cold pool must record at least one miss before clear() is meaningful";

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
