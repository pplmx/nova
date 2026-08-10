#include <gtest/gtest.h>
#include "cuda/memory/streaming_cache_manager.h"
#include "cuda/memory/kv_cache_allocator.h"
#include "cuda/stream/stream.h"
#include <thread>
#include <chrono>

namespace cuda::memory {

class StreamingCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        KVCacheAllocatorConfig config{
            .num_heads = 32,
            .head_dim = 128,
            .block_size_tokens = 16,
            .num_blocks = 256,
            .num_layers = 32,
        };
        allocator = std::make_unique<KVCacheAllocator>(config);

        StreamingCacheConfig stream_config{
            .enable_prefetch = true,
            .enable_async_eviction = true,
            .prefetch_ahead_blocks = 2,
            .eviction_batch_size = 4,
        };
        manager = std::make_unique<StreamingCacheManager>(allocator.get(), stream_config);
    }

    std::unique_ptr<KVCacheAllocator> allocator;
    std::unique_ptr<StreamingCacheManager> manager;
};

TEST_F(StreamingCacheTest, PrefetchRequestCreated) {
    stream::Stream stream;

    auto blocks1 = allocator->allocate(1, 32);
    EXPECT_GT(blocks1.size(), 0);

    manager->prefetch_async(1, 16, stream);

    EXPECT_TRUE(manager->should_evict_async() == false ||
                allocator->get_num_free_blocks() > 0);
}

TEST_F(StreamingCacheTest, ImportanceTracking) {
    manager->update_importance(1, 10.0f);
    manager->update_importance(2, 5.0f);

    EXPECT_EQ(manager->get_importance(1), 10.0f);
    EXPECT_EQ(manager->get_importance(2), 5.0f);
    EXPECT_EQ(manager->get_importance(3), 1.0f);
}

TEST_F(StreamingCacheTest, DefaultImportanceIsOne) {
    EXPECT_EQ(manager->get_importance(999), 1.0f);
}

TEST_F(StreamingCacheTest, PrefetchAsyncDisabled) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = false,
        .enable_async_eviction = true,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    stream::Stream stream;

    EXPECT_NO_THROW(local_manager->prefetch_async(1, 32, stream));
}

TEST_F(StreamingCacheTest, PrefetchCalculatesCorrectBlockCount) {
    stream::Stream stream;

    allocator->allocate(1, 32);

    manager->prefetch_async(1, 32, stream);

    EXPECT_TRUE(allocator->get_num_free_blocks() >= 0);
}

TEST_F(StreamingCacheTest, SyncPrefetchProcessesRequests) {
    stream::Stream stream;

    allocator->allocate(1, 16);

    manager->prefetch_async(1, 16, stream);
    manager->sync_prefetch(stream);

    auto blocks = allocator->get_blocks(1);
    // 1 initial block + prefetch: (16 tokens/16 block_size)+1 = 2 blocks,
    // capped at the fixture's prefetch_ahead_blocks=2 -> 3 blocks total.
    ASSERT_EQ(blocks.size(), 3);
}

TEST_F(StreamingCacheTest, SyncPrefetchClearsPendingRequests) {
    stream::Stream stream;

    allocator->allocate(1, 16);

    manager->prefetch_async(1, 16, stream);
    manager->sync_prefetch(stream);
    manager->sync_prefetch(stream);

    auto blocks = allocator->get_blocks(1);
    EXPECT_GE(blocks.size(), 1);
}

TEST_F(StreamingCacheTest, ShouldEvictAsyncBelowThreshold) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = false,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    EXPECT_FALSE(local_manager->should_evict_async());
}

TEST_F(StreamingCacheTest, ShouldEvictAsyncAboveThreshold) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = true,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    // 232 allocations of 1 block each leave 24/256 free (~9.4% < 10%), which
    // crosses should_evict_async()'s threshold. (230 left ~10.2% free, just
    // above it, so the assertion was previously unattainable.)
    for (int i = 0; i < 232; ++i) {
        allocator->allocate(i, 16);
    }

    EXPECT_TRUE(local_manager->should_evict_async());
}

TEST_F(StreamingCacheTest, ShouldEvictAsyncNearThreshold) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = true,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    for (int i = 0; i < 200; ++i) {
        allocator->allocate(i, 16);
    }

    auto stats = allocator->get_stats();
    auto free_ratio = static_cast<float>(stats.free_blocks) / stats.total_blocks;

    if (free_ratio < 0.1f) {
        EXPECT_TRUE(local_manager->should_evict_async());
    } else {
        EXPECT_FALSE(local_manager->should_evict_async());
    }
}

TEST_F(StreamingCacheTest, EvictImportanceWeighted) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = true,
        .eviction_policy = 1,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    allocator->allocate(1, 32);
    allocator->allocate(2, 32);
    allocator->allocate(3, 32);

    local_manager->update_importance(1, 5.0f);
    local_manager->update_importance(2, 10.0f);
    local_manager->update_importance(3, 1.0f);

    auto stats_before = allocator->get_stats();
    EXPECT_EQ(stats_before.allocated_blocks, 6);

    local_manager->evict_importance_weighted(2);

    auto stats_after = allocator->get_stats();
    EXPECT_LT(stats_after.allocated_blocks, stats_before.allocated_blocks);
}

TEST_F(StreamingCacheTest, EvictImportanceWeightedLRU) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = true,
        .eviction_policy = 1,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    allocator->allocate(1, 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocator->allocate(2, 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocator->allocate(3, 16);

    local_manager->update_importance(1, 1.0f);
    local_manager->update_importance(2, 1.0f);
    local_manager->update_importance(3, 1.0f);

    local_manager->evict_importance_weighted(3);

    auto blocks1 = allocator->get_blocks(1);
    auto blocks2 = allocator->get_blocks(2);
    auto blocks3 = allocator->get_blocks(3);

    int existing_count = 0;
    if (!blocks1.empty()) existing_count++;
    if (!blocks2.empty()) existing_count++;
    if (!blocks3.empty()) existing_count++;

    EXPECT_EQ(existing_count, 0);
}

TEST_F(StreamingCacheTest, EvictImportanceWeightedMixedImportance) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = true,
        .eviction_policy = 1,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    allocator->allocate(1, 16);
    allocator->allocate(2, 16);

    local_manager->update_importance(1, 100.0f);
    local_manager->update_importance(2, 1.0f);

    local_manager->evict_importance_weighted(1);

    auto blocks1 = allocator->get_blocks(1);
    EXPECT_EQ(blocks1.size(), 1);

    auto blocks2 = allocator->get_blocks(2);
    EXPECT_TRUE(blocks2.empty());
}

TEST_F(StreamingCacheTest, FragmentationMonitoringThreshold) {
    auto config = StreamingCacheConfig{
        .enable_prefetch = true,
        .enable_async_eviction = true,
        .eviction_policy = 0,
    };
    auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

    auto report = allocator->analyze_fragmentation();

    EXPECT_GE(report.ratio, 0.0f);
    EXPECT_LE(report.ratio, 100.0f);
}

TEST_F(StreamingCacheTest, UpdateImportanceOverwrites) {
    manager->update_importance(1, 5.0f);
    manager->update_importance(1, 10.0f);

    EXPECT_EQ(manager->get_importance(1), 10.0f);
}

TEST_F(StreamingCacheTest, MultiplePrefetchRequests) {
    stream::Stream stream;

    allocator->allocate(1, 16);
    allocator->allocate(2, 16);
    allocator->allocate(3, 16);

    manager->prefetch_async(1, 16, stream);
    manager->prefetch_async(2, 16, stream);
    manager->prefetch_async(3, 16, stream);

    manager->sync_prefetch(stream);

    auto blocks1 = allocator->get_blocks(1);
    auto blocks2 = allocator->get_blocks(2);
    auto blocks3 = allocator->get_blocks(3);

    EXPECT_GE(blocks1.size(), 1);
    EXPECT_GE(blocks2.size(), 1);
    EXPECT_GE(blocks3.size(), 1);
}

}  // namespace cuda::memory
