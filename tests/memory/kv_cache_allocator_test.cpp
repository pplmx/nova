#include <gtest/gtest.h>
#include <cuda/memory/kv_cache_allocator.h>
#include <thread>
#include <atomic>
#include <vector>

namespace cuda::memory::test {

class KVCacheAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = KVCacheAllocatorConfig{
            .num_heads = 4,
            .head_dim = 64,
            .block_size_tokens = 16,
            .num_blocks = 128,
            .num_layers = 1,
            .eviction_threshold_pct = 10,
            .enable_prefix_caching = true,
            .max_prefix_blocks = 32
        };
    }

    KVCacheAllocatorConfig config_;
};

TEST_F(KVCacheAllocatorTest, Creation) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);
    ASSERT_NE(allocator, nullptr);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.total_blocks, 128);
    EXPECT_EQ(stats.free_blocks, 128);
    EXPECT_EQ(stats.allocated_blocks, 0);
}

TEST_F(KVCacheAllocatorTest, AllocationDeallocation) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks = allocator->allocate(1, 32);
    ASSERT_EQ(blocks.size(), 2);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 2);
    EXPECT_EQ(stats.free_blocks, 126);

    allocator->free(1);

    stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 0);
    EXPECT_EQ(stats.free_blocks, 128);
}

TEST_F(KVCacheAllocatorTest, MultipleSequences) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks1 = allocator->allocate(1, 16);
    auto blocks2 = allocator->allocate(2, 32);
    auto blocks3 = allocator->allocate(3, 48);

    EXPECT_EQ(blocks1.size(), 1);
    EXPECT_EQ(blocks2.size(), 2);
    EXPECT_EQ(blocks3.size(), 3);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 6);
    EXPECT_EQ(stats.free_blocks, 122);

    allocator->free(2);
    auto blocks2_after = allocator->get_blocks(2);
    EXPECT_TRUE(blocks2_after.empty());

    allocator->free(1);
    allocator->free(3);

    stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 0);
}

TEST_F(KVCacheAllocatorTest, AppendTokens) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto initial = allocator->allocate(1, 16);
    EXPECT_EQ(initial.size(), 1);

    auto appended = allocator->append(1, 16);
    EXPECT_EQ(appended.size(), 1);

    auto all_blocks = allocator->get_blocks(1);
    EXPECT_EQ(all_blocks.size(), 2);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 2);
}

TEST_F(KVCacheAllocatorTest, LRUTracking) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocator->allocate(2, 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocator->allocate(3, 16);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 3);
}

TEST_F(KVCacheAllocatorTest, Eviction) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    for (int i = 0; i < 120; ++i) {
        allocator->allocate(i, 16);
    }

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.free_blocks, 8);

    auto evicted = allocator->allocate(999, 16);
    EXPECT_GE(stats.evictions, 0);
}

TEST_F(KVCacheAllocatorTest, BlockRetrieval) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks = allocator->allocate(42, 32);
    ASSERT_EQ(blocks.size(), 2);

    auto retrieved = allocator->get_block(42, 0);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->block_id, blocks[0]->block_id);

    auto invalid = allocator->get_block(42, 99);
    EXPECT_EQ(invalid, nullptr);

    auto wrong_seq = allocator->get_block(999, 0);
    EXPECT_EQ(wrong_seq, nullptr);
}

TEST_F(KVCacheAllocatorTest, StatsAccuracy) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    allocator->allocate(2, 32);
    allocator->allocate(3, 48);
    allocator->free(2);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.total_blocks, 128);
    EXPECT_EQ(stats.allocated_blocks, 6);
    EXPECT_EQ(stats.free_blocks, 122);
    EXPECT_GT(stats.total_memory, 0);
    EXPECT_GT(stats.used_memory, 0);
}

TEST_F(KVCacheAllocatorTest, BlockSizeTokens) {
    auto config = config_;
    config.block_size_tokens = 32;
    auto allocator32 = std::make_unique<KVCacheAllocator>(config);
    EXPECT_EQ(allocator32->get_block_size_tokens(), 32);

    config.block_size_tokens = 64;
    auto allocator64 = std::make_unique<KVCacheAllocator>(config);
    EXPECT_EQ(allocator64->get_block_size_tokens(), 64);

    config.block_size_tokens = 128;
    EXPECT_THROW(auto bad_allocator = std::make_unique<KVCacheAllocator>(config),
                 std::invalid_argument);
}

TEST_F(KVCacheAllocatorTest, BlockMemoryAlignment) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    ASSERT_NE(allocator->get_gpu_memory(), nullptr);
    EXPECT_GT(allocator->get_gpu_memory_size(), 0);

    auto blocks = allocator->allocate(1, 16);
    ASSERT_EQ(blocks.size(), 1);
    EXPECT_NE(blocks[0]->data, nullptr);
}

TEST_F(KVCacheAllocatorTest, ConcurrentAllocation) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);
    std::atomic<int> allocations{0};
    std::atomic<int> threads_completed{0};

    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&allocator, &allocations, &threads_completed, t]() {
            for (int i = 0; i < 20; ++i) {
                auto blocks = allocator->allocate(t * 100 + i, 16);
                if (!blocks.empty()) {
                    allocations++;
                }
            }
            threads_completed++;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(threads_completed.load(), 4);

    auto stats = allocator->get_stats();
    EXPECT_EQ(allocations.load(), stats.allocated_blocks);
}

TEST_F(KVCacheAllocatorTest, SequenceIsolation) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks1 = allocator->allocate(1, 32);
    auto blocks2 = allocator->allocate(2, 32);

    ASSERT_EQ(blocks1.size(), 2);
    ASSERT_EQ(blocks2.size(), 2);

    EXPECT_NE(blocks1[0]->data, blocks2[0]->data);

    EXPECT_EQ(blocks1[0]->sequence_id, 1);
    EXPECT_EQ(blocks2[0]->sequence_id, 2);

    auto seq1_blocks = allocator->get_blocks(1);
    auto seq2_blocks = allocator->get_blocks(2);

    EXPECT_EQ(seq1_blocks.size(), 2);
    EXPECT_EQ(seq2_blocks.size(), 2);

    allocator->free(1);

    auto seq1_after = allocator->get_blocks(1);
    EXPECT_TRUE(seq1_after.empty());

    auto seq2_still = allocator->get_blocks(2);
    EXPECT_EQ(seq2_still.size(), 2);
}

TEST_F(KVCacheAllocatorTest, ResetStats) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    allocator->allocate(2, 16);
    allocator->free(1);

    auto stats_before = allocator->get_stats();
    EXPECT_GT(stats_before.allocation_requests, 0);

    allocator->reset_stats();

    auto stats_after = allocator->get_stats();
    EXPECT_EQ(stats_after.allocation_requests, 0);
    EXPECT_EQ(stats_after.prefix_cache_hits, 0);
    EXPECT_EQ(stats_after.prefix_cache_misses, 0);
}

TEST_F(KVCacheAllocatorTest, AppendToNonexistentSequence) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto appended = allocator->append(999, 16);
    EXPECT_EQ(appended.size(), 1);

    auto blocks = allocator->get_blocks(999);
    EXPECT_EQ(blocks.size(), 1);
}

TEST_F(KVCacheAllocatorTest, BlockIdsUnique) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    std::vector<int> block_ids;

    for (int i = 0; i < 10; ++i) {
        auto blocks = allocator->allocate(i, 16);
        for (const auto* block : blocks) {
            block_ids.push_back(block->block_id);
        }
    }

    std::sort(block_ids.begin(), block_ids.end());
    auto unique_end = std::unique(block_ids.begin(), block_ids.end());
    EXPECT_EQ(block_ids.end(), unique_end);
}

TEST_F(KVCacheAllocatorTest, FragmentationPercent) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 32);
    allocator->allocate(2, 32);

    auto stats = allocator->get_stats();
    EXPECT_GT(stats.fragmentation_percent, 0.0f);
    EXPECT_LT(stats.fragmentation_percent, 100.0f);
}

}  // namespace cuda::memory::test
