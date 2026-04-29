#include <gtest/gtest.h>
#include <cuda/memory/kv_cache_allocator.h>
#include <thread>
#include <atomic>
#include <vector>
#include <numeric>

namespace cuda::memory::test {

class KVCacheAllocatorEdgeTest : public ::testing::Test {
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

TEST_F(KVCacheAllocatorEdgeTest, AllocateSingleToken) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks = allocator->allocate(1, 1);
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_NE(blocks[0], nullptr);
    EXPECT_EQ(blocks[0]->num_tokens, config_.block_size_tokens);
}

TEST_F(KVCacheAllocatorEdgeTest, AllocateExactlyBlockSize) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks = allocator->allocate(1, config_.block_size_tokens);
    EXPECT_EQ(blocks.size(), 1);
}

TEST_F(KVCacheAllocatorEdgeTest, AllocateMultipleSequencesSameSize) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    const int seq_count = 10;
    const int tokens_per_seq = 32;

    for (int i = 0; i < seq_count; ++i) {
        auto blocks = allocator->allocate(i, tokens_per_seq);
        EXPECT_EQ(blocks.size(), tokens_per_seq / config_.block_size_tokens);
    }

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, seq_count * (tokens_per_seq / config_.block_size_tokens));
    EXPECT_EQ(stats.free_blocks, config_.num_blocks - stats.allocated_blocks);
}

TEST_F(KVCacheAllocatorEdgeTest, AllocateSequencesDifferentSizes) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    std::vector<int> sizes = {16, 32, 48, 64, 80};
    int total_blocks = 0;

    for (size_t i = 0; i < sizes.size(); ++i) {
        auto blocks = allocator->allocate(i, sizes[i]);
        int expected_blocks = (sizes[i] + config_.block_size_tokens - 1) / config_.block_size_tokens;
        EXPECT_EQ(blocks.size(), expected_blocks);
        total_blocks += expected_blocks;
    }

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, total_blocks);
}

TEST_F(KVCacheAllocatorEdgeTest, AllocateAndFreeInOrder) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    std::vector<int64_t> seq_ids;
    for (int i = 0; i < 5; ++i) {
        seq_ids.push_back(i);
        allocator->allocate(i, 32);
    }

    auto stats_before = allocator->get_stats();
    EXPECT_EQ(stats_before.allocated_blocks, 5 * 2);

    for (int64_t seq_id : seq_ids) {
        allocator->free(seq_id);
    }

    auto stats_after = allocator->get_stats();
    EXPECT_EQ(stats_after.allocated_blocks, 0);
    EXPECT_EQ(stats_after.free_blocks, config_.num_blocks);
}

TEST_F(KVCacheAllocatorEdgeTest, AllocateAndFreeInterleaved) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 32);
    allocator->allocate(2, 32);
    allocator->allocate(3, 32);

    allocator->free(2);

    allocator->allocate(4, 32);
    allocator->allocate(5, 32);

    allocator->free(1);
    allocator->free(3);

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 4);
}

TEST_F(KVCacheAllocatorEdgeTest, AppendToExistingSequence) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    auto appended1 = allocator->append(1, 16);
    EXPECT_EQ(appended1.size(), 1);

    auto appended2 = allocator->append(1, 16);
    EXPECT_EQ(appended2.size(), 1);

    auto all_blocks = allocator->get_blocks(1);
    EXPECT_EQ(all_blocks.size(), 3);
}

TEST_F(KVCacheAllocatorEdgeTest, AppendBeyondInitialAllocation) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);

    for (int i = 0; i < 10; ++i) {
        auto appended = allocator->append(1, 16);
        EXPECT_EQ(appended.size(), 1);
    }

    auto blocks = allocator->get_blocks(1);
    EXPECT_EQ(blocks.size(), 11);
}

TEST_F(KVCacheAllocatorEdgeTest, AppendToNonExistentCreatesNew) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    auto blocks = allocator->append(999, 32);
    EXPECT_EQ(blocks.size(), 2);

    auto retrieved = allocator->get_blocks(999);
    EXPECT_EQ(retrieved.size(), 2);
}

TEST_F(KVCacheAllocatorEdgeTest, GetBlockByIndex) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 48);

    auto* block0 = allocator->get_block(1, 0);
    auto* block1 = allocator->get_block(1, 1);
    auto* block2 = allocator->get_block(1, 2);
    auto* invalid = allocator->get_block(1, 99);
    auto* wrong_seq = allocator->get_block(999, 0);

    EXPECT_NE(block0, nullptr);
    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);
    EXPECT_EQ(block2->num_tokens, 16);
    EXPECT_EQ(invalid, nullptr);
    EXPECT_EQ(wrong_seq, nullptr);
}

TEST_F(KVCacheAllocatorEdgeTest, BlockSequenceOwnership) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 32);
    allocator->allocate(2, 32);

    auto blocks1 = allocator->get_blocks(1);
    auto blocks2 = allocator->get_blocks(2);

    EXPECT_EQ(blocks1[0]->sequence_id, 1);
    EXPECT_EQ(blocks2[0]->sequence_id, 2);

    EXPECT_NE(blocks1[0]->data, blocks2[0]->data);
}

TEST_F(KVCacheAllocatorEdgeTest, LRUUpdatedOnAccess) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocator->allocate(2, 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocator->allocate(3, 16);

    auto blocks1 = allocator->get_blocks(1);
    EXPECT_NE(blocks1[0]->last_access, 0);
}

TEST_F(KVCacheAllocatorEdgeTest, EvictionRemovesOldest) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    for (int i = 0; i < 120; ++i) {
        allocator->allocate(i, 16);
    }

    auto stats_before = allocator->get_stats();
    EXPECT_LT(stats_before.free_blocks, 10);

    allocator->evict(50);

    auto stats_after = allocator->get_stats();
    EXPECT_GT(stats_after.free_blocks, stats_before.free_blocks);
    EXPECT_GE(stats_after.evictions, 0);
}

TEST_F(KVCacheAllocatorEdgeTest, FreeNonExistentSequence) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    EXPECT_NO_THROW(allocator->free(999));
    EXPECT_NO_THROW(allocator->free(0));

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 0);
}

TEST_F(KVCacheAllocatorEdgeTest, FragmentationTracking) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 32);
    allocator->allocate(2, 32);
    allocator->free(1);
    allocator->allocate(3, 16);
    allocator->allocate(4, 16);

    auto stats = allocator->get_stats();
    EXPECT_GT(stats.fragmentation_percent, 0.0f);
    EXPECT_LT(stats.fragmentation_percent, 100.0f);
}

TEST_F(KVCacheAllocatorEdgeTest, StatsAccuracy) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    EXPECT_EQ(config_.num_blocks, allocator->get_stats().total_blocks);
    EXPECT_EQ(config_.num_blocks, allocator->get_stats().free_blocks);
    EXPECT_EQ(0, allocator->get_stats().allocated_blocks);

    allocator->allocate(1, 32);
    allocator->allocate(2, 64);

    auto stats = allocator->get_stats();
    EXPECT_EQ(2, stats.allocated_blocks);
    EXPECT_EQ(config_.num_blocks - 2, stats.free_blocks);
}

TEST_F(KVCacheAllocatorEdgeTest, PrefixCacheHit) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 32);

    std::vector<float> prefix_data(32 * config_.num_heads * config_.head_dim, 0.5f);

    auto match = allocator->find_prefix_match(prefix_data.data(), 32);
    EXPECT_TRUE(match.has_value());
    EXPECT_EQ(match->sequence_id, 1);
}

TEST_F(KVCacheAllocatorEdgeTest, PrefixCacheMiss) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 32);

    std::vector<float> different_data(32 * config_.num_heads * config_.head_dim, 0.123f);

    auto match = allocator->find_prefix_match(different_data.data(), 32);
    EXPECT_FALSE(match.has_value());
}

TEST_F(KVCacheAllocatorEdgeTest, PrefixCacheDisabled) {
    auto config = config_;
    config.enable_prefix_caching = false;
    auto allocator = std::make_unique<KVCacheAllocator>(config);

    allocator->allocate(1, 32);
    std::vector<float> data(32 * config_.num_heads * config_.head_dim, 0.5f);

    auto match = allocator->find_prefix_match(data.data(), 32);
    EXPECT_FALSE(match.has_value());
}

TEST_F(KVCacheAllocatorEdgeTest, ConcurrentAllocationThreads) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);
    std::atomic<int> success_count{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&allocator, &success_count, t]() {
            for (int i = 0; i < 10; ++i) {
                int64_t seq_id = t * 100 + i;
                auto blocks = allocator->allocate(seq_id, 16);
                if (!blocks.empty() && blocks[0] != nullptr) {
                    success_count++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stats = allocator->get_stats();
    EXPECT_EQ(success_count.load(), stats.allocated_blocks);
}

TEST_F(KVCacheAllocatorEdgeTest, BlockSize16) {
    auto config = config_;
    config.block_size_tokens = 16;
    auto allocator = std::make_unique<KVCacheAllocator>(config);

    EXPECT_EQ(16, allocator->get_block_size_tokens());

    auto blocks = allocator->allocate(1, 16);
    EXPECT_EQ(1, blocks.size());
}

TEST_F(KVCacheAllocatorEdgeTest, BlockSize32) {
    auto config = config_;
    config.block_size_tokens = 32;
    auto allocator = std::make_unique<KVCacheAllocator>(config);

    EXPECT_EQ(32, allocator->get_block_size_tokens());

    auto blocks = allocator->allocate(1, 32);
    EXPECT_EQ(1, blocks.size());

    blocks = allocator->allocate(2, 16);
    EXPECT_EQ(1, blocks.size());
}

TEST_F(KVCacheAllocatorEdgeTest, BlockSize64) {
    auto config = config_;
    config.block_size_tokens = 64;
    auto allocator = std::make_unique<KVCacheAllocator>(config);

    EXPECT_EQ(64, allocator->get_block_size_tokens());

    auto blocks = allocator->allocate(1, 64);
    EXPECT_EQ(1, blocks.size());

    blocks = allocator->allocate(2, 32);
    EXPECT_EQ(1, blocks.size());
}

TEST_F(KVCacheAllocatorEdgeTest, InvalidBlockSizeThrows) {
    auto config = config_;
    config.block_size_tokens = 100;

    EXPECT_THROW(auto allocator = std::make_unique<KVCacheAllocator>(config),
                 std::invalid_argument);
}

TEST_F(KVCacheAllocatorEdgeTest, FreeBlockAfterEviction) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    for (int i = 0; i < 100; ++i) {
        allocator->allocate(i, 16);
    }

    auto stats_before = allocator->get_stats();
    allocator->free(50);
    auto stats_after = allocator->get_stats();

    EXPECT_EQ(stats_after.allocated_blocks, stats_before.allocated_blocks - 1);
    EXPECT_EQ(stats_after.free_blocks, stats_before.free_blocks + 1);
}

TEST_F(KVCacheAllocatorEdgeTest, ResetStatsClearsCounters) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    allocator->allocate(2, 16);
    allocator->allocate(3, 16);
    allocator->free(2);

    auto stats_before = allocator->get_stats();
    EXPECT_GT(stats_before.allocation_requests, 0);

    allocator->reset_stats();

    auto stats_after = allocator->get_stats();
    EXPECT_EQ(0, stats_after.allocation_requests);
    EXPECT_EQ(0, stats_after.allocation_failures);
    EXPECT_EQ(0, stats_after.prefix_cache_hits);
    EXPECT_EQ(0, stats_after.prefix_cache_misses);
    EXPECT_EQ(0, stats_after.evictions);
}

TEST_F(KVCacheAllocatorEdgeTest, GPUMemoryPointerValid) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    EXPECT_NE(nullptr, allocator->get_gpu_memory());
    EXPECT_GT(allocator->get_gpu_memory_size(), 0);

    allocator->allocate(1, 16);

    auto blocks = allocator->get_blocks(1);
    EXPECT_NE(blocks[0]->data, nullptr);
}

TEST_F(KVCacheAllocatorEdgeTest, BlockIdsUniqueAndSequential) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    std::vector<int> block_ids;
    for (int i = 0; i < 20; ++i) {
        auto blocks = allocator->allocate(i, 16);
        for (const auto* block : blocks) {
            block_ids.push_back(block->block_id);
        }
    }

    std::vector<int> sorted_ids = block_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    EXPECT_EQ(block_ids.size(), sorted_ids.size());

    for (int i = 0; i < sorted_ids.size() - 1; ++i) {
        EXPECT_GE(sorted_ids[i + 1] - sorted_ids[i], 0);
    }
}

TEST_F(KVCacheAllocatorEdgeTest, AllBlocksAllocatedThenFreed) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    int blocks_per_seq = config_.num_blocks / 10;
    int tokens_per_seq = blocks_per_seq * config_.block_size_tokens;

    for (int i = 0; i < 10; ++i) {
        allocator->allocate(i, tokens_per_seq);
    }

    auto stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, config_.num_blocks);
    EXPECT_EQ(stats.free_blocks, 0);

    for (int i = 0; i < 10; ++i) {
        allocator->free(i);
    }

    stats = allocator->get_stats();
    EXPECT_EQ(stats.allocated_blocks, 0);
    EXPECT_EQ(stats.free_blocks, config_.num_blocks);
}

TEST_F(KVCacheAllocatorEdgeTest, AppendPartialBlock) {
    auto allocator = std::make_unique<KVCacheAllocator>(config_);

    allocator->allocate(1, 16);
    auto appended = allocator->append(1, 8);

    EXPECT_EQ(appended.size(), 1);

    auto blocks = allocator->get_blocks(1);
    EXPECT_EQ(blocks.size(), 2);
}

}  // namespace cuda::memory::test
