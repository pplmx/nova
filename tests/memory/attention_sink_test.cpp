#include <gtest/gtest.h>
#include "cuda/memory/kv_cache_allocator.h"

namespace cuda::memory {

    class AttentionSinkTest : public ::testing::Test {
    protected:
        void SetUp() override {
            KVCacheAllocatorConfig config{
                .num_heads = 32,
                .head_dim = 128,
                .block_size_tokens = 16,
                .num_blocks = 256,
                .num_layers = 32,
                .enable_sink_separation = true,
                .num_sink_positions = 4,
                .sink_eviction_bonus = 1000,
            };
            allocator = std::make_unique<KVCacheAllocator>(config);
        }

        std::unique_ptr<KVCacheAllocator> allocator;
    };

    TEST_F(AttentionSinkTest, PromoteToSink) {
        auto blocks = allocator->allocate(1, 64);
        ASSERT_GT(blocks.size(), 0);

        EXPECT_FALSE(blocks[0]->is_attention_sink);

        allocator->promote_to_sink(blocks[0]->block_id, 0);

        EXPECT_TRUE(blocks[0]->is_attention_sink);
        EXPECT_EQ(blocks[0]->sink_position, 0);
    }

    TEST_F(AttentionSinkTest, SinkBlocksTrackedSeparately) {
        auto blocks = allocator->allocate(1, 64);
        ASSERT_GT(blocks.size(), 1);

        allocator->promote_to_sink(blocks[0]->block_id, 0);
        allocator->promote_to_sink(blocks[1]->block_id, 1);

        EXPECT_EQ(allocator->get_sink_blocks().size(), 2);
    }

    TEST_F(AttentionSinkTest, IsSinkBlock) {
        auto blocks = allocator->allocate(1, 32);
        ASSERT_GT(blocks.size(), 0);

        EXPECT_FALSE(allocator->is_sink_block(blocks[0]->block_id));

        allocator->promote_to_sink(blocks[0]->block_id, 0);

        EXPECT_TRUE(allocator->is_sink_block(blocks[0]->block_id));
    }

    TEST_F(AttentionSinkTest, DemoteFromSink) {
        auto blocks = allocator->allocate(1, 32);
        ASSERT_GT(blocks.size(), 0);

        allocator->promote_to_sink(blocks[0]->block_id, 0);
        EXPECT_EQ(allocator->get_sink_blocks().size(), 1);

        allocator->demote_from_sink(blocks[0]->block_id);
        EXPECT_EQ(allocator->get_sink_blocks().size(), 0);
        EXPECT_FALSE(blocks[0]->is_attention_sink);
    }

}  // namespace cuda::memory
