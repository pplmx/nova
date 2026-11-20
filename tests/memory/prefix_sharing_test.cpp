#include <gtest/gtest.h>
#include "cuda/memory/kv_cache_allocator.h"

namespace cuda::memory {

    class PrefixSharingTest : public ::testing::Test {
    protected:
        void SetUp() override {
            KVCacheAllocatorConfig config {
                .num_heads = 32,
                .head_dim = 128,
                .block_size_tokens = 16,
                .num_blocks = 256,
                .num_layers = 32,
                .enable_prefix_caching = true,
            };
            config_ = config;
            allocator = std::make_unique<KVCacheAllocator>(config);
        }

        std::unique_ptr<KVCacheAllocator> allocator;
        KVCacheAllocatorConfig config_;
    };

    TEST_F(PrefixSharingTest, ForkPrefixBlocks) {
        auto blocks1 = allocator->allocate(1, 32);
        ASSERT_GT(blocks1.size(), 0);

        EXPECT_EQ(blocks1[0]->ref_count, 1);
        EXPECT_TRUE(blocks1[0]->shared_by.empty());

        auto forked_blocks = allocator->fork_prefix_blocks(1, 2, 1);

        ASSERT_EQ(forked_blocks.size(), 1);
        EXPECT_EQ(forked_blocks[0]->ref_count, 2);
        EXPECT_EQ(forked_blocks[0]->block_id, blocks1[0]->block_id);
    }

    TEST_F(PrefixSharingTest, ReferenceCountDecrement) {
        auto blocks1 = allocator->allocate(1, 32);
        ASSERT_GT(blocks1.size(), 0);

        allocator->fork_prefix_blocks(1, 2, 1);
        EXPECT_EQ(blocks1[0]->ref_count, 2);

        allocator->merge_prefix_blocks(2);
        EXPECT_EQ(blocks1[0]->ref_count, 1);
    }

    TEST_F(PrefixSharingTest, FindSequencesWithPrefix) {
        auto blocks1 = allocator->allocate(1, 32);
        ASSERT_GT(blocks1.size(), 0);

        allocator->fork_prefix_blocks(1, 2, 1);

        auto sharing = allocator->find_sequences_with_prefix(1);
        EXPECT_EQ(sharing.size(), 1);
        EXPECT_EQ(sharing[0], 2);
    }

    // compute_content_hash reads num_tokens * num_heads * head_dim floats, because
    // num_tokens counts tokens and each token carries a full head_dim vector per head.
    // These two cases passed a flat 512 floats against a fixture configured for
    // 32 heads x 128 head_dim, i.e. 131072 floats for 32 tokens: a 256x over-read.
    // AddressSanitizer caught it as "READ of size 4 ... heap-buffer-overflow" inside
    // compute_content_hash. The buffers are now sized from the same formula the
    // function documents.
    TEST_F(PrefixSharingTest, ContentHashConsistency) {
        const auto& cfg = config_;
        std::vector<float> tokens(static_cast<size_t>(32) * cfg.num_heads * cfg.head_dim, 1.0f);

        uint64_t hash1 = allocator->compute_content_hash(tokens.data(), 32);
        uint64_t hash2 = allocator->compute_content_hash(tokens.data(), 32);

        EXPECT_EQ(hash1, hash2);
    }

    TEST_F(PrefixSharingTest, ContentHashDifferentForDifferentTokens) {
        const auto& cfg = config_;
        const size_t floats = static_cast<size_t>(32) * cfg.num_heads * cfg.head_dim;
        std::vector<float> tokens1(floats, 1.0f);
        std::vector<float> tokens2(floats, 2.0f);

        uint64_t hash1 = allocator->compute_content_hash(tokens1.data(), 32);
        uint64_t hash2 = allocator->compute_content_hash(tokens2.data(), 32);

        EXPECT_NE(hash1, hash2);
    }

}  // namespace cuda::memory
