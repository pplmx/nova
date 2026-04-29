#include <gtest/gtest.h>
#include <cuda/inference/block_manager.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>

namespace cuda::inference::test {

class BlockManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();

        config_ = BlockManagerConfig{
            .max_model_len = 512,
            .block_size = 16,
            .num_cpu_blocks = 128,
            .num_gpu_blocks = 256,
            .enable_cuda_graph = false,
            .kv_cache_config{
                .num_heads = 4,
                .head_dim = 64,
                .block_size_tokens = 16,
                .num_blocks = 256,
                .num_layers = 1
            },
            .attention_config{
                .num_heads = 4,
                .num_kv_heads = 4,
                .head_dim = 64,
                .seq_len = 128,
                .batch_size = 1
            }
        };
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
    BlockManagerConfig config_;
};

TEST_F(BlockManagerTest, Creation) {
    auto manager = std::make_unique<BlockManager>(config_);
    ASSERT_NE(manager, nullptr);
    EXPECT_GT(manager->get_num_free_blocks(), 0);
}

TEST_F(BlockManagerTest, CreateSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 32);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->id, 1);
    EXPECT_EQ(seq->num_tokens, 0);
    EXPECT_GT(seq->block_table.size(), 0);
}

TEST_F(BlockManagerTest, AppendTokens) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 64);
    EXPECT_EQ(seq->num_tokens, 0);

    manager->append_tokens(1, 16);
    EXPECT_EQ(seq->num_tokens, 16);
    EXPECT_GE(seq->block_table.size(), 1);

    manager->append_tokens(1, 16);
    EXPECT_EQ(seq->num_tokens, 32);

    manager->append_tokens(1, 32);
    EXPECT_EQ(seq->num_tokens, 64);
}

TEST_F(BlockManagerTest, AppendTokensExceedsMax) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 32);
    EXPECT_THROW(manager->append_tokens(1, 64), std::runtime_error);
}

TEST_F(BlockManagerTest, GetSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 32);
    manager->create_sequence(2, 32);

    auto* seq1 = manager->get_sequence(1);
    auto* seq2 = manager->get_sequence(2);
    auto* seq3 = manager->get_sequence(999);

    ASSERT_NE(seq1, nullptr);
    ASSERT_NE(seq2, nullptr);
    EXPECT_EQ(seq3, nullptr);

    EXPECT_EQ(seq1->id, 1);
    EXPECT_EQ(seq2->id, 2);
}

TEST_F(BlockManagerTest, FreeSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);
    manager->create_sequence(2, 64);

    const int free_before = manager->get_num_free_blocks();

    manager->free_sequence(1);

    const int free_after = manager->get_num_free_blocks();
    EXPECT_GT(free_after, free_before);

    auto* seq1 = manager->get_sequence(1);
    EXPECT_EQ(seq1, nullptr);
}

TEST_F(BlockManagerTest, DuplicateSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 32);
    EXPECT_THROW(manager->create_sequence(1, 32), std::runtime_error);
}

TEST_F(BlockManagerTest, BlockTableAllocation) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 64);
    ASSERT_EQ(seq->block_table.size(), 4);

    for (const int block_id : seq->block_table) {
        EXPECT_GE(block_id, 0);
    }

    auto* seq2 = manager->create_sequence(2, 64);
    for (const int block_id : seq2->block_table) {
        EXPECT_GE(block_id, 0);
    }

    bool overlap = false;
    for (int id1 : seq->block_table) {
        for (int id2 : seq2->block_table) {
            if (id1 == id2) {
                overlap = true;
                break;
            }
        }
    }
    EXPECT_FALSE(overlap);
}

TEST_F(BlockManagerTest, MultipleSequences) {
    auto manager = std::make_unique<BlockManager>(config_);

    for (int i = 0; i < 10; ++i) {
        manager->create_sequence(i, 32);
    }

    for (int i = 0; i < 10; ++i) {
        auto* seq = manager->get_sequence(i);
        ASSERT_NE(seq, nullptr);
        EXPECT_EQ(seq->id, i);
    }
}

TEST_F(BlockManagerTest, AppendDifferentSequences) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);
    manager->create_sequence(2, 64);

    manager->append_tokens(1, 16);
    manager->append_tokens(2, 48);

    auto* seq1 = manager->get_sequence(1);
    auto* seq2 = manager->get_sequence(2);

    EXPECT_EQ(seq1->num_tokens, 16);
    EXPECT_EQ(seq2->num_tokens, 48);
}

TEST_F(BlockManagerTest, SequenceIsolation) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq1 = manager->create_sequence(1, 64);
    auto* seq2 = manager->create_sequence(2, 64);

    manager->append_tokens(1, 48);
    manager->append_tokens(2, 16);

    EXPECT_EQ(seq1->num_tokens, 48);
    EXPECT_EQ(seq2->num_tokens, 16);

    EXPECT_EQ(manager->get_sequence(1)->num_tokens, 48);
    EXPECT_EQ(manager->get_sequence(2)->num_tokens, 16);
}

TEST_F(BlockManagerTest, ForwardBatchSequenceNotFound) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 32);

    memory::Buffer<float> query(32 * 4 * 64);
    memory::Buffer<float> output(32 * 4 * 64);

    std::vector<int64_t> sequence_ids = {1, 999};

    EXPECT_THROW(
        manager->forward_batch(sequence_ids, query, output, *stream_),
        std::runtime_error
    );
}

TEST_F(BlockManagerTest, KVCacheIntegration) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* kv_cache = manager->get_kv_cache();
    ASSERT_NE(kv_cache, nullptr);

    manager->create_sequence(1, 64);
    manager->create_sequence(2, 64);

    auto stats = kv_cache->get_stats();
    EXPECT_GT(stats.allocated_blocks, 0);
}

TEST_F(BlockManagerTest, MaybeEvict) {
    auto manager = std::make_unique<BlockManager>(config_);

    for (int i = 0; i < 100; ++i) {
        manager->create_sequence(i, 64);
    }

    EXPECT_NO_THROW(manager->maybe_evict());
}

TEST_F(BlockManagerTest, MaxTokensBoundary) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 32);
    EXPECT_EQ(seq->max_tokens, 32);

    manager->append_tokens(1, 32);
    EXPECT_EQ(seq->num_tokens, 32);

    EXPECT_THROW(manager->append_tokens(1, 1), std::runtime_error);
}

}  // namespace cuda::inference::test
