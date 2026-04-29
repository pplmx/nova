#include <gtest/gtest.h>
#include <cuda/inference/block_manager.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>

namespace cuda::inference::test {

class BlockManagerEdgeTest : public ::testing::Test {
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

TEST_F(BlockManagerEdgeTest, CreateSequenceWithMaxTokens) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, config_.max_model_len);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->id, 1);
    EXPECT_EQ(seq->max_tokens, config_.max_model_len);
    EXPECT_EQ(seq->num_tokens, 0);
}

TEST_F(BlockManagerEdgeTest, CreateSequenceWithSingleToken) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 1);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->max_tokens, 1);
}

TEST_F(BlockManagerEdgeTest, AppendTokensExactlyFillsBlock) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, config_.block_size);
    EXPECT_EQ(seq->num_tokens, 0);

    manager->append_tokens(1, config_.block_size);
    EXPECT_EQ(seq->num_tokens, config_.block_size);

    auto blocks = manager->get_sequence(1)->block_table;
    EXPECT_EQ(blocks.size(), 1);
}

TEST_F(BlockManagerEdgeTest, AppendTokensSpansMultipleBlocks) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 100);
    manager->append_tokens(1, 50);

    EXPECT_EQ(seq->num_tokens, 50);
    EXPECT_GE(seq->block_table.size(), 3);
}

TEST_F(BlockManagerEdgeTest, AppendTokensBoundaryCondition) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, config_.max_model_len);
    manager->append_tokens(1, config_.max_model_len - 1);

    EXPECT_EQ(seq->num_tokens, config_.max_model_len - 1);

    EXPECT_THROW(manager->append_tokens(1, 10), std::runtime_error);
}

TEST_F(BlockManagerEdgeTest, AppendTokensExactBoundary) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 64);
    manager->append_tokens(1, 64);

    EXPECT_EQ(seq->num_tokens, 64);
    EXPECT_NO_THROW(manager->append_tokens(1, 0));
}

TEST_F(BlockManagerEdgeTest, FreeSequenceReturnsBlocks) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);
    const int free_before = manager->get_num_free_blocks();

    manager->free_sequence(1);
    const int free_after = manager->get_num_free_blocks();

    EXPECT_GT(free_after, free_before);
}

TEST_F(BlockManagerEdgeTest, FreeNonExistentSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    EXPECT_NO_THROW(manager->free_sequence(999));
    EXPECT_EQ(manager->get_sequence(999), nullptr);
}

TEST_F(BlockManagerEdgeTest, GetNonExistentSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    EXPECT_EQ(manager->get_sequence(999), nullptr);
    EXPECT_EQ(manager->get_sequence(0), nullptr);
}

TEST_F(BlockManagerEdgeTest, BlockTableContainsValidIndices) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 64);
    ASSERT_GT(seq->block_table.size(), 0);

    for (int block_id : seq->block_table) {
        EXPECT_GE(block_id, 0);
        EXPECT_LT(block_id, config_.num_gpu_blocks);
    }
}

TEST_F(BlockManagerEdgeTest, MultipleSequencesBlockTablesIndependent) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq1 = manager->create_sequence(1, 64);
    auto* seq2 = manager->create_sequence(2, 64);
    auto* seq3 = manager->create_sequence(3, 64);

    std::vector<int> ids1 = seq1->block_table;
    std::vector<int> ids2 = seq2->block_table;
    std::vector<int> ids3 = seq3->block_table;

    for (int id1 : ids1) {
        for (int id2 : ids2) {
            EXPECT_NE(id1, id2) << "Block tables should not overlap";
        }
        for (int id3 : ids3) {
            EXPECT_NE(id1, id3) << "Block tables should not overlap";
        }
    }
}

TEST_F(BlockManagerEdgeTest, AppendMultipleTimes) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 256);

    for (int i = 0; i < 10; ++i) {
        manager->append_tokens(1, 16);
    }

    EXPECT_EQ(seq->num_tokens, 160);
    EXPECT_GE(seq->block_table.size(), 10);
}

TEST_F(BlockManagerEdgeTest, ForwardBatchWithEmptyBatch) {
    auto manager = std::make_unique<BlockManager>(config_);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    std::vector<int64_t> empty_batch;
    EXPECT_NO_THROW(manager->forward_batch(empty_batch, query, output, *stream_));
}

TEST_F(BlockManagerEdgeTest, ForwardBatchWithSingleSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    std::vector<int64_t> batch = {1};
    EXPECT_NO_THROW(manager->forward_batch(batch, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(BlockManagerEdgeTest, ForwardBatchWithMultipleSequences) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);
    manager->create_sequence(2, 64);
    manager->create_sequence(3, 64);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    std::vector<int64_t> batch = {1, 2, 3};
    EXPECT_NO_THROW(manager->forward_batch(batch, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(BlockManagerEdgeTest, ForwardBatchWithMissingSequence) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    std::vector<int64_t> batch = {1, 999};
    EXPECT_THROW(manager->forward_batch(batch, query, output, *stream_), std::runtime_error);
}

TEST_F(BlockManagerEdgeTest, MaybeEvictDoesNotThrow) {
    auto manager = std::make_unique<BlockManager>(config_);

    for (int i = 0; i < 100; ++i) {
        manager->create_sequence(i, 32);
    }

    EXPECT_NO_THROW(manager->maybe_evict());
}

TEST_F(BlockManagerEdgeTest, KVCacheAccessAfterSequenceCreation) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);

    auto* kv_cache = manager->get_kv_cache();
    ASSERT_NE(kv_cache, nullptr);

    auto stats = kv_cache->get_stats();
    EXPECT_GT(stats.allocated_blocks, 0);
}

TEST_F(BlockManagerEdgeTest, SequenceStateTracking) {
    auto manager = std::make_unique<BlockManager>(config_);

    auto* seq = manager->create_sequence(1, 64);
    EXPECT_EQ(seq->num_tokens, 0);
    EXPECT_GT(seq->created_at, 0);

    manager->append_tokens(1, 16);
    EXPECT_EQ(seq->num_tokens, 16);

    manager->free_sequence(1);
    EXPECT_EQ(manager->get_sequence(1), nullptr);
}

TEST_F(BlockManagerEdgeTest, BlockTableSyncStream) {
    auto manager = std::make_unique<BlockManager>(config_);

    manager->create_sequence(1, 64);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    EXPECT_NO_THROW(manager->sync_block_tables(*stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

}  // namespace cuda::inference::test
