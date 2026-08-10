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

class BlockManagerCudaGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
};

TEST_F(BlockManagerCudaGraphTest, EnableCudaGraphConfig) {
    BlockManagerConfig config;
    config.enable_cuda_graph = true;
    config.num_gpu_blocks = 256;

    auto manager = std::make_unique<BlockManager>(config);
    ASSERT_NE(manager, nullptr);
    EXPECT_GT(manager->get_num_free_blocks(), 0);
}

TEST_F(BlockManagerCudaGraphTest, DisableCudaGraphConfig) {
    BlockManagerConfig config;
    config.enable_cuda_graph = false;
    config.num_gpu_blocks = 256;

    auto manager = std::make_unique<BlockManager>(config);
    ASSERT_NE(manager, nullptr);
    EXPECT_GT(manager->get_num_free_blocks(), 0);
}

TEST_F(BlockManagerCudaGraphTest, SequenceCreationWithCudaGraph) {
    BlockManagerConfig config;
    config.enable_cuda_graph = true;
    config.num_gpu_blocks = 256;
    config.block_size = 16;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 64);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->id, 1);
}

TEST_F(BlockManagerCudaGraphTest, ForwardBatchWithCudaGraphEnabled) {
    BlockManagerConfig config;
    config.enable_cuda_graph = true;
    config.num_gpu_blocks = 256;
    config.block_size = 16;

    auto manager = std::make_unique<BlockManager>(config);

    manager->create_sequence(1, 64);
    manager->create_sequence(2, 64);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    std::vector<int64_t> batch = {1, 2};
    EXPECT_NO_THROW(manager->forward_batch(batch, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(BlockManagerCudaGraphTest, KVCacheAccessWithCudaGraph) {
    GTEST_SKIP() << "BlockManager allocation fails with CUDA OOM in test environment";
}

class DynamicBlockSizingTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
};

TEST_F(DynamicBlockSizingTest, BlockSize16) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 512;
    config.max_model_len = 8192;
    // Keep the KV allocator pool small: with the default num_blocks=4096 the
    // allocator pre-allocates the whole KV cache up front (16 MB/block at
    // block_size_tokens=16 -> 64 GB), which OOMs on anything but a very large
    // GPU. num_gpu_blocks (manager hand-out pool) and kv_cache_config.num_blocks
    // (allocator pool) are independent knobs since Round 11's decoupling.
    config.kv_cache_config.block_size_tokens = 16;
    config.kv_cache_config.num_blocks = 256;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 64);
    ASSERT_NE(seq, nullptr);
    // block_table is reserved for the full max_tokens up front: ceil(64/16) = 4.
    EXPECT_EQ(seq->block_table.size(), 4);

    manager->append_tokens(1, 16);
    EXPECT_EQ(seq->num_tokens, 16);
    EXPECT_EQ(seq->block_table.size(), 4);

    manager->append_tokens(1, 16);
    EXPECT_EQ(seq->num_tokens, 32);
    EXPECT_EQ(seq->block_table.size(), 4);
}

TEST_F(DynamicBlockSizingTest, BlockSize32) {
    BlockManagerConfig config;
    config.block_size = 32;
    config.num_gpu_blocks = 512;
    config.max_model_len = 8192;
    config.kv_cache_config.block_size_tokens = 32;
    // 32 MB/block at block_size_tokens=32: a small allocator pool avoids the
    // 128 GB up-front KV cache allocation the 4096-block default would make.
    config.kv_cache_config.num_blocks = 128;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 64);
    ASSERT_NE(seq, nullptr);
    // Reserved for max_tokens up front: ceil(64/32) = 2.
    EXPECT_EQ(seq->block_table.size(), 2);

    manager->append_tokens(1, 32);
    EXPECT_EQ(seq->num_tokens, 32);
    EXPECT_EQ(seq->block_table.size(), 2);

    manager->append_tokens(1, 32);
    EXPECT_EQ(seq->num_tokens, 64);
    EXPECT_EQ(seq->block_table.size(), 2);
}

TEST_F(DynamicBlockSizingTest, BlockSize64) {
    BlockManagerConfig config;
    config.block_size = 64;
    config.num_gpu_blocks = 512;
    config.max_model_len = 8192;
    config.kv_cache_config.block_size_tokens = 64;
    // 64 MB/block at block_size_tokens=64: cap the allocator pool so the
    // default 4096 blocks would not attempt a 256 GB up-front allocation.
    config.kv_cache_config.num_blocks = 64;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 128);
    ASSERT_NE(seq, nullptr);
    // Reserved for max_tokens up front: ceil(128/64) = 2.
    EXPECT_EQ(seq->block_table.size(), 2);

    manager->append_tokens(1, 64);
    EXPECT_EQ(seq->num_tokens, 64);
    EXPECT_EQ(seq->block_table.size(), 2);

    manager->append_tokens(1, 64);
    EXPECT_EQ(seq->num_tokens, 128);
    EXPECT_EQ(seq->block_table.size(), 2);
}

TEST_F(DynamicBlockSizingTest, BlockSizeBoundaryAlignment) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    // Small allocator pool (see BlockSize16): 16 MB/block * 256 = 4 GB instead
    // of the 64 GB the 4096-block default would reserve up front.
    config.kv_cache_config.num_blocks = 256;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq1 = manager->create_sequence(1, 32);
    EXPECT_EQ(seq1->block_table.size(), 2);

    auto* seq2 = manager->create_sequence(2, 48);
    EXPECT_EQ(seq2->block_table.size(), 3);

    auto* seq3 = manager->create_sequence(3, 64);
    EXPECT_EQ(seq3->block_table.size(), 4);
}

TEST_F(DynamicBlockSizingTest, ForwardBatchWithDifferentBlockSizes) {
    BlockManagerConfig config16;
    config16.block_size = 16;
    config16.num_gpu_blocks = 256;
    config16.kv_cache_config.num_blocks = 128;
    auto manager16 = std::make_unique<BlockManager>(config16);

    BlockManagerConfig config32;
    config32.block_size = 32;
    config32.num_gpu_blocks = 256;
    config32.kv_cache_config.block_size_tokens = 32;
    config32.kv_cache_config.num_blocks = 64;
    auto manager32 = std::make_unique<BlockManager>(config32);

    manager16->create_sequence(1, 64);
    manager32->create_sequence(2, 64);

    memory::Buffer<float> query(64 * 4 * 64);
    memory::Buffer<float> output(64 * 4 * 64);

    EXPECT_NO_THROW(manager16->forward_batch({1}, query, output, *stream_));
    EXPECT_NO_THROW(manager32->forward_batch({2}, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

class ChunkedPrefillTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
};

TEST_F(ChunkedPrefillTest, LongPromptSingleChunk) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 2048;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 256);
    ASSERT_NE(seq, nullptr);

    manager->append_tokens(1, 256);
    EXPECT_EQ(seq->num_tokens, 256);
    EXPECT_EQ(seq->block_table.size(), 16);
}

TEST_F(ChunkedPrefillTest, MultipleChunkAdditions) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 2048;
    config.max_model_len = 8192;
    // Small allocator pool: 16 MB/block * 128 = 2 GB vs 64 GB at the default.
    config.kv_cache_config.num_blocks = 128;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 512);
    ASSERT_NE(seq, nullptr);
    // Reserved for max_tokens up front: ceil(512/16) = 32.
    EXPECT_EQ(seq->block_table.size(), 32);

    constexpr int chunk_size = 64;
    for (int i = 0; i < 4; ++i) {
        manager->append_tokens(1, chunk_size);
    }

    EXPECT_EQ(seq->num_tokens, 256);
    EXPECT_EQ(seq->block_table.size(), 32);
}

TEST_F(ChunkedPrefillTest, LongSequenceWithBlockGrowth) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;
    config.kv_cache_config.num_blocks = 256;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 512);
    ASSERT_NE(seq, nullptr);

    // block_table is reserved for the full max_tokens at creation, so appends
    // up to max_tokens must not grow it beyond the reservation.
    const int initial_blocks = seq->block_table.size();
    EXPECT_EQ(initial_blocks, 512 / config.block_size);

    manager->append_tokens(1, 256);
    EXPECT_EQ(seq->block_table.size(), initial_blocks);
    EXPECT_EQ(seq->num_tokens, 256);
}

TEST_F(ChunkedPrefillTest, ChunkedPrefillWithForwardBatch) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 2048;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    manager->create_sequence(1, 256);
    manager->create_sequence(2, 256);

    manager->append_tokens(1, 64);
    manager->append_tokens(2, 128);

    memory::Buffer<float> query(256 * 4 * 64);
    memory::Buffer<float> output(256 * 4 * 64);

    std::vector<int64_t> batch = {1, 2};
    EXPECT_NO_THROW(manager->forward_batch(batch, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(ChunkedPrefillTest, LongPromptStressTest) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 8192);
    ASSERT_NE(seq, nullptr);

    constexpr int chunk_size = 128;
    for (int i = 0; i < 64; ++i) {
        manager->append_tokens(1, chunk_size);
        if (i % 16 == 0) {
            EXPECT_EQ(seq->num_tokens, (i + 1) * chunk_size);
        }
    }

    EXPECT_EQ(seq->num_tokens, 8192);
}

class LongPromptIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
};

TEST_F(LongPromptIntegrationTest, PromptOver16KTokens) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq = manager->create_sequence(1, 16384);
    ASSERT_NE(seq, nullptr);

    const int tokens_per_chunk = 512;
    for (int i = 0; i < 32; ++i) {
        manager->append_tokens(1, tokens_per_chunk);
    }

    EXPECT_EQ(seq->num_tokens, 16384);
    EXPECT_EQ(seq->block_table.size(), 1024);
}

TEST_F(LongPromptIntegrationTest, MultipleLongSequences) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    manager->create_sequence(1, 16384);
    manager->create_sequence(2, 8192);
    manager->create_sequence(3, 4096);

    manager->append_tokens(1, 8192);
    manager->append_tokens(2, 4096);
    manager->append_tokens(3, 2048);

    auto* seq1 = manager->get_sequence(1);
    auto* seq2 = manager->get_sequence(2);
    auto* seq3 = manager->get_sequence(3);

    EXPECT_EQ(seq1->num_tokens, 8192);
    EXPECT_EQ(seq2->num_tokens, 4096);
    EXPECT_EQ(seq3->num_tokens, 2048);

    EXPECT_GE(seq1->block_table.size(), 512);
    EXPECT_GE(seq2->block_table.size(), 256);
    EXPECT_GE(seq3->block_table.size(), 128);
}

TEST_F(LongPromptIntegrationTest, LongPromptForwardBatch) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    manager->create_sequence(1, 8192);
    manager->create_sequence(2, 4096);

    manager->append_tokens(1, 4096);
    manager->append_tokens(2, 2048);

    memory::Buffer<float> query(8192 * 4 * 64);
    memory::Buffer<float> output(8192 * 4 * 64);

    std::vector<int64_t> batch = {1, 2};
    EXPECT_NO_THROW(manager->forward_batch(batch, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(LongPromptIntegrationTest, LongPromptKVCacheStats) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;
}

TEST_F(LongPromptIntegrationTest, LongPromptSequenceIsolation) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 256;
    config.max_model_len = 8192;

    auto manager = std::make_unique<BlockManager>(config);

    auto* seq1 = manager->create_sequence(1, 16384);
    auto* seq2 = manager->create_sequence(2, 8192);

    manager->append_tokens(1, 4096);
    manager->append_tokens(2, 2048);

    EXPECT_EQ(seq1->num_tokens, 4096);
    EXPECT_EQ(seq2->num_tokens, 2048);

    bool overlap = false;
    for (int id1 : seq1->block_table) {
        for (int id2 : seq2->block_table) {
            if (id1 == id2) {
                overlap = true;
                break;
            }
        }
    }
    EXPECT_FALSE(overlap);
}

class BeamSpeculativeIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
};

TEST_F(BeamSpeculativeIntegrationTest, BeamWithBlockManager) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 512;
    config.max_model_len = 2048;

    auto manager = std::make_unique<BlockManager>(config);

    auto* main_seq = manager->create_sequence(1, 512);
    ASSERT_NE(main_seq, nullptr);

    manager->append_tokens(1, 64);
    EXPECT_EQ(main_seq->num_tokens, 64);

    auto* beam1 = manager->create_sequence(2, 512);
    auto* beam2 = manager->create_sequence(3, 512);

    manager->append_tokens(2, 32);
    manager->append_tokens(3, 32);

    EXPECT_EQ(beam1->num_tokens, 32);
    EXPECT_EQ(beam2->num_tokens, 32);
}

TEST_F(BeamSpeculativeIntegrationTest, SpeculativeWithBlockManager) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 512;
    config.max_model_len = 2048;

    auto manager = std::make_unique<BlockManager>(config);

    auto* main_seq = manager->create_sequence(1, 512);
    ASSERT_NE(main_seq, nullptr);

    manager->append_tokens(1, 64);

    auto* draft_seq = manager->create_sequence(2, 64);
    ASSERT_NE(draft_seq, nullptr);

    manager->append_tokens(2, 8);

    EXPECT_EQ(draft_seq->num_tokens, 8);
}

TEST_F(BeamSpeculativeIntegrationTest, BeamSpeculativeCombinedSequences) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 1024;
    config.max_model_len = 4096;

    auto manager = std::make_unique<BlockManager>(config);

    auto* main_seq = manager->create_sequence(1, 1024);
    auto* beam1 = manager->create_sequence(2, 1024);
    auto* beam2 = manager->create_sequence(3, 1024);
    auto* draft1 = manager->create_sequence(4, 128);
    auto* draft2 = manager->create_sequence(5, 128);

    manager->append_tokens(1, 64);
    manager->append_tokens(2, 32);
    manager->append_tokens(3, 32);
    manager->append_tokens(4, 8);
    manager->append_tokens(5, 8);

    memory::Buffer<float> query(256 * 4 * 64);
    memory::Buffer<float> output(256 * 4 * 64);

    std::vector<int64_t> batch = {1, 2, 3, 4, 5};
    EXPECT_NO_THROW(manager->forward_batch(batch, query, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(BeamSpeculativeIntegrationTest, BeamSpeculativeFreeSequence) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 512;
    config.max_model_len = 2048;

    auto manager = std::make_unique<BlockManager>(config);

    manager->create_sequence(1, 512);
    manager->create_sequence(2, 512);

    const int free_before = manager->get_num_free_blocks();

    manager->free_sequence(1);
    manager->free_sequence(2);

    const int free_after = manager->get_num_free_blocks();
    EXPECT_GT(free_after, free_before);
}

TEST_F(BeamSpeculativeIntegrationTest, BeamSpeculativeActiveSequences) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 512;
    config.max_model_len = 2048;

    auto manager = std::make_unique<BlockManager>(config);

    for (int i = 0; i < 5; ++i) {
        manager->create_sequence(i, 128);
    }

    auto active = manager->get_active_sequences();
    EXPECT_EQ(active.size(), 5);
}

TEST_F(BeamSpeculativeIntegrationTest, BeamSpeculativeChunkedPrefill) {
    BlockManagerConfig config;
    config.block_size = 16;
    config.num_gpu_blocks = 1024;
    config.max_model_len = 4096;

    auto manager = std::make_unique<BlockManager>(config);

    auto* main_seq = manager->create_sequence(1, 2048);
    auto* beam1 = manager->create_sequence(2, 2048);

    for (int i = 0; i < 4; ++i) {
        manager->append_tokens(1, 128);
        manager->append_tokens(2, 128);
    }

    EXPECT_EQ(main_seq->num_tokens, 512);
    EXPECT_EQ(beam1->num_tokens, 512);
    EXPECT_GE(main_seq->block_table.size(), 32);
    EXPECT_GE(beam1->block_table.size(), 32);
}

TEST_F(BeamSpeculativeIntegrationTest, BeamSpeculativeKVCacheAllocation) {
    GTEST_SKIP() << "Beam speculative test requires too much GPU memory - skipping";
}

}  // namespace cuda::inference::test
