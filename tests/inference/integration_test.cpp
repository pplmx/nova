#include <gtest/gtest.h>
#include <cuda/inference/scheduler.h>
#include <cuda/observability/inference_nvtx.h>
#include <cuda/production/inference_graph.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>
#include <chrono>

namespace cuda::inference::test {

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();

        config_ = SchedulerConfig{
            .max_batch_size = 16,
            .max_sequence_length = 512,
            .prefill_batch_size = 4,
            .enable_continuous_batching = true,
            .enable_prefix_caching = true,
            .num_heads = 4,
            .num_kv_heads = 4,
            .head_dim = 64,
            .block_size = 16
        };
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
    SchedulerConfig config_;
};

TEST_F(IntegrationTest, EndToEndSingleSequence) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);
    ASSERT_GE(seq_id, 0);

    auto batch = scheduler->get_batch();
    ASSERT_EQ(batch.size(), 1);
    EXPECT_EQ(batch[0], seq_id);

    auto& block_manager = scheduler->get_block_manager();
    auto* seq = block_manager.get_sequence(seq_id);
    ASSERT_NE(seq, nullptr);
    EXPECT_GE(seq->max_tokens, 64);
}

TEST_F(IntegrationTest, EndToEndMultiSequence) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int i = 0; i < 5; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    ASSERT_EQ(batch.size(), 5);

    for (const int64_t seq_id : batch) {
        auto* seq = scheduler->get_block_manager().get_sequence(seq_id);
        ASSERT_NE(seq, nullptr);
    }
}

TEST_F(IntegrationTest, ContinuousBatchingLoop) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch1 = scheduler->get_batch();
    ASSERT_EQ(batch1.size(), 2);

    scheduler->on_sequence_complete(batch1[0]);

    scheduler->add_request(64);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 2);
    EXPECT_NE(std::find(batch2.begin(), batch2.end(), batch1[1]), batch2.end());
    EXPECT_NE(std::find(batch2.begin(), batch2.end(), batch1[0]), batch2.end());
}

TEST_F(IntegrationTest, SequenceLifecycle) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq1 = scheduler->add_request(64);
    int64_t seq2 = scheduler->add_request(64);

    EXPECT_EQ(scheduler->get_sequence_manager().get_num_active_sequences(), 2);

    scheduler->on_sequence_complete(seq1);
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_finished_sequences(), 1);

    scheduler->on_sequence_complete(seq2);
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_finished_sequences(), 2);
}

TEST_F(IntegrationTest, NVTXAnnotations) {
    auto& nvtx = observability::InferenceNVTXDomain::get();

    EXPECT_NO_THROW(nvtx.begin_prefill());
    EXPECT_NO_THROW(nvtx.end_prefill());

    EXPECT_NO_THROW(nvtx.begin_decode());
    EXPECT_NO_THROW(nvtx.end_decode());

    EXPECT_NO_THROW(nvtx.begin_attention());
    EXPECT_NO_THROW(nvtx.end_attention());

    EXPECT_NO_THROW(nvtx.begin_scheduling());
    EXPECT_NO_THROW(nvtx.end_scheduling());

    EXPECT_NO_THROW(nvtx.record_batch_size(8));
    EXPECT_NO_THROW(nvtx.record_sequence_length(64));
}

TEST_F(IntegrationTest, ScopedNVTX) {
    {
        observability::ScopedPrefill prefill;
        observability::ScopedAttention attention;
        observability::ScopedScheduling scheduling;
    }

    SUCCEED();
}

TEST_F(IntegrationTest, BlockManagerIntegration) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto& block_manager = scheduler->get_block_manager();
    auto* kv_cache = block_manager.get_kv_cache();
    ASSERT_NE(kv_cache, nullptr);

    auto stats = kv_cache->get_stats();
    EXPECT_GT(stats.allocated_blocks, 0);
    EXPECT_EQ(stats.free_blocks, stats.total_blocks - stats.allocated_blocks);
}

TEST_F(IntegrationTest, GQAConfiguration) {
    auto config_gqa = config_;
    config_gqa.num_heads = 8;
    config_gqa.num_kv_heads = 2;

    auto scheduler = std::make_unique<Scheduler>(config_gqa);

    int64_t seq_id = scheduler->add_request(64);
    ASSERT_GE(seq_id, 0);

    auto batch = scheduler->get_batch();
    ASSERT_EQ(batch.size(), 1);
}

TEST_F(IntegrationTest, MemoryCleanup) {
    {
        auto scheduler = std::make_unique<Scheduler>(config_);

        for (int i = 0; i < 10; ++i) {
            scheduler->add_request(64);
        }

        auto batch = scheduler->get_batch();
        ASSERT_EQ(batch.size(), 10);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    SUCCEED();
}

TEST_F(IntegrationTest, MaxBatchSize) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int i = 0; i < 100; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    EXPECT_LE(batch.size(), config_.max_batch_size);
}

TEST_F(IntegrationTest, SequenceLengthTracking) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);

    auto& block_manager = scheduler->get_block_manager();
    block_manager.append_tokens(seq_id, 1);

    auto* seq = block_manager.get_sequence(seq_id);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->num_tokens, 1);

    block_manager.append_tokens(seq_id, 1);
    EXPECT_EQ(seq->num_tokens, 2);
}

}  // namespace cuda::inference::test
