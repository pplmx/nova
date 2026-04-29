#include <gtest/gtest.h>
#include <cuda/inference/scheduler.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>

namespace cuda::inference::test {

class SchedulerEdgeTest : public ::testing::Test {
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
            .num_kv_heads = 2,
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

TEST_F(SchedulerEdgeTest, AddSingleRequest) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_GE(seq_id, 0);
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_active_sequences(), 1);
}

TEST_F(SchedulerEdgeTest, AddMultipleRequests) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    std::vector<int64_t> ids;
    for (int i = 0; i < 5; ++i) {
        ids.push_back(scheduler->add_request(64));
    }

    EXPECT_EQ(ids.size(), 5);
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_active_sequences(), 5);
}

TEST_F(SchedulerEdgeTest, GetBatchUnderLimit) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int i = 0; i < 8; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    EXPECT_LE(batch.size(), config_.max_batch_size);
    EXPECT_EQ(batch.size(), 8);
}

TEST_F(SchedulerEdgeTest, GetBatchAtLimit) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int i = 0; i < config_.max_batch_size; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    EXPECT_EQ(batch.size(), config_.max_batch_size);
}

TEST_F(SchedulerEdgeTest, GetBatchOverLimit) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int i = 0; i < config_.max_batch_size + 10; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    EXPECT_EQ(batch.size(), config_.max_batch_size);
}

TEST_F(SchedulerEdgeTest, ContinuousBatchingReplacesCompleted) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch1 = scheduler->get_batch();
    EXPECT_EQ(batch1.size(), 2);

    scheduler->on_sequence_complete(batch1[0]);

    scheduler->add_request(64);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 2);
    EXPECT_NE(std::find(batch2.begin(), batch2.end(), batch1[1]), batch2.end());
}

TEST_F(SchedulerEdgeTest, SequenceCompleteUpdatesState) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_finished_sequences(), 0);

    scheduler->on_sequence_complete(seq_id);
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_finished_sequences(), 1);
}

TEST_F(SchedulerEdgeTest, SequenceCompleteRemovesFromBatch) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch1 = scheduler->get_batch();
    EXPECT_EQ(batch1.size(), 2);

    scheduler->on_sequence_complete(seq_id);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 1);
    EXPECT_NE(std::find(batch2.begin(), batch2.end(), batch1[1]), batch2.end());
}

TEST_F(SchedulerEdgeTest, GQAConfiguration) {
    auto config_gqa = config_;
    config_gqa.num_heads = 8;
    config_gqa.num_kv_heads = 2;

    auto scheduler = std::make_unique<Scheduler>(config_gqa);
    EXPECT_EQ(scheduler->config().num_heads, 8);
    EXPECT_EQ(scheduler->config().num_kv_heads, 2);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_GE(seq_id, 0);
}

TEST_F(SchedulerEdgeTest, MQAConfiguration) {
    auto config_mqa = config_;
    config_mqa.num_heads = 8;
    config_mqa.num_kv_heads = 1;

    auto scheduler = std::make_unique<Scheduler>(config_mqa);
    EXPECT_EQ(scheduler->config().num_heads, 8);
    EXPECT_EQ(scheduler->config().num_kv_heads, 1);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_GE(seq_id, 0);
}

TEST_F(SchedulerEdgeTest, MaxSequenceLengthConfiguration) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(config_.max_sequence_length);
    EXPECT_GE(seq_id, 0);

    auto seq = scheduler->get_block_manager().get_sequence(seq_id);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->max_tokens, config_.max_sequence_length);
}

TEST_F(SchedulerEdgeTest, BatchCompositionAfterMultipleCompletes) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    std::vector<int64_t> seq_ids;
    for (int i = 0; i < 5; ++i) {
        seq_ids.push_back(scheduler->add_request(64));
    }

    auto batch1 = scheduler->get_batch();
    EXPECT_EQ(batch1.size(), 5);

    scheduler->on_sequence_complete(seq_ids[0]);
    scheduler->on_sequence_complete(seq_ids[2]);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 3);
}

TEST_F(SchedulerEdgeTest, StepRecomposesBatch) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch1 = scheduler->get_batch();
    EXPECT_EQ(batch1.size(), 2);

    scheduler->step();

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 2);
}

TEST_F(SchedulerEdgeTest, EmptyScheduler) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    auto batch = scheduler->get_batch();
    EXPECT_TRUE(batch.empty());

    EXPECT_EQ(scheduler->get_sequence_manager().get_num_active_sequences(), 0);
}

TEST_F(SchedulerEdgeTest, AllSequencesComplete) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t id1 = scheduler->add_request(64);
    int64_t id2 = scheduler->add_request(64);

    scheduler->on_sequence_complete(id1);
    scheduler->on_sequence_complete(id2);

    auto batch = scheduler->get_batch();
    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(scheduler->get_sequence_manager().get_num_finished_sequences(), 2);
}

TEST_F(SchedulerEdgeTest, BlockManagerIntegration) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto& block_manager = scheduler->get_block_manager();
    auto* kv_cache = block_manager.get_kv_cache();

    ASSERT_NE(kv_cache, nullptr);
    auto stats = kv_cache->get_stats();
    EXPECT_GT(stats.allocated_blocks, 0);
}

TEST_F(SchedulerEdgeTest, SequenceManagerAccess) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t id1 = scheduler->add_request(64);
    int64_t id2 = scheduler->add_request(64);
    int64_t id3 = scheduler->add_request(64);

    auto& seq_manager = scheduler->get_sequence_manager();

    auto running = seq_manager.get_running_sequences();
    EXPECT_EQ(running.size(), 3);
    EXPECT_TRUE(std::find(running.begin(), running.end(), id1) != running.end());
    EXPECT_TRUE(std::find(running.begin(), running.end(), id2) != running.end());
    EXPECT_TRUE(std::find(running.begin(), running.end(), id3) != running.end());

    scheduler->on_sequence_complete(id2);

    running = seq_manager.get_running_sequences();
    EXPECT_EQ(running.size(), 2);
    EXPECT_FALSE(std::find(running.begin(), running.end(), id2) != running.end());
}

TEST_F(SchedulerEdgeTest, NonContinuousBatchingMode) {
    auto config_ncb = config_;
    config_ncb.enable_continuous_batching = false;

    auto scheduler = std::make_unique<Scheduler>(config_ncb);

    for (int i = 0; i < 5; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    EXPECT_EQ(batch.size(), 5);
}

TEST_F(SchedulerEdgeTest, SequenceStateTransitions) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);

    EXPECT_EQ(scheduler->get_sequence_manager().get_state(seq_id), SequenceState::Running);

    scheduler->on_sequence_complete(seq_id);

    EXPECT_EQ(scheduler->get_sequence_manager().get_state(seq_id), SequenceState::Finished);
}

TEST_F(SchedulerEdgeTest, MultipleBatchesOverTime) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int round = 0; round < 5; ++round) {
        scheduler->add_request(64);
        scheduler->add_request(64);

        auto batch = scheduler->get_batch();
        EXPECT_EQ(batch.size(), 2);

        scheduler->on_sequence_complete(batch[0]);

        scheduler->step();
    }
}

}  // namespace cuda::inference::test
