#include <gtest/gtest.h>
#include <cuda/inference/scheduler.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>

namespace cuda::inference::test {

class SchedulerTest : public ::testing::Test {
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

TEST_F(SchedulerTest, Creation) {
    auto scheduler = std::make_unique<Scheduler>(config_);
    ASSERT_NE(scheduler, nullptr);

    EXPECT_EQ(scheduler->config().max_batch_size, 16);
    EXPECT_EQ(scheduler->config().num_heads, 4);
    EXPECT_EQ(scheduler->config().num_kv_heads, 2);
}

TEST_F(SchedulerTest, AddRequest) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_GE(seq_id, 0);

    auto& seq_manager = scheduler->get_sequence_manager();
    EXPECT_EQ(seq_manager.get_num_active_sequences(), 1);
}

TEST_F(SchedulerTest, GetBatch) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch = scheduler->get_batch();
    EXPECT_EQ(batch.size(), 3);
}

TEST_F(SchedulerTest, BatchSizeLimit) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    for (int i = 0; i < 20; ++i) {
        scheduler->add_request(64);
    }

    auto batch = scheduler->get_batch();
    EXPECT_LE(batch.size(), config_.max_batch_size);
}

TEST_F(SchedulerTest, ContinuousBatching) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch1 = scheduler->get_batch();
    EXPECT_EQ(batch1.size(), 2);

    scheduler->on_sequence_complete(batch1[0]);

    scheduler->add_request(64);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 2);
}

TEST_F(SchedulerTest, GQASupport) {
    auto config_gqa = config_;
    config_gqa.num_heads = 8;
    config_gqa.num_kv_heads = 2;

    auto scheduler = std::make_unique<Scheduler>(config_gqa);

    EXPECT_EQ(scheduler->config().num_heads, 8);
    EXPECT_EQ(scheduler->config().num_kv_heads, 2);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_GE(seq_id, 0);
}

TEST_F(SchedulerTest, MQASupport) {
    auto config_mqa = config_;
    config_mqa.num_heads = 8;
    config_mqa.num_kv_heads = 1;

    auto scheduler = std::make_unique<Scheduler>(config_mqa);

    EXPECT_EQ(scheduler->config().num_heads, 8);
    EXPECT_EQ(scheduler->config().num_kv_heads, 1);

    int64_t seq_id = scheduler->add_request(64);
    EXPECT_GE(seq_id, 0);
}

TEST_F(SchedulerTest, SequenceComplete) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);
    scheduler->add_request(64);

    EXPECT_EQ(scheduler->get_sequence_manager().get_num_active_sequences(), 2);

    scheduler->on_sequence_complete(seq_id);

    EXPECT_EQ(scheduler->get_sequence_manager().get_num_finished_sequences(), 1);
}

TEST_F(SchedulerTest, TokenGenerated) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(64);

    auto seq_before = scheduler->get_block_manager().get_sequence(seq_id);
    ASSERT_NE(seq_before, nullptr);
    const int tokens_before = seq_before->num_tokens;

    scheduler->get_block_manager().append_tokens(seq_id, 1);

    auto seq_after = scheduler->get_block_manager().get_sequence(seq_id);
    ASSERT_NE(seq_after, nullptr);

    EXPECT_EQ(seq_after->num_tokens, tokens_before + 1);
}

TEST_F(SchedulerTest, MultipleBatches) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch1 = scheduler->get_batch();
    EXPECT_EQ(batch1.size(), 2);

    scheduler->step();

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 4);
}

TEST_F(SchedulerTest, SequenceIsolation) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq1 = scheduler->add_request(64);
    int64_t seq2 = scheduler->add_request(64);

    scheduler->on_sequence_complete(seq1);

    auto& seq_manager = scheduler->get_sequence_manager();
    EXPECT_EQ(seq_manager.get_state(seq1), SequenceState::Finished);
    EXPECT_EQ(seq_manager.get_state(seq2), SequenceState::Running);
}

TEST_F(SchedulerTest, BatchComposition) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    scheduler->add_request(32);
    scheduler->add_request(64);
    scheduler->add_request(96);

    auto batch = scheduler->get_batch();
    EXPECT_EQ(batch.size(), 3);

    scheduler->on_sequence_complete(batch[0]);

    scheduler->add_request(128);

    auto batch2 = scheduler->get_batch();
    EXPECT_EQ(batch2.size(), 3);
}

TEST_F(SchedulerTest, BlockManagerIntegration) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    auto& block_manager = scheduler->get_block_manager();
    ASSERT_NE(block_manager.get_kv_cache(), nullptr);

    scheduler->add_request(64);
    scheduler->add_request(64);

    auto stats = block_manager.get_kv_cache()->get_stats();
    EXPECT_GT(stats.allocated_blocks, 0);
}

TEST_F(SchedulerTest, SequenceManagerAccess) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    auto& seq_manager = scheduler->get_sequence_manager();

    int64_t seq1 = scheduler->add_request(64);
    int64_t seq2 = scheduler->add_request(64);

    auto running = seq_manager.get_running_sequences();
    EXPECT_EQ(running.size(), 2);
    EXPECT_TRUE(std::find(running.begin(), running.end(), seq1) != running.end());
    EXPECT_TRUE(std::find(running.begin(), running.end(), seq2) != running.end());
}

TEST_F(SchedulerTest, MaxSequenceLength) {
    auto scheduler = std::make_unique<Scheduler>(config_);

    int64_t seq_id = scheduler->add_request(config_.max_sequence_length);
    EXPECT_GE(seq_id, 0);

    auto seq = scheduler->get_block_manager().get_sequence(seq_id);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->max_tokens, config_.max_sequence_length);
}

}  // namespace cuda::inference::test
