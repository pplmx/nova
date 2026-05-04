#include <gtest/gtest.h>
#include "cuda/inference/speculative_decoding.h"
#include "cuda/inference/block_manager.h"

namespace cuda::inference {

class SpeculativeDecodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        BlockManagerConfig config;
        config.max_model_len = 2048;
        config.block_size = 16;
        config.num_gpu_blocks = 256;
        block_manager = std::make_unique<BlockManager>(config);

        SpeculativeDecodingConfig spec_config;
        spec_config.draft_depth = 4;
        spec_config.acceptance_threshold = 0.8f;
        spec_runner = std::make_unique<SpeculativeDecodingRunner>(
            block_manager.get(), spec_config);
    }

    std::unique_ptr<BlockManager> block_manager;
    std::unique_ptr<SpeculativeDecodingRunner> spec_runner;
};

TEST_F(SpeculativeDecodingTest, Construction) {
    EXPECT_EQ(spec_runner->get_config().draft_depth, 4);
    EXPECT_EQ(spec_runner->get_config().acceptance_threshold, 0.8f);
}

TEST_F(SpeculativeDecodingTest, Configure) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 6;
    config.enable_tree_attention = true;

    spec_runner->configure(config);

    EXPECT_EQ(spec_runner->get_config().draft_depth, 6);
    EXPECT_TRUE(spec_runner->get_config().enable_tree_attention);
}

TEST_F(SpeculativeDecodingTest, SnapshotRollbackCommit) {
    spec_runner->snapshot_kv_state();
    spec_runner->rollback_kv_state();
    spec_runner->commit_kv_state();
}

TEST(LogProbTracker, Record) {
    LogProbTracker tracker;
    tracker.record(42, -0.5f, -0.3f, true);
    tracker.record(17, -1.2f, -0.8f, false);

    EXPECT_EQ(tracker.num_accepted(), 1);
    EXPECT_EQ(tracker.num_rejected(), 1);
}

TEST(LogProbTracker, KLDivergence) {
    LogProbTracker tracker;
    tracker.record(1, -0.5f, -0.5f, true);
    tracker.record(2, -0.7f, -0.7f, true);

    float avg_kl = tracker.compute_average_kl_divergence();
    EXPECT_NEAR(avg_kl, 0.0f, 0.001f);
}

TEST(LogProbTracker, Clear) {
    LogProbTracker tracker;
    tracker.record(1, -0.5f, -0.3f, true);
    tracker.clear();

    EXPECT_EQ(tracker.num_accepted(), 0);
    EXPECT_EQ(tracker.get_history().size(), 0);
}

}  // namespace cuda::inference
