#include <gtest/gtest.h>
#include "cuda/inference/speculative_decoding.h"
#include "cuda/inference/block_manager.h"

namespace cuda::inference {

class SpeculativeDecodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);

        BlockManagerConfig config;
        config.max_model_len = 2048;
        config.block_size = 16;
        config.num_gpu_blocks = 32;
        try {
            block_manager = std::make_unique<BlockManager>(config);
        } catch (const std::exception&) {
            GTEST_SKIP() << "BlockManager allocation failed";
        }

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

TEST(SpeculativeDecodingConfig, DefaultValues) {
    SpeculativeDecodingConfig config;

    EXPECT_EQ(config.draft_depth, 4);
    EXPECT_EQ(config.acceptance_threshold, 0.8f);
    EXPECT_TRUE(config.enable_tree_attention);
    EXPECT_TRUE(config.enable_async_draft);
    EXPECT_FALSE(config.enable_eagle3);
    EXPECT_FALSE(config.enable_xgrammar);
    EXPECT_EQ(config.max_draft_depth, 8);
    EXPECT_EQ(config.vocab_size, 0);
    EXPECT_EQ(config.temperature, 0.8f);
}

TEST(SpeculativeDecodingConfig, CustomValues) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 6;
    config.acceptance_threshold = 0.9f;
    config.enable_tree_attention = false;
    config.enable_async_draft = false;
    config.enable_eagle3 = true;
    config.enable_xgrammar = true;
    config.max_draft_depth = 16;
    config.vocab_size = 50000;
    config.temperature = 0.5f;

    EXPECT_EQ(config.draft_depth, 6);
    EXPECT_EQ(config.acceptance_threshold, 0.9f);
    EXPECT_FALSE(config.enable_tree_attention);
    EXPECT_FALSE(config.enable_async_draft);
    EXPECT_TRUE(config.enable_eagle3);
    EXPECT_TRUE(config.enable_xgrammar);
    EXPECT_EQ(config.max_draft_depth, 16);
    EXPECT_EQ(config.vocab_size, 50000);
    EXPECT_EQ(config.temperature, 0.5f);
}

TEST_F(SpeculativeDecodingTest, KVSnapshotCapture) {
    auto* seq = block_manager->create_sequence(100, 128);
    int64_t seq_id = seq->id;
    block_manager->append_tokens(seq_id, 4);

    spec_runner->snapshot_kv_state();

    auto captured = spec_runner->get_config();
    EXPECT_EQ(captured.draft_depth, 4);
}

TEST_F(SpeculativeDecodingTest, KVSnapshotRollback) {
    auto* seq1 = block_manager->create_sequence(200, 128);
    int64_t seq1_id = seq1->id;
    block_manager->append_tokens(seq1_id, 8);

    auto* seq2 = block_manager->create_sequence(201, 128);
    int64_t seq2_id = seq2->id;
    block_manager->append_tokens(seq2_id, 4);

    int before_rollback = static_cast<int>(block_manager->get_active_sequences().size());

    spec_runner->snapshot_kv_state();
    spec_runner->rollback_kv_state();

    int after_rollback = static_cast<int>(block_manager->get_active_sequences().size());
    EXPECT_LT(after_rollback, before_rollback);
}

TEST_F(SpeculativeDecodingTest, KVCommitWithoutRollback) {
    auto* seq = block_manager->create_sequence(300, 128);
    int64_t seq_id = seq->id;
    block_manager->append_tokens(seq_id, 4);

    spec_runner->snapshot_kv_state();
    spec_runner->commit_kv_state();

    auto& tracker = spec_runner->get_logprob_tracker();
    EXPECT_EQ(tracker.num_accepted(), 0);
}

TEST_F(SpeculativeDecodingTest, AcceptanceRatioEpsilonGuard) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 2;
    config.vocab_size = 1024;
    config.temperature = 0.0f;
    spec_runner->configure(config);

    const int vocab_size = 1024;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    // Stage on the host and upload: the previous code wrote to
    // Buffer::data() (device memory) directly from the CPU, which segfaults.
    std::vector<float> h_draft(vocab_size, 0.0f);
    std::vector<float> h_target(vocab_size, 0.0f);
    h_target[0] = 0.5f;
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {0, 1};

    stream::Stream stream;
    auto result = spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    EXPECT_EQ(static_cast<int>(result.tokens.size()), 2);
}

TEST_F(SpeculativeDecodingTest, AcceptanceRatioFullAccept) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 1;
    config.acceptance_threshold = 0.0f;
    config.vocab_size = 512;
    config.temperature = 0.0f;
    spec_runner->configure(config);

    const int vocab_size = 512;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    std::vector<float> h_draft(vocab_size, -100.0f);
    std::vector<float> h_target(vocab_size, -100.0f);
    h_draft[5] = 1.0f;
    h_target[5] = 1.0f;
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {5};

    stream::Stream stream;
    auto result = spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    EXPECT_EQ(result.num_accepted, 1);
    EXPECT_TRUE(result.tokens[0].accepted);
}

TEST_F(SpeculativeDecodingTest, AcceptanceRatioFullReject) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 1;
    config.acceptance_threshold = 0.99f;
    config.vocab_size = 512;
    config.temperature = 0.0f;
    spec_runner->configure(config);

    const int vocab_size = 512;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    std::vector<float> h_draft(vocab_size, -100.0f);
    std::vector<float> h_target(vocab_size, -100.0f);
    h_draft[10] = 1.0f;
    h_target[11] = 1.0f;  // target probability peaks at a different token, so
                          // token 10 must be rejected (acceptance ratio ~0).
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {10};

    stream::Stream stream;
    auto result = spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    EXPECT_EQ(result.num_accepted, 0);
    EXPECT_FALSE(result.tokens[0].accepted);
}

TEST_F(SpeculativeDecodingTest, LogProbabilityTracking) {
    const int vocab_size = 256;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    std::vector<float> h_draft(vocab_size, -10.0f);
    std::vector<float> h_target(vocab_size, -10.0f);
    h_draft[7] = 0.0f;
    h_target[7] = 0.0f;
    h_draft[8] = 0.0f;
    // target[8] must match draft so both tokens are accepted under the default
    // 0.8 acceptance threshold (target=-1 here gives a ~0.54 ratio -> reject,
    // contradicting the num_accepted==2 assertion).
    h_target[8] = 0.0f;
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {7, 8};

    stream::Stream stream;
    spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    auto& tracker = spec_runner->get_logprob_tracker();
    EXPECT_EQ(tracker.num_accepted(), 2);
    EXPECT_EQ(static_cast<int>(tracker.get_history().size()), 2);
    EXPECT_EQ(tracker.get_history()[0].token_id, 7);
    EXPECT_EQ(tracker.get_history()[1].token_id, 8);
}

TEST_F(SpeculativeDecodingTest, LogProbabilityTrackingMixed) {
    SpeculativeDecodingConfig config;
    config.acceptance_threshold = 0.5f;
    spec_runner->configure(config);

    const int vocab_size = 256;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    std::vector<float> h_draft(vocab_size, -10.0f);
    std::vector<float> h_target(vocab_size, -10.0f);
    h_draft[1] = 0.0f;
    h_target[1] = 0.0f;
    h_draft[2] = 0.0f;
    h_target[2] = -100.0f;
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {1, 2};

    stream::Stream stream;
    spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    auto& tracker = spec_runner->get_logprob_tracker();
    EXPECT_EQ(tracker.num_accepted() + tracker.num_rejected(), 2);
}

TEST_F(SpeculativeDecodingTest, KLDivergenceComputation) {
    const int vocab_size = 256;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    std::vector<float> h_draft(vocab_size, -10.0f);
    std::vector<float> h_target(vocab_size, -10.0f);
    h_draft[3] = 0.0f;
    h_target[3] = 0.0f;
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {3};

    stream::Stream stream;
    auto result = spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    EXPECT_GE(result.kl_divergence, 0.0f);
}

TEST_F(SpeculativeDecodingTest, TreeAttentionMaskApplication) {
    SpeculativeDecodingConfig config;
    config.enable_tree_attention = true;
    config.draft_depth = 3;
    spec_runner->configure(config);

    stream::Stream stream;
    spec_runner->apply_tree_attention_mask(3, stream);

    auto cfg = spec_runner->get_config();
    EXPECT_TRUE(cfg.enable_tree_attention);
}

TEST_F(SpeculativeDecodingTest, TreeAttentionDisabled) {
    SpeculativeDecodingConfig config;
    config.enable_tree_attention = false;
    config.draft_depth = 3;
    spec_runner->configure(config);

    stream::Stream stream;
    spec_runner->apply_tree_attention_mask(3, stream);

    auto cfg = spec_runner->get_config();
    EXPECT_FALSE(cfg.enable_tree_attention);
}

TEST_F(SpeculativeDecodingTest, DraftTokensGenerationCount) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 5;
    config.vocab_size = 1024;
    config.temperature = 0.0f;
    spec_runner->configure(config);

    auto* seq = block_manager->create_sequence(999, 256);
    int64_t seq_id = seq->id;
    block_manager->append_tokens(seq_id, 32);

    memory::Buffer<float> embeddings(512);
    stream::Stream stream;

    auto draft_tokens = spec_runner->generate_draft_tokens(embeddings, 32, stream,
        [](memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&) {});
    (void)draft_tokens;
}

TEST_F(SpeculativeDecodingTest, TemperatureZeroSampling) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 1;
    config.vocab_size = 128;
    config.temperature = 0.0f;
    spec_runner->configure(config);

    auto* seq = block_manager->create_sequence(888, 256);
    (void)seq;

    memory::Buffer<float> embeddings(512);
    stream::Stream stream;

    auto forward_fn = [](memory::Buffer<float>& logits, const std::vector<int64_t>&, bool, const stream::Stream&) {
        // Stage on the host and upload: logits is a device buffer, so writing
        // logits.data() from the CPU segfaults.
        std::vector<float> h(128);
        for (int i = 0; i < 128; ++i) {
            h[i] = static_cast<float>(i);
        }
        logits.copy_from(h.data(), 128);
    };

    auto draft_tokens = spec_runner->generate_draft_tokens(embeddings, 32, stream, forward_fn);
    EXPECT_EQ(static_cast<int>(draft_tokens.size()), 1);
    EXPECT_GE(draft_tokens[0], 0);
    EXPECT_LT(draft_tokens[0], 128);
}

TEST_F(SpeculativeDecodingTest, TemperaturePositiveSampling) {
    SpeculativeDecodingConfig config;
    config.draft_depth = 2;
    config.vocab_size = 128;
    config.temperature = 1.0f;
    spec_runner->configure(config);

    auto* seq = block_manager->create_sequence(777, 256);
    (void)seq;

    memory::Buffer<float> embeddings(512);
    stream::Stream stream;

    auto forward_fn = [](memory::Buffer<float>& logits, const std::vector<int64_t>&, bool, const stream::Stream&) {
        std::vector<float> h(128, 1.0f);
        logits.copy_from(h.data(), 128);
    };

    auto draft_tokens = spec_runner->generate_draft_tokens(embeddings, 32, stream, forward_fn);
    EXPECT_EQ(static_cast<int>(draft_tokens.size()), 2);
    for (int tok : draft_tokens) {
        EXPECT_GE(tok, 0);
        EXPECT_LT(tok, 128);
    }
}

TEST_F(SpeculativeDecodingTest, VerifyOutOfBoundsToken) {
    const int vocab_size = 256;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    // Stage on the host and upload (device writes from the CPU segfault).
    std::vector<float> h_draft(vocab_size, 0.0f);
    std::vector<float> h_target(vocab_size, 0.0f);
    draft_logits.copy_from(h_draft.data(), vocab_size);
    target_logits.copy_from(h_target.data(), vocab_size);

    std::vector<int> draft_tokens = {999};

    stream::Stream stream;
    auto result = spec_runner->verify_draft_tokens(draft_tokens, draft_logits, target_logits, stream);

    EXPECT_EQ(static_cast<int>(result.tokens.size()), 0);
}

TEST_F(SpeculativeDecodingTest, ReconfigureUpdatesConfig) {
    SpeculativeDecodingConfig c1;
    c1.draft_depth = 3;
    c1.temperature = 0.3f;
    spec_runner->configure(c1);

    SpeculativeDecodingConfig c2;
    c2.draft_depth = 7;
    c2.temperature = 1.2f;
    spec_runner->configure(c2);

    auto cfg = spec_runner->get_config();
    EXPECT_EQ(cfg.draft_depth, 7);
    EXPECT_EQ(cfg.temperature, 1.2f);
}

}  // namespace cuda::inference
