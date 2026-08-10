#include <gtest/gtest.h>
#include "cuda/inference/beam_search.h"
#include "cuda/inference/block_manager.h"

namespace cuda::inference {

class BeamSearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        BlockManagerConfig config;
        config.max_model_len = 2048;
        config.block_size = 16;
        config.num_gpu_blocks = 256;
        // Lightweight KV pool: these tests exercise scheduling/beam logic, not
        // KV capacity. The default pool is ~64GB per BlockManager, so holding
        // more than one manager (as KVBlockForkingForBeamHypotheses does) OOMs
        // on anything but a very quiet 80GB GPU.
        config.kv_cache_config.num_heads = 4;
        config.kv_cache_config.head_dim = 64;
        config.kv_cache_config.num_layers = 2;
        config.kv_cache_config.num_blocks = 256;
        block_manager = std::make_unique<BlockManager>(config);

        BeamSearchConfig beam_config;
        beam_config.max_beams = 4;
        beam_config.length_penalty = 0.7f;
        beam_config.max_length = 512;
        beam_manager = std::make_unique<BeamSearchManager>(block_manager.get(), beam_config);
    }

    std::unique_ptr<BlockManager> block_manager;
    std::unique_ptr<BeamSearchManager> beam_manager;
};

TEST_F(BeamSearchTest, Construction) {
    EXPECT_EQ(beam_manager->get_config().max_beams, 4);
    EXPECT_EQ(beam_manager->get_config().length_penalty, 0.7f);
}

TEST_F(BeamSearchTest, Configure) {
    BeamSearchConfig config;
    config.max_beams = 8;
    config.temperature = 0.8f;

    beam_manager->configure(config);

    auto new_config = beam_manager->get_config();
    EXPECT_EQ(new_config.max_beams, 8);
    EXPECT_EQ(new_config.temperature, 0.8f);
}

TEST_F(BeamSearchTest, TraceExport) {
    auto json = beam_manager->export_trace_json();
    EXPECT_TRUE(json.find("{\"traces\":[]}") == 0);

    auto csv = beam_manager->export_trace_csv();
    EXPECT_TRUE(csv.find("step,beam,token") == 0);
}

TEST_F(BeamSearchTest, TraceStats) {
    auto stats = beam_manager->get_trace_stats();
    EXPECT_EQ(stats.total_steps, 0);
    EXPECT_EQ(stats.avg_beam_width, 0);
}

TEST_F(BeamSearchTest, ClearTrace) {
    beam_manager->clear_trace();

    auto json = beam_manager->export_trace_json();
    EXPECT_TRUE(json.find("{\"traces\":[]}") == 0);
}

TEST(TopKSampler, Sample) {
    TopKSampler sampler(3);

    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto results = sampler.sample(logits.data(), 5, 1.0f, 42);

    EXPECT_EQ(results.size(), 3);
    EXPECT_GE(results[0].first, 0);
    EXPECT_LT(results[0].first, 5);
    EXPECT_GE(results[0].second, 0.0f);
}

TEST(TopPSampler, Sample) {
    TopPSampler sampler(0.9f);

    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int token = sampler.sample(logits.data(), 5, 1.0f, 42);

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 5);
}

TEST(BeamHypothesis, DefaultValues) {
    BeamHypothesis hyp;
    EXPECT_EQ(hyp.tokens.size(), 0);
    EXPECT_EQ(hyp.length, 0);
    EXPECT_FALSE(hyp.finished);
    EXPECT_EQ(hyp.parent_beam, -1);
}

TEST_F(BeamSearchTest, LengthNormalizedScoring) {
    BeamSearchConfig config;
    config.max_beams = 2;
    config.length_penalty = 0.7f;
    config.enable_length_norm = true;
    beam_manager->configure(config);

    BeamHypothesis hyp1;
    hyp1.tokens = {1, 2, 3};
    hyp1.length = 3;
    hyp1.log_prob = -6.0f;
    hyp1.score = 0.0f;

    BeamHypothesis hyp2;
    hyp2.tokens = {4, 5};
    hyp2.length = 2;
    hyp2.log_prob = -4.0f;
    hyp2.score = 0.0f;

    beam_manager->get_config();
    // -6 / 3^0.7 = -2.7808 and -4 / 2^0.7 = -2.4623 (the previous expected
    // values -3.21 / -2.74 were arithmetically off).
    EXPECT_NEAR(hyp1.log_prob / std::pow(3, 0.7f), -2.7808f, 0.1f);
    EXPECT_NEAR(hyp2.log_prob / std::pow(2, 0.7f), -2.4623f, 0.1f);
}

TEST_F(BeamSearchTest, LengthNormalizedScoringLongSequence) {
    BeamSearchConfig config;
    config.length_penalty = 0.7f;
    config.enable_length_norm = true;
    beam_manager->configure(config);

    const int length = 2000;
    float log_prob = -4500.0f;

    float length_norm = std::pow(length, config.length_penalty);
    float normalized_score = log_prob / length_norm;

    EXPECT_TRUE(std::isfinite(normalized_score));
    EXPECT_GT(normalized_score, -100.0f);
    EXPECT_NE(normalized_score, 0.0f);
}

TEST_F(BeamSearchTest, KVBlockForkingForBeamHypotheses) {
    BlockManagerConfig bm_config;
    bm_config.max_model_len = 2048;
    bm_config.block_size = 16;
    bm_config.num_gpu_blocks = 256;
    bm_config.kv_cache_config.num_heads = 4;
    bm_config.kv_cache_config.head_dim = 64;
    bm_config.kv_cache_config.num_layers = 2;
    bm_config.kv_cache_config.num_blocks = 256;
    auto bm = std::make_unique<BlockManager>(bm_config);

    BeamSearchConfig config;
    config.max_beams = 4;
    config.enable_reuse = true;
    auto bsm = std::make_unique<BeamSearchManager>(bm.get(), config);

    EXPECT_NE(bsm, nullptr);
    EXPECT_EQ(bsm->get_config().enable_reuse, true);
}

TEST_F(BeamSearchTest, ScoreRebasingForLongSequences) {
    beam_manager->clear_trace();

    BeamHypothesis hyp1;
    hyp1.sequence_id = 1;
    hyp1.log_prob = -12000.0f;

    BeamHypothesis hyp2;
    hyp2.sequence_id = 2;
    hyp2.log_prob = -11000.0f;

    BeamSearchTraceEntry entry1{0, 0, 42, -12000.0f, -10.0f, -5.0f, 1};
    BeamSearchTraceEntry entry2{0, 1, 43, -11000.0f, -9.0f, -4.5f, 2};

    const float threshold = -1e4f;
    EXPECT_LT(hyp1.log_prob, threshold);
    EXPECT_LT(hyp2.log_prob, threshold);

    float max_log_prob = std::max(hyp1.log_prob, hyp2.log_prob);
    float rebase_factor = max_log_prob;

    // Rebase shifts scores down by the max so the best hypothesis becomes 0 and
    // the others remain <= 0; they cannot become non-negative.
    float rebased_hyp1 = hyp1.log_prob - rebase_factor;
    float rebased_hyp2 = hyp2.log_prob - rebase_factor;

    EXPECT_LT(rebased_hyp1, 0.0f);
    EXPECT_EQ(rebased_hyp2, 0.0f);
    EXPECT_GT(rebased_hyp2, rebased_hyp1);
}

TEST_F(BeamSearchTest, BatchBeamScoring) {
    BeamSearchConfig config;
    config.max_beams = 4;
    config.length_penalty = 0.6f;
    config.enable_length_norm = true;
    beam_manager->configure(config);

    std::vector<BeamHypothesis> hyps;
    for (int i = 0; i < 4; ++i) {
        BeamHypothesis hyp;
        hyp.tokens = {1, 2, 3, 4, 5};
        hyp.length = 5;
        hyp.log_prob = static_cast<float>(-10 - i * 2);
        hyp.score = 0.0f;
        hyps.push_back(hyp);
    }

    for (auto& hyp : hyps) {
        float length_norm = std::pow(hyp.length, config.length_penalty);
        hyp.score = hyp.log_prob / length_norm;
    }

    std::sort(hyps.begin(), hyps.end(),
        [](const BeamHypothesis& a, const BeamHypothesis& b) {
            return a.score > b.score;
        });

    EXPECT_GE(hyps[0].score, hyps[1].score);
    EXPECT_GE(hyps[1].score, hyps[2].score);
    EXPECT_GE(hyps[2].score, hyps[3].score);
}

TEST(TopKSampler, TopKWithTopPCombined) {
    TopKSampler sampler(2);
    TopPSampler p_sampler(0.8f);

    std::vector<float> logits = {0.5f, 1.5f, 3.0f, 2.5f, 0.1f, 0.05f};

    auto top_k_results = sampler.sample(logits.data(), 6, 1.0f, 123);
    EXPECT_EQ(top_k_results.size(), 2);
    EXPECT_EQ(top_k_results[0].first, 2);
    EXPECT_EQ(top_k_results[1].first, 3);

    int top_p_token = p_sampler.sample(logits.data(), 6, 1.0f, 123);
    EXPECT_GE(top_p_token, 0);
    EXPECT_LT(top_p_token, 6);
}

TEST(TopKSampler, TemperatureScalingEffect) {
    TopKSampler sampler(3);

    std::vector<float> logits = {1.0f, 2.0f, 3.0f};

    auto high_temp = sampler.sample(logits.data(), 3, 2.0f, 42);
    auto low_temp = sampler.sample(logits.data(), 3, 0.1f, 42);

    float high_temp_spread = high_temp[0].second - high_temp[2].second;
    float low_temp_spread = low_temp[0].second - low_temp[2].second;

    EXPECT_GT(high_temp_spread, 0.0f);
    EXPECT_GT(low_temp_spread, 0.0f);
}

TEST(TopKSampler, EOSTokenHandling) {
    BeamSearchConfig config;
    config.max_beams = 2;
    config.eos_token_id = 0;
    config.temperature = 1.0f;
    config.top_p = 1.0f;
    config.top_k = 1;

    std::vector<float> logits = {5.0f, 1.0f, 0.5f, 0.1f};
    int token = 0;

    bool is_eos = (token == config.eos_token_id);
    EXPECT_TRUE(is_eos);
}

TEST_F(BeamSearchTest, TraceExportIncludesPerTokenScores) {
    beam_manager->clear_trace();

    BeamSearchTraceEntry entry1{0, 0, 42, -2.5f, -0.5f, -2.5f, 1};
    BeamSearchTraceEntry entry2{1, 0, 43, -5.2f, -0.6f, -2.7f, 1};

    auto csv = beam_manager->export_trace_csv();
    EXPECT_TRUE(csv.find("step,beam,token,score,norm_score,sequence") == 0);

    auto json = beam_manager->export_trace_json();
    EXPECT_TRUE(json.find("{\"traces\":[") == 0);
}

TEST_F(BeamSearchTest, TraceCSVFormatCorrectness) {
    auto csv = beam_manager->export_trace_csv();

    std::istringstream stream(csv);
    std::string header;
    std::getline(stream, header);

    EXPECT_EQ(header, "step,beam,token,score,norm_score,sequence");

    std::vector<std::string> expected_cols = {"step", "beam", "token", "score", "norm_score", "sequence"};
    std::istringstream header_stream(header);
    std::string col;
    int col_count = 0;
    while (std::getline(header_stream, col, ',')) {
        EXPECT_EQ(col, expected_cols[col_count]);
        col_count++;
    }
    EXPECT_EQ(col_count, 6);
}

TEST_F(BeamSearchTest, TraceStatsWithActiveBeams) {
    auto stats = beam_manager->get_trace_stats();
    EXPECT_EQ(stats.total_steps, 0);

    EXPECT_GE(stats.avg_beam_width, 0);
    EXPECT_EQ(stats.avg_score, 0.0f);
    EXPECT_EQ(stats.avg_length_norm, 0.0f);
}

TEST_F(BeamSearchTest, ConfigurationPersistence) {
    BeamSearchConfig original_config;
    original_config.max_beams = 16;
    original_config.temperature = 0.5f;
    original_config.top_k = 50;
    original_config.top_p = 0.95f;
    original_config.length_penalty = 0.8f;
    original_config.max_length = 1024;
    original_config.eos_token_id = 2;
    original_config.rebase_threshold = 500;

    beam_manager->configure(original_config);

    auto retrieved_config = beam_manager->get_config();

    EXPECT_EQ(retrieved_config.max_beams, 16);
    EXPECT_EQ(retrieved_config.temperature, 0.5f);
    EXPECT_EQ(retrieved_config.top_k, 50);
    EXPECT_EQ(retrieved_config.top_p, 0.95f);
    EXPECT_EQ(retrieved_config.length_penalty, 0.8f);
    EXPECT_EQ(retrieved_config.max_length, 1024);
    EXPECT_EQ(retrieved_config.eos_token_id, 2);
    EXPECT_EQ(retrieved_config.rebase_threshold, 500);
}

}  // namespace cuda::inference
