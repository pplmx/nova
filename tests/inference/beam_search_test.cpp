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

}  // namespace cuda::inference
