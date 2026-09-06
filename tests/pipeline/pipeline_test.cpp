#include <gtest/gtest.h>

#include <algorithm>
#include <utility>

#include "cuda/pipeline/pipeline_scheduler.h"
#include "cuda/pipeline/stage_balance.h"

using namespace cuda::pipeline;

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(PipelineTest, BubbleOverheadSingleStage) {
    ::cuda::nccl::NcclContext ctx;
    ctx.initialize();

    PipelineScheduler scheduler(ctx, 1, 8, 32);

    EXPECT_FLOAT_EQ(scheduler.bubble_overhead_percent(), 0.0f);
}

TEST_F(PipelineTest, BubbleOverheadMultipleStages) {
    ::cuda::nccl::NcclContext ctx;
    ctx.initialize();

    PipelineScheduler scheduler(ctx, 4, 16, 32);
    float overhead = scheduler.bubble_overhead_percent();

    EXPECT_GE(overhead, 0.0f);
    EXPECT_LE(overhead, 25.0f);
}

TEST_F(PipelineTest, RecommendMicrobatches) {
    int recommended = recommend_microbatches(4);
    EXPECT_GE(recommended, 16);
}

TEST_F(PipelineTest, ScheduleTypeEnum) {
    EXPECT_EQ(static_cast<int>(ScheduleType::OneForwardOneBackward), 0);
    EXPECT_EQ(static_cast<int>(ScheduleType::Interleaved), 1);
}

// The old schedule_1f1b dropped almost all work: the steady-state forward
// passed stage = step >= K (ignored by run_forward's bounds check) and the
// warmup only issued stage 0. The schedule must run every (stage, microbatch)
// exactly once for both directions, in a dependency-safe order.
TEST_F(PipelineTest, Schedule1F1BCoversEveryStageMicrobatchOnce) {
    ::cuda::nccl::NcclContext ctx;
    ctx.initialize();

    const int K = 4;
    const int M = 8;
    PipelineScheduler scheduler(ctx, K, M, 32);

    std::vector<std::pair<int, int>> forwards;
    std::vector<std::pair<int, int>> backwards;
    scheduler.set_forward_fn([&](int s, int mb) { forwards.emplace_back(s, mb); });
    scheduler.set_backward_fn([&](int s, int mb) { backwards.emplace_back(s, mb); });

    scheduler.set_schedule_type(ScheduleType::OneForwardOneBackward);
    scheduler.run();

    const size_t total = static_cast<size_t>(K) * M;
    ASSERT_EQ(forwards.size(), total);
    ASSERT_EQ(backwards.size(), total);

    for (int s = 0; s < K; ++s) {
        for (int mb = 0; mb < M; ++mb) {
            EXPECT_EQ(std::count(forwards.begin(), forwards.end(), std::make_pair(s, mb)), 1)
                << "forward (stage " << s << ", mb " << mb << ")";
            EXPECT_EQ(std::count(backwards.begin(), backwards.end(), std::make_pair(s, mb)), 1)
                << "backward (stage " << s << ", mb " << mb << ")";
        }
    }
}

TEST_F(PipelineTest, ScheduleInterleavedCoversEveryStageMicrobatchOnce) {
    ::cuda::nccl::NcclContext ctx;
    ctx.initialize();

    const int K = 3;
    const int M = 6;
    PipelineScheduler scheduler(ctx, K, M, 32);

    std::vector<std::pair<int, int>> forwards;
    std::vector<std::pair<int, int>> backwards;
    scheduler.set_forward_fn([&](int s, int mb) { forwards.emplace_back(s, mb); });
    scheduler.set_backward_fn([&](int s, int mb) { backwards.emplace_back(s, mb); });

    scheduler.set_schedule_type(ScheduleType::Interleaved);
    scheduler.run();

    const size_t total = static_cast<size_t>(K) * M;
    ASSERT_EQ(forwards.size(), total);
    ASSERT_EQ(backwards.size(), total);
    for (int s = 0; s < K; ++s) {
        for (int mb = 0; mb < M; ++mb) {
            EXPECT_EQ(std::count(forwards.begin(), forwards.end(), std::make_pair(s, mb)), 1);
            EXPECT_EQ(std::count(backwards.begin(), backwards.end(), std::make_pair(s, mb)), 1);
        }
    }
}

TEST_F(PipelineTest, StageBalanceValidator) {
    StageBalanceValidator validator;

    validator.profile_stage(0, 10.0f);
    validator.profile_stage(1, 10.5f);
    validator.profile_stage(2, 9.8f);
    validator.profile_stage(3, 10.2f);

    EXPECT_TRUE(validator.is_balanced());
    EXPECT_LT(validator.variance_percent(), 5.0f);
    EXPECT_FLOAT_EQ(validator.average_time_ms(), 10.125f);
}

TEST_F(PipelineTest, StageBalanceValidatorUnbalanced) {
    StageBalanceValidator validator;

    validator.profile_stage(0, 10.0f);
    validator.profile_stage(1, 15.0f);
    validator.profile_stage(2, 10.0f);

    EXPECT_FALSE(validator.is_balanced());
    EXPECT_GT(validator.variance_percent(), 20.0f);
}

TEST_F(PipelineTest, StageBalanceValidatorSuggestRebalance) {
    StageBalanceValidator validator;

    validator.profile_stage(0, 10.0f);
    validator.profile_stage(1, 15.0f);
    validator.profile_stage(2, 10.0f);

    if (!validator.is_balanced()) {
        auto suggestions = validator.suggest_rebalance();
        EXPECT_FALSE(suggestions.empty());
    }
}

TEST_F(PipelineTest, StageBalanceValidatorGetTime) {
    StageBalanceValidator validator;

    validator.profile_stage(0, 10.0f);
    validator.profile_stage(2, 15.0f);

    EXPECT_FLOAT_EQ(validator.get_time(0), 10.0f);
    EXPECT_FLOAT_EQ(validator.get_time(1), 0.0f);
    EXPECT_FLOAT_EQ(validator.get_time(2), 15.0f);
}

TEST_F(PipelineTest, StageBalanceValidatorReset) {
    StageBalanceValidator validator;

    validator.profile_stage(0, 10.0f);
    validator.reset();

    EXPECT_EQ(validator.stage_count(), 0u);
    EXPECT_TRUE(validator.is_balanced());
}
