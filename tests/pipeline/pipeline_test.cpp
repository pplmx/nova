#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "cuda/pipeline/pipeline_scheduler.h"
#include "cuda/pipeline/stage_balance.h"

using namespace cuda::pipeline;

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
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

// The constructor validates all three of its numeric arguments and refuses a bad one, but only two
// of them can be read back afterwards: num_stages() and num_microbatches() exist and
// microbatch_size() did not. A caller that hands the scheduler a shape -- or a test that wants to
// confirm what it was given -- had no way to ask what the object was actually built with, and
// microbatch_size_ was stored and then never read by anything.
TEST_F(PipelineTest, MicrobatchSizeIsReadableBack) {
    ::cuda::nccl::NcclContext ctx;
    ctx.initialize();

    PipelineScheduler scheduler(ctx, 4, 16, 32);

    EXPECT_EQ(scheduler.microbatch_size(), 32) << "the value the constructor validated must be readable back";
    EXPECT_EQ(scheduler.num_stages(), 4);
    EXPECT_EQ(scheduler.num_microbatches(), 16);
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
            EXPECT_EQ(std::count(forwards.begin(), forwards.end(), std::make_pair(s, mb)), 1) << "forward (stage " << s << ", mb " << mb << ")";
            EXPECT_EQ(std::count(backwards.begin(), backwards.end(), std::make_pair(s, mb)), 1) << "backward (stage " << s << ", mb " << mb << ")";
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

// ---------------------------------------------------------------------------
// StageBalanceValidator: three contract defects.
//
//  - profile_stage() with a negative index wrote before the vector: a negative
//    index passes `stage >= size()`, so no resize happened and the assignment
//    landed out of bounds. ASan: heap-buffer-overflow WRITE inside profile_stage.
//  - Unprofiled gaps in the stage range were averaged in as 0 ms, so two equal
//    10 ms stages profiled as 0 and 5 reported a mean of 3.33 ms, a variance of
//    141% and "unbalanced".
//  - A NaN time made variance_percent() return NaN; NaN < threshold is false, so
//    is_balanced() reported a confident "unbalanced" from an unvalidated input.
// ---------------------------------------------------------------------------

TEST_F(PipelineTest, StageBalanceRejectsANegativeStageIndex) {
    StageBalanceValidator validator;
    validator.profile_stage(0, 10.0f);
    validator.profile_stage(2, 12.0f);

    // Before the guard this wrote one float before the vector's first element.
    EXPECT_THROW(validator.profile_stage(-1, 5.0f), std::invalid_argument) << "a negative index is not a stage";
    EXPECT_THROW(validator.profile_stage(-100, 5.0f), std::invalid_argument);

    // And the rejection must not disturb what was already recorded.
    EXPECT_EQ(validator.stage_count(), 3u);
    EXPECT_FLOAT_EQ(validator.get_time(0), 10.0f);
    EXPECT_FLOAT_EQ(validator.get_time(2), 12.0f);
}

TEST_F(PipelineTest, StageBalanceRejectsNonFiniteAndNegativeTimes) {
    StageBalanceValidator validator;

    EXPECT_THROW(validator.profile_stage(0, -1.0f), std::invalid_argument) << "a negative duration is not a time";
    EXPECT_THROW(validator.profile_stage(0, std::nanf("")), std::invalid_argument) << "NaN must not reach the statistics";
    EXPECT_THROW(validator.profile_stage(0, std::numeric_limits<float>::infinity()), std::invalid_argument);
    EXPECT_THROW(validator.profile_stage(0, -std::numeric_limits<float>::infinity()), std::invalid_argument);

    // None of those may leave a half-recorded stage behind.
    EXPECT_EQ(validator.profiled_stage_count(), 0u);
    EXPECT_FLOAT_EQ(validator.variance_percent(), 0.0f);
}

// The statistics run over profiled stages only. Two equal 10 ms stages profiled as 0 and 5
// are perfectly balanced; averaging in the unprofiled gaps 1..4 made them look 141% apart.
TEST_F(PipelineTest, StageBalanceIgnoresUnprofiledGaps) {
    StageBalanceValidator validator;
    validator.profile_stage(0, 10.0f);
    validator.profile_stage(5, 10.0f);

    EXPECT_EQ(validator.stage_count(), 6u) << "the index range reaches past the highest stage profiled";
    EXPECT_EQ(validator.profiled_stage_count(), 2u) << "but only two stages were actually profiled";

    EXPECT_FLOAT_EQ(validator.average_time_ms(), 10.0f);
    EXPECT_FLOAT_EQ(validator.variance_percent(), 0.0f);
    EXPECT_TRUE(validator.is_balanced());
}

// A gap holds 0.0f, so it would otherwise be reported as the fastest stage.
TEST_F(PipelineTest, StageBalanceSlowestAndFastestIgnoreUnprofiledGaps) {
    StageBalanceValidator validator;
    validator.profile_stage(2, 20.0f);
    validator.profile_stage(4, 10.0f);

    EXPECT_EQ(validator.slowest_stage(), 2);
    EXPECT_EQ(validator.fastest_stage(), 4) << "stage 0 is unprofiled, not the fastest stage";
}

// get_time() on a gap must report "not profiled" consistently, which is 0.0f.
TEST_F(PipelineTest, StageBalanceGetTimeReportsZeroForAnUnprofiledGap) {
    StageBalanceValidator validator;
    validator.profile_stage(2, 7.0f);

    EXPECT_FLOAT_EQ(validator.get_time(0), 0.0f);
    EXPECT_FLOAT_EQ(validator.get_time(1), 0.0f);
    EXPECT_FLOAT_EQ(validator.get_time(2), 7.0f);
    EXPECT_FLOAT_EQ(validator.get_time(3), 0.0f);
}

// Re-profiling replaces; it does not add a stage.
TEST_F(PipelineTest, StageBalanceReprofilingReplacesTheTime) {
    StageBalanceValidator validator;
    validator.profile_stage(0, 10.0f);
    validator.profile_stage(0, 20.0f);

    EXPECT_EQ(validator.profiled_stage_count(), 1u);
    EXPECT_FLOAT_EQ(validator.get_time(0), 20.0f);
    EXPECT_FLOAT_EQ(validator.average_time_ms(), 20.0f);
}

// reset() must clear the profiled flags too, or a later query would read stale ones.
TEST_F(PipelineTest, StageBalanceResetClearsTheProfiledState) {
    StageBalanceValidator validator;
    validator.profile_stage(0, 10.0f);
    validator.profile_stage(1, 30.0f);
    ASSERT_EQ(validator.profiled_stage_count(), 2u);

    validator.reset();

    EXPECT_EQ(validator.stage_count(), 0u);
    EXPECT_EQ(validator.profiled_stage_count(), 0u);
    EXPECT_FLOAT_EQ(validator.variance_percent(), 0.0f);
    EXPECT_TRUE(validator.is_balanced()) << "an empty profile has nothing to be unbalanced about";

    // And the validator must be usable again afterwards.
    validator.profile_stage(0, 5.0f);
    EXPECT_EQ(validator.profiled_stage_count(), 1u);
    EXPECT_FLOAT_EQ(validator.get_time(0), 5.0f);
}

// ---------------------------------------------------------------------------
// PipelineScheduler accepted any counts. Each bad combination produced a
// confident wrong number instead of an error:
//
//   num_microbatches = 0  -> bubble_overhead_percent() = 100% for a pipeline
//                            that would run nothing
//   num_stages < 0        -> 0%, because the `num_stages_ <= 1` early return
//                            swallowed it, while run() iterated an empty range
//                            and still set completed_ = true
// ---------------------------------------------------------------------------

TEST_F(PipelineTest, SchedulerRejectsANonPositiveStageCount) {
    auto& ctx = cuda::nccl::NcclContext::instance();

    EXPECT_THROW(PipelineScheduler(ctx, 0, 8, 1), std::invalid_argument);
    EXPECT_THROW(PipelineScheduler(ctx, -3, 8, 1), std::invalid_argument);
    EXPECT_NO_THROW(PipelineScheduler(ctx, 1, 8, 1));
}

TEST_F(PipelineTest, SchedulerRejectsANonPositiveMicrobatchCount) {
    auto& ctx = cuda::nccl::NcclContext::instance();

    EXPECT_THROW(PipelineScheduler(ctx, 4, 0, 1), std::invalid_argument) << "zero microbatches would report a 100% bubble";
    EXPECT_THROW(PipelineScheduler(ctx, 4, -2, 1), std::invalid_argument);
    EXPECT_NO_THROW(PipelineScheduler(ctx, 4, 1, 1));
}

TEST_F(PipelineTest, SchedulerRejectsANonPositiveMicrobatchSize) {
    auto& ctx = cuda::nccl::NcclContext::instance();

    EXPECT_THROW(PipelineScheduler(ctx, 4, 8, 0), std::invalid_argument);
    EXPECT_THROW(PipelineScheduler(ctx, 4, 8, -1), std::invalid_argument);
}

// The bubble estimate itself, on inputs that are now guaranteed valid.
TEST_F(PipelineTest, SchedulerBubbleOverheadMatchesThe1F1BFormula) {
    auto& ctx = cuda::nccl::NcclContext::instance();

    {
        PipelineScheduler scheduler(ctx, 4, 8, 1);
        // (K - 1) / (M + K - 1) = 3 / 11
        EXPECT_NEAR(scheduler.bubble_overhead_percent(), 300.0f / 11.0f, 1e-4f);
    }
    {
        PipelineScheduler scheduler(ctx, 1, 8, 1);
        EXPECT_FLOAT_EQ(scheduler.bubble_overhead_percent(), 0.0f) << "a single stage has no bubble";
    }
    {
        // More microbatches amortise the same fixed fill/drain cost.
        PipelineScheduler narrower(ctx, 4, 8, 1);
        PipelineScheduler wider(ctx, 4, 64, 1);
        EXPECT_LT(wider.bubble_overhead_percent(), narrower.bubble_overhead_percent());
    }
}

// Every (stage, microbatch) pair must run forward once and backward once: that coverage is
// the scheduler's whole contract, and the comment in schedule_1f1b() records that an earlier
// version silently dropped most of it.
TEST_F(PipelineTest, SchedulerRunsEveryStageAndMicrobatchOnce) {
    auto& ctx = cuda::nccl::NcclContext::instance();
    constexpr int stages = 4;
    constexpr int microbatches = 3;

    std::vector<int> forward_calls(static_cast<size_t>(stages) * microbatches, 0);
    std::vector<int> backward_calls(static_cast<size_t>(stages) * microbatches, 0);

    PipelineScheduler scheduler(ctx, stages, microbatches, 1);
    scheduler.set_forward_fn([&](int stage, int mb) { ++forward_calls[static_cast<size_t>(stage) * microbatches + mb]; });
    scheduler.set_backward_fn([&](int stage, int mb) { ++backward_calls[static_cast<size_t>(stage) * microbatches + mb]; });
    scheduler.run();

    for (int stage = 0; stage < stages; ++stage) {
        for (int mb = 0; mb < microbatches; ++mb) {
            const size_t i = static_cast<size_t>(stage) * microbatches + mb;
            EXPECT_EQ(forward_calls[i], 1) << "forward at stage " << stage << ", microbatch " << mb;
            EXPECT_EQ(backward_calls[i], 1) << "backward at stage " << stage << ", microbatch " << mb;
        }
    }
}

// With no callbacks the schedule does nothing, but it must still report completion rather
// than leaving wait() to block on an event that was never recorded.
TEST_F(PipelineTest, SchedulerCompletesWhenNoCallbacksAreSet) {
    auto& ctx = cuda::nccl::NcclContext::instance();
    PipelineScheduler scheduler(ctx, 4, 8, 1);

    EXPECT_NO_THROW(scheduler.run());
    EXPECT_NO_THROW(scheduler.wait());
}

// The recommendation helper must keep scaling with the stage count and never fall below its
// own floor.
TEST_F(PipelineTest, RecommendMicrobatchesScalesWithStages) {
    EXPECT_GE(recommend_microbatches(1), 16);
    EXPECT_GE(recommend_microbatches(4), 16);
    EXPECT_GT(recommend_microbatches(8), recommend_microbatches(4));
    EXPECT_GE(recommend_microbatches(2), recommend_microbatches(1)) << "a deeper pipeline needs at least as many microbatches";
}
