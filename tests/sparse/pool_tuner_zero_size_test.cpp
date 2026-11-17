/**
 * @file pool_tuner_zero_size_test.cpp
 * @brief The adaptive pool tuner must not divide by a zero initial_pool_size
 *
 * should_grow() and should_shrink() both compute
 *
 *     float utilization = static_cast<float>(current_usage_) / config_.initial_pool_size;
 *
 * PoolTuningConfig is a public struct whose initial_pool_size defaults to 256 MB but can be set
 * to anything. At zero the division is inf: should_grow() answers true unconditionally and
 * should_shrink() can never answer true, so the tuner would grow the pool without limit and
 * never release anything.
 *
 * Both now return false when the configured size is zero — there is no utilisation to compare
 * against a pool that has no size.
 */

#include <cuda/memory_opt/memory_optimizer.h>

#include <gtest/gtest.h>

namespace pool_tuner_zero_size_test {

using cuda::memory_opt::AdaptiveMemoryPoolTuner;
using cuda::memory_opt::PoolTuningConfig;

namespace {

PoolTuningConfig zero_sized_config() {
    PoolTuningConfig config;
    config.initial_pool_size = 0;
    return config;
}

}  // namespace

TEST(PoolTunerZeroSizeTest, ShouldGrowDoesNotClaimGrowthFromAnUndefinedUsage) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();
    tuner.set_config(zero_sized_config());
    tuner.enable_adaptive_tuning();

    // Record enough allocation that a real pool would be considered over-utilised.
    tuner.record_allocation(1024);

    // Before the guard this was inf > 0.8 == true, so the tuner would keep growing a pool whose
    // configured size is zero: an unbounded response to an undefined ratio.
    EXPECT_FALSE(tuner.should_grow())
        << "there is no utilisation to compare against a zero-sized pool";
}

TEST(PoolTunerZeroSizeTest, ShouldShrinkDoesNotDecideFromAnUndefinedUsage) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();
    tuner.set_config(zero_sized_config());
    tuner.enable_adaptive_tuning();

    // should_shrink() also requires total_allocations_ >= 10 before it looks at the ratio.
    for (int i = 0; i < 12; ++i) {
        tuner.record_allocation(16);
    }

    // inf < threshold is false, so this one already answered false by accident. Asserting it
    // makes the answer intentional rather than incidental.
    EXPECT_FALSE(tuner.should_shrink())
        << "a zero-sized pool has no utilisation to shrink from";
}

TEST(PoolTunerZeroSizeTest, AConfiguredSizeStillDrivesBothDecisions) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();
    PoolTuningConfig config;
    config.initial_pool_size = 1024;
    tuner.set_config(config);
    tuner.enable_adaptive_tuning();

    // The guard must not disable the tuner for ordinary configurations: filling the pool past
    // 80% still asks for growth.
    tuner.record_allocation(900);
    EXPECT_TRUE(tuner.should_grow())
        << "900 of 1024 bytes is 88% and should request growth";
}

}  // namespace pool_tuner_zero_size_test
