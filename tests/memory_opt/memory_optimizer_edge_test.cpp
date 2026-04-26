#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/memory_opt/memory_optimizer.h>

namespace cuda::memory_opt::test {

class MemoryOptimizerEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MemoryOptimizerEdgeCaseTest, CompressorDisabled) {
    auto& compressor = CheckpointCompressor::instance();

    CompressionConfig config;
    config.enable_compression = false;
    compressor.set_config(config);

    std::vector<float> input(64, 1.0f);
    std::vector<float> output(64);

    size_t compressed_size = compressor.compress(
        input.data(), input.size() * sizeof(float),
        output.data(), output.size() * sizeof(float));

    EXPECT_EQ(compressed_size, input.size() * sizeof(float));
}

TEST_F(MemoryOptimizerEdgeCaseTest, CompressorBelowMinSize) {
    auto& compressor = CheckpointCompressor::instance();

    CompressionConfig config;
    config.enable_compression = true;
    config.min_size_for_compression = 1024;
    compressor.set_config(config);

    std::vector<float> input(64, 1.0f);
    std::vector<float> output(64);

    size_t compressed_size = compressor.compress(
        input.data(), input.size() * sizeof(float),
        output.data(), output.size() * sizeof(float));

    EXPECT_EQ(compressed_size, input.size() * sizeof(float));
}

TEST_F(MemoryOptimizerEdgeCaseTest, CompressorOutputTooSmall) {
    auto& compressor = CheckpointCompressor::instance();

    CompressionConfig config;
    config.enable_compression = false;
    compressor.set_config(config);

    std::vector<float> input(1024, 1.0f);
    std::vector<float> output(64);

    size_t compressed_size = compressor.compress(
        input.data(), input.size() * sizeof(float),
        output.data(), output.size() * sizeof(float));

    EXPECT_EQ(compressed_size, 0);
}

TEST_F(MemoryOptimizerEdgeCaseTest, AdaptiveTunerZeroAllocations) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();

    size_t suggested = tuner.suggest_pool_size();
    EXPECT_GT(suggested, 0);

    EXPECT_FALSE(tuner.should_grow());
    EXPECT_FALSE(tuner.should_shrink());
}

TEST_F(MemoryOptimizerEdgeCaseTest, AdaptiveTunerManySmallAllocations) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();

    for (int i = 0; i < 1000; ++i) {
        tuner.record_allocation(64);
        tuner.record_deallocation(64);
    }

    auto stats = tuner.get_stats();
    EXPECT_EQ(stats.total_allocations, 1000);
    EXPECT_EQ(stats.total_deallocations, 1000);
}

TEST_F(MemoryOptimizerEdgeCaseTest, AdaptiveTunerAlternatingSizes) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();

    for (int i = 0; i < 10; ++i) {
        tuner.record_allocation(1024);
        tuner.record_allocation(1024 * 1024);
    }

    auto stats = tuner.get_stats();
    EXPECT_GT(stats.average_allocation_size, 0);
}

TEST_F(MemoryOptimizerEdgeCaseTest, AdaptiveTunerProfileDetectionEdgeCases) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();
    tuner.reset_stats();

    tuner.record_allocation(8 * 1024 * 1024);
    auto profile_inference = tuner.detect_workload_profile();
    EXPECT_EQ(profile_inference, WorkloadProfile::Inference);

    tuner.reset_stats();
    tuner.record_allocation(200 * 1024 * 1024);
    auto profile_large = tuner.detect_workload_profile();
    EXPECT_EQ(profile_large, WorkloadProfile::LargeBatch);
}

TEST_F(MemoryOptimizerEdgeCaseTest, WorkloadProfileSetting) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();

    tuner.set_workload_profile(WorkloadProfile::Training);
    auto config1 = tuner.get_config();
    EXPECT_EQ(config1.profile, WorkloadProfile::Training);

    tuner.set_workload_profile(WorkloadProfile::SmallBatch);
    auto config2 = tuner.get_config();
    EXPECT_EQ(config2.profile, WorkloadProfile::SmallBatch);

    tuner.set_workload_profile(WorkloadProfile::Inference);
    auto config3 = tuner.get_config();
    EXPECT_EQ(config3.profile, WorkloadProfile::Inference);

    tuner.set_workload_profile(WorkloadProfile::LargeBatch);
    auto config4 = tuner.get_config();
    EXPECT_EQ(config4.profile, WorkloadProfile::LargeBatch);
}

TEST_F(MemoryOptimizerEdgeCaseTest, AdaptiveTunerDisableEnable) {
    auto& tuner = AdaptiveMemoryPoolTuner::instance();

    tuner.enable_adaptive_tuning();
    EXPECT_TRUE(tuner.is_adaptive_enabled());

    tuner.disable_adaptive_tuning();
    EXPECT_FALSE(tuner.is_adaptive_enabled());

    tuner.enable_adaptive_tuning();
    EXPECT_TRUE(tuner.is_adaptive_enabled());
}

TEST_F(MemoryOptimizerEdgeCaseTest, GradientAccumulatorSingleStep) {
    GradientAccumulator accum(1);

    std::vector<float> grad(1024, 1.0f);
    accum.add_gradient(0, grad.data(), grad.size());

    EXPECT_TRUE(accum.is_ready_to_apply());
    EXPECT_EQ(accum.current_step(), 0);
    EXPECT_EQ(accum.max_steps(), 1);
}

TEST_F(MemoryOptimizerEdgeCaseTest, GradientAccumulatorReset) {
    GradientAccumulator accum(4);

    std::vector<float> grad(512, 0.5f);
    accum.add_gradient(0, grad.data(), grad.size());
    EXPECT_TRUE(accum.is_ready_to_apply());

    accum.reset();
    EXPECT_FALSE(accum.is_ready_to_apply());
    EXPECT_EQ(accum.current_step(), 0);
}

TEST_F(MemoryOptimizerEdgeCaseTest, GradientAccumulatorChangingSize) {
    GradientAccumulator accum(2);

    std::vector<float> grad1(256, 1.0f);
    std::vector<float> grad2(512, 1.0f);
    std::vector<float> output(512);

    accum.add_gradient(0, grad1.data(), grad1.size());
    accum.add_gradient(1, grad2.data(), grad2.size());

    EXPECT_TRUE(accum.is_ready_to_apply());
    accum.get_accumulated_gradient(output.data());
}

TEST_F(MemoryOptimizerEdgeCaseTest, MemoryOptimizationManagerMultipleRecords) {
    auto& manager = MemoryOptimizationManager::instance();

    for (int i = 0; i < 10; ++i) {
        manager.record_checkpoint_size(1024 * 1024, 512 * 1024);
    }

    auto stats = manager.get_stats();
    EXPECT_EQ(stats.original_bytes, 10 * 1024 * 1024);
    EXPECT_EQ(stats.compressed_bytes, 10 * 512 * 1024);
}

TEST_F(MemoryOptimizerEdgeCaseTest, MemoryOptimizationManagerReset) {
    auto& manager = MemoryOptimizationManager::instance();

    manager.record_checkpoint_size(1024 * 1024, 512 * 1024);
    manager.record_defragmentation();

    manager.reset_stats();

    auto stats = manager.get_stats();
    EXPECT_EQ(stats.original_bytes, 0);
    EXPECT_EQ(stats.compressed_bytes, 0);
    EXPECT_EQ(stats.num_defragmentations, 0);
}

}  // namespace cuda::memory_opt::test
