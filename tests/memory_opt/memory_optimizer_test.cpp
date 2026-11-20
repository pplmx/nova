#include <gtest/gtest.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/memory_opt/memory_optimizer.h>

namespace cuda::memory_opt::test {

    class MemoryOptimizerTest : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    TEST_F(MemoryOptimizerTest, CheckpointCompressorBasic) {
        auto& compressor = CheckpointCompressor::instance();

        CompressionConfig config;
        config.enable_compression = true;
        config.compression_level = 3;
        config.min_size_for_compression = 64;
        compressor.set_config(config);

        std::vector<float> input(1024, 1.0f);
        std::vector<float> compressed(1024);
        std::vector<float> decompressed(1024);

        size_t compressed_size = compressor.compress(input.data(), input.size() * sizeof(float), compressed.data(), compressed.size() * sizeof(float));

        EXPECT_GT(compressed_size, 0);

        size_t decompressed_size = compressor.decompress(compressed.data(), compressed_size, decompressed.data(), decompressed.size() * sizeof(float));

        EXPECT_EQ(decompressed_size, input.size() * sizeof(float));
    }

    TEST_F(MemoryOptimizerTest, CheckpointCompressorStats) {
        auto& compressor = CheckpointCompressor::instance();

        std::vector<float> input(1024, 0.5f);
        std::vector<float> buffer(1024);

        compressor.compress(input.data(), input.size() * sizeof(float), buffer.data(), buffer.size() * sizeof(float));

        EXPECT_GT(compressor.get_total_original_bytes(), 0);
    }

    TEST_F(MemoryOptimizerTest, AdaptiveMemoryPoolTunerBasic) {
        auto& tuner = AdaptiveMemoryPoolTuner::instance();
        tuner.reset_stats();

        tuner.record_allocation(1024 * 1024);
        tuner.record_allocation(2048 * 1024);
        tuner.record_deallocation(512 * 1024);

        EXPECT_TRUE(tuner.is_adaptive_enabled());

        tuner.disable_adaptive_tuning();
        EXPECT_FALSE(tuner.is_adaptive_enabled());

        tuner.enable_adaptive_tuning();
        EXPECT_TRUE(tuner.is_adaptive_enabled());
    }

    TEST_F(MemoryOptimizerTest, AdaptiveMemoryPoolTunerSuggestSize) {
        auto& tuner = AdaptiveMemoryPoolTuner::instance();
        tuner.reset_stats();

        PoolTuningConfig config;
        config.initial_pool_size = 256 * 1024 * 1024;
        config.max_pool_size = 2ULL * 1024 * 1024 * 1024;
        tuner.set_config(config);

        for (int i = 0; i < 10; ++i) {
            tuner.record_allocation((i + 1) * 1024 * 1024);
        }

        size_t suggested = tuner.suggest_pool_size();
        EXPECT_GE(suggested, config.initial_pool_size);
        EXPECT_LE(suggested, config.max_pool_size);
    }

    TEST_F(MemoryOptimizerTest, AdaptiveMemoryPoolTunerDefaultMaxPoolCapIsFinite) {
        auto& tuner = AdaptiveMemoryPoolTuner::instance();
        tuner.reset_stats();

        // Use the structurally default config. With a correct default max_pool_size
        // (2 GiB), the cap must remain a finite, sane bound that actually throttles
        // the suggested pool size regardless of sustained allocation pressure.
        PoolTuningConfig config;
        EXPECT_EQ(config.max_pool_size, 2ULL * 1024 * 1024 * 1024);
        tuner.set_config(config);

        // Simulate heavy sustained allocation pressure so the suggested size would
        // grow far past the cap; it must never exceed the configured maximum.
        for (int i = 0; i < 100; ++i) {
            tuner.record_allocation(256ULL * 1024 * 1024);
        }
        tuner.record_allocation_failure();  // force the grow path

        size_t suggested = tuner.suggest_pool_size();
        EXPECT_LE(suggested, config.max_pool_size);
        EXPECT_LE(suggested, 2ULL * 1024 * 1024 * 1024);
    }

    TEST_F(MemoryOptimizerTest, AdaptiveMemoryPoolTunerShouldGrow) {
        auto& tuner = AdaptiveMemoryPoolTuner::instance();
        tuner.reset_stats();

        EXPECT_FALSE(tuner.should_grow());

        tuner.record_allocation_failure();
        EXPECT_TRUE(tuner.should_grow());
    }

    TEST_F(MemoryOptimizerTest, AdaptiveMemoryPoolTunerProfileDetection) {
        auto& tuner = AdaptiveMemoryPoolTuner::instance();
        tuner.reset_stats();

        tuner.record_allocation(5 * 1024 * 1024);
        auto profile = tuner.detect_workload_profile();

        // 5 MiB sits below the 10 MiB threshold, so it must classify as Inference.
        EXPECT_EQ(profile, WorkloadProfile::Inference);

        tuner.set_workload_profile(WorkloadProfile::Training);
        auto config = tuner.get_config();
        EXPECT_EQ(config.profile, WorkloadProfile::Training);

        tuner.set_workload_profile(WorkloadProfile::Inference);
        config = tuner.get_config();
        EXPECT_EQ(config.profile, WorkloadProfile::Inference);
    }

    TEST_F(MemoryOptimizerTest, AdaptiveMemoryPoolTunerStats) {
        auto& tuner = AdaptiveMemoryPoolTuner::instance();
        tuner.reset_stats();

        // The tuner reports usage through get_stats(), so this case can check that the three
        // calls above are reflected. It previously only made the calls: the numbers could have
        // been ignored, doubled, or recorded under the wrong field and the test passed.
        const auto after_reset = tuner.get_stats();
        EXPECT_EQ(after_reset.total_allocations, 0u) << "reset_stats must zero the counters";
        EXPECT_EQ(after_reset.current_usage, 0u);

        tuner.record_allocation(1024);
        tuner.record_allocation(2048);
        tuner.record_deallocation(512);

        const auto stats = tuner.get_stats();
        EXPECT_EQ(stats.total_allocations, 2u) << "two allocations were recorded";
        EXPECT_EQ(stats.total_deallocations, 1u) << "one deallocation was recorded";
        EXPECT_GE(stats.peak_usage, 3072u) << "peak usage should have reached the sum of both allocations";
        // 1024 + 2048 - 512. The deallocation releases bytes from the running total.
        EXPECT_EQ(stats.current_usage, 2560u) << "current usage should be the allocations minus the release";

        // Toggling the tuner must not disturb what has been measured.
        tuner.enable_adaptive_tuning();
        const auto enabled = tuner.get_stats();
        EXPECT_EQ(enabled.current_usage, stats.current_usage) << "toggling the tuner must not change the recorded usage";
        tuner.disable_adaptive_tuning();
        tuner.enable_adaptive_tuning();
    }

    TEST_F(MemoryOptimizerTest, MemoryOptimizationManager) {
        auto& manager = MemoryOptimizationManager::instance();

        manager.enable_checkpoint_compression(true);
        manager.set_gradient_accumulation_steps(4);
        manager.enable_defragmentation(true);

        manager.record_checkpoint_size(1024 * 1024, 512 * 1024);
        manager.record_defragmentation();

        auto stats = manager.get_stats();
        EXPECT_EQ(stats.compressed_bytes, 512 * 1024);
        EXPECT_EQ(stats.original_bytes, 1024 * 1024);
        EXPECT_GT(stats.compression_ratio, 0);
    }

    TEST_F(MemoryOptimizerTest, WorkloadProfileEnum) {
        EXPECT_EQ(static_cast<int>(WorkloadProfile::SmallBatch), 0);
        EXPECT_EQ(static_cast<int>(WorkloadProfile::LargeBatch), 1);
        EXPECT_EQ(static_cast<int>(WorkloadProfile::Inference), 2);
        EXPECT_EQ(static_cast<int>(WorkloadProfile::Training), 3);
    }

    // Regression: get_stats() divided by compressed_bytes without checking it.
    //
    //     ratio = total_original_bytes_ > 0
    //                 ? float(total_original_bytes_) / total_compressed_bytes_
    //                 : 1.0f;
    //
    // The guard tests the NUMERATOR. Recording a checkpoint size where the compressed
    // side is zero -- nothing was written, the codec failed, or the caller only tracks
    // originals -- makes the ratio +inf, and a stats consumer that formats or compares it
    // gets a number that is not a ratio at all. The guard now tests the denominator.
    TEST_F(MemoryOptimizerTest, ZeroCompressedBytesDoesNotProduceAnInfiniteRatio) {
        auto& manager = MemoryOptimizationManager::instance();
        manager.reset_stats();
        manager.record_checkpoint_size(/*original=*/1000, /*compressed=*/0);

        const auto stats = manager.get_stats();
        EXPECT_EQ(stats.original_bytes, 1000u);
        EXPECT_EQ(stats.compressed_bytes, 0u);
        EXPECT_TRUE(std::isfinite(stats.compression_ratio)) << "ratio was " << stats.compression_ratio << " with zero compressed bytes";
        // Nothing was compressed, so the ratio has to say so rather than being +inf.
        EXPECT_FLOAT_EQ(stats.compression_ratio, 1.0f);
        manager.reset_stats();
    }

    // The two enable_* setters stored flags that nothing read. They are observable now, so
    // a caller can confirm the configuration took and can tell why a compression ratio is
    // zero.
    TEST_F(MemoryOptimizerTest, EnablingFlagsIsVisibleInTheStats) {
        auto& manager = MemoryOptimizationManager::instance();
        manager.reset_stats();

        manager.enable_checkpoint_compression(false);
        manager.enable_defragmentation(false);
        auto stats = manager.get_stats();
        EXPECT_FALSE(stats.compression_enabled);
        EXPECT_FALSE(stats.defragmentation_enabled);

        manager.enable_checkpoint_compression(true);
        manager.enable_defragmentation(true);
        stats = manager.get_stats();
        EXPECT_TRUE(stats.compression_enabled);
        EXPECT_TRUE(stats.defragmentation_enabled);

        manager.reset_stats();
    }

}  // namespace cuda::memory_opt::test
