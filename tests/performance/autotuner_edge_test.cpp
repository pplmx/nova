#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/performance/autotuner.h>
#include <cuda/performance/device_info.h>

namespace cuda::performance::test {

class AutotunerEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
    }

    int device_ = 0;
};

TEST_F(AutotunerEdgeCaseTest, EmptyBlockSizes) {
    AutotuneConfig config;
    config.block_sizes = {};
    config.grid_sizes = {1, 2, 4};

    Autotuner tuner(config);
    tuner.set_block_sizes({});

    auto result = tuner.load_cached_result("empty_test");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AutotunerEdgeCaseTest, EmptyGridSizes) {
    AutotuneConfig config;
    config.block_sizes = {64, 128, 256};
    config.grid_sizes = {};

    Autotuner tuner(config);
    tuner.set_grid_sizes({});

    auto result = tuner.load_cached_result("empty_grid_test");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AutotunerEdgeCaseTest, ZeroWarmupIterations) {
    AutotuneConfig config;
    config.warmup_iterations = 0;
    config.measure_iterations = 1;

    Autotuner tuner(config);
    tuner.set_warmup_iterations(0);

    auto result = tuner.load_cached_result("zero_warmup_test");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AutotunerEdgeCaseTest, ZeroMeasureIterations) {
    AutotuneConfig config;
    config.warmup_iterations = 1;
    config.measure_iterations = 0;

    Autotuner tuner(config);
    tuner.set_measure_iterations(0);

    auto result = tuner.load_cached_result("zero_measure_test");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AutotunerEdgeCaseTest, CustomConfigPath) {
    AutotuneConfig config;
    config.config_path = "/tmp/custom_autotune.json";

    Autotuner tuner(config);
    tuner.save_result("custom_path_test", AutotuneResult{});

    auto result = tuner.load_cached_result("custom_path_test");
    EXPECT_TRUE(result.has_value());
}

TEST_F(AutotunerEdgeCaseTest, MultipleDeviceIDs) {
    AutotuneConfig config0;
    config0.device_id = 0;
    config0.kernel_name = "test_kernel";

    AutotuneConfig config1;
    config1.device_id = 1;
    config1.kernel_name = "test_kernel";

    Autotuner tuner0(config0);
    Autotuner tuner1(config1);

    AutotuneResult result0;
    result0.optimal_block_size = 128;
    tuner0.save_result("same_kernel", result0);

    auto loaded0 = tuner0.load_cached_result("same_kernel");
    auto loaded1 = tuner1.load_cached_result("same_kernel");

    EXPECT_TRUE(loaded0.has_value());
    EXPECT_FALSE(loaded1.has_value());
}

TEST_F(AutotunerEdgeCaseTest, RegistryAcrossDevices) {
    auto& registry = AutotuneRegistry::instance();
    registry.clear();

    for (int device = 0; device < 4; ++device) {
        AutotuneResult result;
        result.optimal_block_size = 128 * (device + 1);
        result.optimal_grid_size = 32 * (device + 1);
        registry.register_result("kernel_d" + std::to_string(device), device, result);
    }

    for (int device = 0; device < 4; ++device) {
        auto result = registry.get_result("kernel_d" + std::to_string(device), device);
        EXPECT_TRUE(result.has_value());
        EXPECT_EQ(result->optimal_block_size, 128 * (device + 1));
    }

    for (int device = 0; device < 4; ++device) {
        auto wrong_device = registry.get_result("kernel_d0", device);
        if (device != 0) {
            EXPECT_FALSE(wrong_device.has_value());
        }
    }
}

TEST_F(AutotunerEdgeCaseTest, DefaultConfigPathConsistency) {
    std::string path1 = Autotuner::get_default_config_path();
    std::string path2 = Autotuner::get_default_config_path();
    EXPECT_EQ(path1, path2);
    EXPECT_EQ(path1, "autotune_config.json");
}

TEST_F(AutotunerEdgeCaseTest, ConfigPersistence) {
    AutotuneRegistry registry;
    registry.clear();

    AutotuneResult result;
    result.optimal_block_size = 512;
    result.optimal_grid_size = 64;
    result.best_time_ms = 0.123f;
    result.speedup_vs_default = 1.5f;
    registry.register_result("persist_test", 0, result);

    auto loaded = registry.get_result("persist_test", 0);
    EXPECT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->optimal_block_size, 512);
    EXPECT_EQ(loaded->optimal_grid_size, 64);
    EXPECT_FLOAT_EQ(loaded->best_time_ms, 0.123f);
    EXPECT_FLOAT_EQ(loaded->speedup_vs_default, 1.5f);
}

}  // namespace cuda::performance::test
