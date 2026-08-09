#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/performance/autotuner.h>
#include <cuda/performance/device_info.h>

namespace cuda::performance::test {

class AutotunerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
    }

    int device_ = 0;
};

__global__ void dummy_kernel(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value * 2.0f;
    }
}

TEST_F(AutotunerTest, BasicAutotuning) {
    AutotuneConfig config;
    config.device_id = device_;
    config.kernel_name = "dummy_kernel";
    config.block_sizes = {64, 128, 256};
    config.grid_sizes = {1, 2, 4};
    config.warmup_iterations = 2;
    config.measure_iterations = 5;
    config.config_path = "/tmp/autotuner_basic.json";

    Autotuner tuner(config);

    std::vector<float> h_data(1024, 1.0f);
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));

    auto kernel_func = [&](int block_size, int grid_size) {
        cudaMemcpy(d_data, h_data.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice);
        dummy_kernel<<<grid_size, block_size>>>(d_data, 1024, 1.0f);
        cudaStreamSynchronize(0);
    };

    auto result = tuner.tune(kernel_func);

    EXPECT_GE(result.optimal_block_size, 64);
    EXPECT_LE(result.optimal_block_size, 256);
    EXPECT_GE(result.optimal_grid_size, 1);
    EXPECT_LE(result.optimal_grid_size, 4);
    EXPECT_GT(result.best_time_ms, 0.0f);

    cudaFree(d_data);
}

TEST_F(AutotunerTest, ConfigSetters) {
    AutotuneConfig config;
    config.config_path = "/tmp/autotuner_configsetters.json";
    Autotuner tuner(config);

    tuner.set_block_sizes({128, 256, 512});
    tuner.set_grid_sizes({8, 16, 32});
    tuner.set_warmup_iterations(5);
    tuner.set_measure_iterations(20);
}

TEST_F(AutotunerTest, CacheOperations) {
    AutotuneConfig config;
    config.config_path = "/tmp/autotuner_cache.json";
    Autotuner tuner(config);

    AutotuneResult expected;
    expected.optimal_block_size = 256;
    expected.optimal_grid_size = 32;
    expected.best_time_ms = 0.5f;
    expected.speedup_vs_default = 1.5f;

    tuner.save_result("test_kernel", expected);
    tuner.save_all_results();

    auto loaded = tuner.load_cached_result("test_kernel");
    EXPECT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->optimal_block_size, 256);
    EXPECT_EQ(loaded->optimal_grid_size, 32);
}

TEST_F(AutotunerTest, AutotuneRegistry) {
    auto& registry = AutotuneRegistry::instance();

    AutotuneResult result1;
    result1.optimal_block_size = 128;
    result1.optimal_grid_size = 16;
    result1.best_time_ms = 1.0f;
    result1.speedup_vs_default = 1.2f;

    registry.register_result("kernel1", 0, result1);

    auto retrieved = registry.get_result("kernel1", 0);
    EXPECT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->optimal_block_size, 128);
    EXPECT_EQ(retrieved->optimal_grid_size, 16);

    registry.clear();
    auto cleared = registry.get_result("kernel1", 0);
    EXPECT_FALSE(cleared.has_value());
}

TEST_F(AutotunerTest, DefaultConfigPath) {
    std::string path = Autotuner::get_default_config_path();
    EXPECT_EQ(path, "autotune_config.json");
}

}  // namespace cuda::performance::test
