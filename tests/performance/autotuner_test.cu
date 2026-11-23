#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/performance/autotuner.h>
#include <cuda/performance/device_info.h>

#include <algorithm>
#include <vector>

namespace cuda::performance::test {

    class AutotunerTest : public ::testing::Test {
    protected:
        void SetUp() override { cudaGetDevice(&device_); }

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

    // The four setters return void and Autotuner has no getters, so calling them and stopping
    // verified nothing: a no-op setter would have passed this case just as easily. They can only
    // be checked through the search they drive -- tune() walks every (block, grid) pair
    // warmup+measure times and reports one of the configured block sizes as the optimum -- so
    // this case pins all three of those facts.
    TEST_F(AutotunerTest, ConfigSetters) {
        AutotuneConfig config;
        config.config_path = "/tmp/autotuner_configsetters.json";
        Autotuner tuner(config);

        const std::vector<int> blocks {128, 256, 512};
        const std::vector<int> grids {8, 16};
        constexpr int kWarmup = 2;
        constexpr int kMeasure = 3;

        tuner.set_block_sizes(blocks);
        tuner.set_grid_sizes(grids);
        tuner.set_warmup_iterations(kWarmup);
        tuner.set_measure_iterations(kMeasure);

        int calls = 0;
        std::vector<int> visited_blocks;
        const auto result = tuner.tune([&](int block_size, int grid_size) {
            ++calls;
            (void)grid_size;
            if (visited_blocks.empty() || visited_blocks.back() != block_size) {
                visited_blocks.push_back(block_size);
            }
        });

        EXPECT_EQ(calls, static_cast<int>(blocks.size() * grids.size() * (kWarmup + kMeasure)))
            << "the search must run every pair exactly warmup+measure times; a no-op setter changes this product";

        // Every configured block size must have been visited, in the configured set only.
        std::vector<int> sorted_visited = visited_blocks;
        std::sort(sorted_visited.begin(), sorted_visited.end());
        EXPECT_EQ(sorted_visited, blocks) << "the block-size setter must drive the search";

        EXPECT_NE(std::find(blocks.begin(), blocks.end(), result.optimal_block_size), blocks.end())
            << "optimal_block_size " << result.optimal_block_size << " is outside the configured set";
        EXPECT_NE(std::find(grids.begin(), grids.end(), result.optimal_grid_size), grids.end())
            << "optimal_grid_size " << result.optimal_grid_size << " is outside the configured set";
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
