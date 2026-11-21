#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <cuda/performance/autotuner.h>
#include <cuda/performance/device_info.h>

namespace cuda::performance::test {

    class AutotunerEdgeCaseTest : public ::testing::Test {
    protected:
        void SetUp() override { cudaGetDevice(&device_); }

        int device_ = 0;
    };

    TEST_F(AutotunerEdgeCaseTest, EmptyBlockSizes) {
        AutotuneConfig config;
        config.block_sizes = {};
        config.grid_sizes = {1, 2, 4};
        config.config_path = "/tmp/autotuner_edge.json";

        Autotuner tuner(config);
        tuner.set_block_sizes({});

        auto result = tuner.load_cached_result("empty_test");
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(AutotunerEdgeCaseTest, EmptyGridSizes) {
        AutotuneConfig config;
        config.block_sizes = {64, 128, 256};
        config.grid_sizes = {};
        config.config_path = "/tmp/autotuner_edge.json";

        Autotuner tuner(config);
        tuner.set_grid_sizes({});

        auto result = tuner.load_cached_result("empty_grid_test");
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(AutotunerEdgeCaseTest, ZeroWarmupIterations) {
        AutotuneConfig config;
        config.warmup_iterations = 0;
        config.measure_iterations = 1;
        config.config_path = "/tmp/autotuner_edge.json";

        Autotuner tuner(config);
        tuner.set_warmup_iterations(0);

        auto result = tuner.load_cached_result("zero_warmup_test");
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(AutotunerEdgeCaseTest, ZeroMeasureIterations) {
        AutotuneConfig config;
        config.warmup_iterations = 1;
        config.measure_iterations = 0;
        config.config_path = "/tmp/autotuner_edge.json";

        Autotuner tuner(config);
        tuner.set_measure_iterations(0);

        auto result = tuner.load_cached_result("zero_measure_test");
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(AutotunerEdgeCaseTest, CustomConfigPath) {
        AutotuneConfig config;
        config.config_path = "/tmp/custom_autotune.json";

        Autotuner tuner(config);
        tuner.save_result("custom_path_test", AutotuneResult {});

        auto result = tuner.load_cached_result("custom_path_test");
        EXPECT_TRUE(result.has_value());
    }

    TEST_F(AutotunerEdgeCaseTest, MultipleDeviceIDs) {
        AutotuneConfig config0;
        config0.device_id = 0;
        config0.kernel_name = "test_kernel";
        // Use a temp path so the destructor (which persists the cache) never
        // truncates/overwrites the tracked autotune_config.json in the repo.
        config0.config_path = "/tmp/autotuner_multidevice.json";

        AutotuneConfig config1;
        config1.device_id = 1;
        config1.kernel_name = "test_kernel";
        config1.config_path = "/tmp/autotuner_multidevice.json";

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

    // Regression: the cache readers called std::stoi/std::stof directly, so one damaged field
    // threw out of the Autotuner CONSTRUCTOR (which is what reads the cache) and made the
    // tuner unusable. Measured on the field-by-field shapes below:
    //
    //     block_size=abc                    -> std::invalid_argument: stoi
    //     block_size=99999999999999999999   -> std::out_of_range: stoi
    //     block_size=                       -> std::invalid_argument: stoi
    //     time_ms=nan                        -> accepted, timing = nan
    //
    // A cache file is written by a previous run and may be truncated by a crash or edited by
    // hand, so parsing it must be total. A field that cannot be read keeps its default and the
    // rest of the line still loads; a non-finite timing is not a measurement.
    TEST_F(AutotunerEdgeCaseTest, DamagedCacheFileIsToleratedNotFatal) {
        struct Case {
            const char* label;
            const char* line;
            int expected_block_size;
            bool expect_finite_time;  // false: the timing field must be rejected
        };
        const Case cases[] = {
            {"non-numeric block size", "k_d0:block_size=abc,grid_size=1,time_ms=1.0,speedup=2.0", 256, true},
            {"block size past an int", "k_d0:block_size=99999999999999999999,grid_size=1,time_ms=1.0,speedup=2.0", 256, true},
            {"empty block size", "k_d0:block_size=,grid_size=1,time_ms=1.0,speedup=2.0", 256, true},
            // A NaN timing is rejected, so the field keeps its default of 0.0 -- which IS
            // finite. The assertion below checks the value is the default, not that it is
            // non-finite; conflating the two is how the original defect went unnoticed.
            {"nan timing", "k_d0:block_size=128,grid_size=1,time_ms=nan,speedup=2.0", 128, true},
            {"trailing junk", "k_d0:block_size=128x,grid_size=1,time_ms=1.0,speedup=2.0", 256, true},
        };

        int index = 0;
        for (const auto& c : cases) {
            const std::string path = std::string("/tmp/nova_autotune_damaged_") + std::to_string(index++) + ".json";
            {
                std::ofstream out(path);
                ASSERT_TRUE(out.is_open());
                out << "# written by a previous run\n" << c.line << "\n";
            }

            AutotuneConfig config;
            config.config_path = path;
            try {
                // Constructing the tuner is what reads the cache; it must not throw.
                Autotuner tuner(config);
                const auto loaded = tuner.load_cached_result("k");
                ASSERT_TRUE(loaded.has_value()) << c.label << ": the whole line was dropped instead of the bad field";
                EXPECT_EQ(loaded->optimal_block_size, c.expected_block_size) << c.label;
                // A rejected timing keeps its default of 0.0 and never stores a NaN: a NaN is
                // not a measurement, and carrying it into AutotuneResult would put it into
                // speedup comparisons where every comparison against NaN is false.
                EXPECT_FALSE(std::isnan(loaded->best_time_ms)) << c.label << ": the result carries a NaN timing";
                if (!c.expect_finite_time) {
                    EXPECT_FLOAT_EQ(loaded->best_time_ms, 0.0f) << c.label << ": a rejected timing must keep its default";
                }
            } catch (const std::exception& error) {
                ADD_FAILURE() << c.label << ": constructing the tuner threw " << error.what();
            }
            std::filesystem::remove(path);
        }
    }

    // The registry had a byte-for-byte copy of the same loop, so it shared the defect. Measured
    // the same way: load_from_file() threw on a damaged field.
    TEST_F(AutotunerEdgeCaseTest, DamagedRegistryFileIsToleratedNotFatal) {
        const std::string path = "/tmp/nova_autotune_registry_damaged.json";
        {
            std::ofstream out(path);
            ASSERT_TRUE(out.is_open());
            out << "good_d0:block_size=512,grid_size=32,time_ms=0.5,speedup=1.5\n";
            out << "bad_d0:block_size=not-a-number,grid_size=32,time_ms=0.5,speedup=1.5\n";
        }

        auto& registry = AutotuneRegistry::instance();
        registry.clear();
        EXPECT_NO_THROW(registry.load_from_file(path)) << "one damaged line must not abort the load";

        // The damaged line still yields an entry, with the unreadable field at its default.
        const auto bad = registry.get_result("bad", 0);
        ASSERT_TRUE(bad.has_value()) << "the line was dropped entirely rather than the field";
        EXPECT_EQ(bad->optimal_block_size, 256) << "the unreadable field keeps its default";

        // And the good line is unaffected.
        const auto good = registry.get_result("good", 0);
        ASSERT_TRUE(good.has_value());
        EXPECT_EQ(good->optimal_block_size, 512);

        registry.clear();
        std::filesystem::remove(path);
    }

}  // namespace cuda::performance::test
