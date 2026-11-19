/**
 * @file autotuner_search_space_test.cpp
 * @brief Contracts for the autotuner's search-space handling
 *
 * include/cuda/performance/autotuner.h searches a block-size by grid-size grid and
 * keeps the fastest measurement. The best-time field starts at
 * std::numeric_limits<float>::max() so the first real measurement replaces it, but
 * the loop could complete without ever measuring anything:
 *
 *   block_sizes or grid_sizes empty      -> the nested loop never entered
 *   measure_iterations == 0              -> 0/0 = NaN, and NaN < FLT_MAX is false
 *
 * In both cases the sentinel was returned as the answer. Measured before the fix:
 *
 *   empty grid_sizes     -> best_time_ms = 3.40282e+38
 *   measure_iterations=0 -> best_time_ms = 3.40282e+38
 *
 * 3.4e38 milliseconds is roughly 10^28 years; no timing can produce it, so a caller
 * comparing configurations silently got a meaningless number instead of an error.
 * Both conditions now raise std::invalid_argument naming the offending setting.
 */

#include <cuda/performance/autotuner.h>

#include <gtest/gtest.h>

#include <unistd.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cuda::performance::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Runs `call` and returns the exception message, or an empty string.
        template <typename Callable>
        std::string message_from(Callable&& call) {
            try {
                call();
            } catch (const std::exception& error) {
                return error.what();
            }
            return "";
        }

        /// A search that measures quickly: one warm-up skipped, a couple of iterations.
        ///
        /// config_path points at a per-process temporary file. The default is the relative
        /// path "autotune_config.json", and Autotuner::tune() persists its cache there — so
        /// running these tests from the repository root rewrote a tracked file.
        AutotuneConfig small_config() {
            AutotuneConfig config;
            config.block_sizes = {256};
            config.grid_sizes = {1};
            config.warmup_iterations = 0;
            config.measure_iterations = 2;
            config.config_path = "/tmp/nova_autotune_test_" + std::to_string(::getpid()) + ".json";
            return config;
        }

    }  // namespace

    // Regression: this returned FLT_MAX as the best measured time.
    TEST(AutotunerSearchSpaceTest, EmptyBlockSizesIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();
        config.block_sizes.clear();

        Autotuner tuner(config);
        const std::string message = message_from([&] { (void)tuner.tune([](int, int) {}); });

        EXPECT_FALSE(message.empty()) << "an empty search space has no best configuration to report";
        EXPECT_NE(message.find("block size"), std::string::npos) << message;
    }

    // Regression: same sentinel leak through the other empty dimension.
    TEST(AutotunerSearchSpaceTest, EmptyGridSizesIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();
        config.grid_sizes.clear();

        Autotuner tuner(config);
        const std::string message = message_from([&] { (void)tuner.tune([](int, int) {}); });

        EXPECT_FALSE(message.empty());
        EXPECT_NE(message.find("grid size"), std::string::npos) << message;
    }

    // Regression: this produced NaN internally and also returned the sentinel.
    TEST(AutotunerSearchSpaceTest, ZeroMeasureIterationsIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();
        config.measure_iterations = 0;

        Autotuner tuner(config);
        const std::string message = message_from([&] { (void)tuner.tune([](int, int) {}); });

        EXPECT_FALSE(message.empty()) << "averaging over zero iterations divides by zero";
        EXPECT_NE(message.find("measure_iterations"), std::string::npos) << message;
    }

    TEST(AutotunerSearchSpaceTest, NegativeMeasureIterationsIsRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();
        config.measure_iterations = -5;

        Autotuner tuner(config);
        EXPECT_THROW(tuner.tune([](int, int) {}), std::invalid_argument);
    }

    TEST(AutotunerSearchSpaceTest, AValidSearchReportsARealMeasurement) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();

        Autotuner tuner(config);
        int calls = 0;
        const AutotuneResult result = tuner.tune([&calls](int, int) { ++calls; });

        EXPECT_EQ(calls, config.measure_iterations) << "the callable runs once per measurement iteration";
        EXPECT_TRUE(std::isfinite(result.best_time_ms)) << result.best_time_ms;
        EXPECT_GE(result.best_time_ms, 0.0f);
        EXPECT_LT(result.best_time_ms, std::numeric_limits<float>::max()) << "a real measurement must replace the sentinel";
        EXPECT_EQ(result.optimal_block_size, 256);
        EXPECT_EQ(result.optimal_grid_size, 1);
    }

    TEST(AutotunerSearchSpaceTest, TheFastestCandidateWins) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();
        config.block_sizes = {64, 128, 256};
        config.grid_sizes = {1};

        Autotuner tuner(config);
        const AutotuneResult result = tuner.tune([](int, int) {});

        // Every candidate measures the same no-op, so the winner is whichever recorded
        // the smallest time. What matters is that it is one of the offered candidates,
        // not a default left over from construction.
        EXPECT_NE(config.block_sizes.end(), std::find(config.block_sizes.begin(), config.block_sizes.end(), result.optimal_block_size))
            << "the reported block size was never measured";
        EXPECT_TRUE(std::isfinite(result.best_time_ms));
    }

    TEST(AutotunerSearchSpaceTest, SearchOverBothDimensionsCallsEveryCombination) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        AutotuneConfig config = small_config();
        config.block_sizes = {128, 256};
        config.grid_sizes = {1, 2, 4};
        config.measure_iterations = 1;

        Autotuner tuner(config);
        int calls = 0;
        (void)tuner.tune([&calls](int, int) { ++calls; });

        EXPECT_EQ(calls, 2 * 3 * 1) << "block sizes times grid sizes times iterations";
    }

}  // namespace cuda::performance::test
