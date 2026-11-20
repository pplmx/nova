/**
 * @file occupancy_analyzer_test.cpp
 * @brief Contracts for the occupancy and block-size analyser
 *
 * src/observability/occupancy_analyzer.cpp queries CUDA occupancy limits. Two of
 * its functions had any test reference, and one defect was found in three places:
 *
 *   int old_device;
 *   cudaGetDevice(&old_device);
 *   ...
 *   cudaSetDevice(old_device);
 *
 * Nobody checked cudaGetDevice's result and old_device was uninitialised, so on
 * failure the garbage value went straight to cudaSetDevice and switched the
 * calling thread to an arbitrary device — corrupting the caller's device context
 * for every subsequent CUDA call in that thread. The compiler stays silent here
 * because the callee writes through a pointer. old_device is now initialised and
 * the restore is skipped when the read failed.
 *
 * The analyser's arithmetic and its degradation on a bad device were checked and
 * are correct: the occupancy ratio, the ceil-division for a full-occupancy block
 * count, an empty range, and a nonexistent device reporting zero rather than
 * throwing.
 */

#include <cuda/observability/occupancy_analyzer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <vector>

namespace cuda::observability::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// A plain host function used as an opaque kernel handle.
        ///
        /// This file is a .cpp, so it cannot define a __global__ function. It used to
        /// pass this to the occupancy API on the reasoning that "the API returns an error
        /// for a non-kernel pointer" and that error was harmless. Measured: the error is
        /// cudaErrorInvalidResourceHandle and it is left in the thread's error state, so
        /// the next unrelated CUDA call reads it -- in the full single-process run that was
        /// HistogramBinsTest.Explicit256IsAccepted two suites later. The analyser now
        /// clears the error it provokes (see OccupancyAnalyzer::analyze), which is what
        /// makes this handle safe to use.
        void occupancy_probe_target() {}

    }  // namespace

    TEST(OccupancyQueryTest, ReadsDeviceLimits) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        OccupancyAnalyzer analyzer(0);

        EXPECT_GT(analyzer.sm_count(), 0) << "a real device has SMs";
    }

    // Regression: an uninitialised old_device could be passed to cudaSetDevice.
    //
    // LIMITATION, stated rather than hidden: cudaGetDevice does not fail on a healthy
    // driver, so this test cannot exercise the failing branch and therefore does not
    // distinguish the fixed code from the broken code — it passes either way. It is
    // kept because it pins the property that must hold on the normal path (the
    // caller's device is restored) and would catch a regression that dropped the
    // restore altogether. The uninitialised read itself was verified by reproducing
    // the pattern outside CUDA: with the getter failing, the pre-fix code calls the
    // setter with whatever was on the stack, the fixed code skips the call.
    TEST(OccupancyQueryTest, LeavesTheCallingDeviceUnchanged) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        int before = -1;
        ASSERT_EQ(cudaGetDevice(&before), cudaSuccess);

        OccupancyAnalyzer analyzer(0);
        (void)analyzer.analyze(reinterpret_cast<const void*>(&occupancy_probe_target), 128, 0);
        (void)analyzer.recommend(reinterpret_cast<const void*>(&occupancy_probe_target), 0, 256);

        int after = -1;
        ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
        EXPECT_EQ(after, before) << "the analyser must restore the caller's device, got " << after << " instead of " << before;
    }

    TEST(OccupancyQueryTest, AnalysisOfANullKernelIsZeroed) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        OccupancyAnalyzer analyzer(0);
        const auto analysis = analyzer.analyze(nullptr, 128, 0);

        // A null kernel cannot be measured, so every field must stay at its default
        // rather than carry a value computed from a failed query.
        EXPECT_DOUBLE_EQ(analysis.theoretical_occupancy, 0.0);
        EXPECT_DOUBLE_EQ(analysis.achieved_occupancy, 0.0);
        EXPECT_EQ(analysis.active_blocks_per_sm, 0);
        EXPECT_EQ(analysis.max_blocks_per_sm, 0);
        EXPECT_EQ(analysis.max_threads_per_block, 128) << "the request is still echoed";
    }

    TEST(OccupancyQueryTest, RecommendationStaysWithinTheHardwareLimits) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        OccupancyAnalyzer analyzer(0);
        const auto recommendation = analyzer.recommend(reinterpret_cast<const void*>(&occupancy_probe_target), 0, 0);

        EXPECT_GT(recommendation.recommended_block_size, 0);
        EXPECT_LE(recommendation.recommended_block_size, 1024) << "no kernel may launch more than 1024 threads per block";
        EXPECT_GE(recommendation.recommended_grid_size, 0);
        EXPECT_GE(recommendation.expected_occupancy, 0.0);
        EXPECT_LE(recommendation.expected_occupancy, 1.0);
    }

    TEST(OccupancyQueryTest, AnalyzeRangeStepsByWarpSize) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        OccupancyAnalyzer analyzer(0);
        const auto results = analyzer.analyze_range(reinterpret_cast<const void*>(&occupancy_probe_target), 32, 128, 0);

        // 32, 64, 96, 128 on a warp size of 32.
        ASSERT_EQ(results.size(), 4u);
        EXPECT_EQ(results.front().max_threads_per_block, 32);
        EXPECT_EQ(results.back().max_threads_per_block, 128);
    }

    TEST(OccupancyQueryTest, ReversedRangeIsEmpty) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        OccupancyAnalyzer analyzer(0);
        const auto results = analyzer.analyze_range(reinterpret_cast<const void*>(&occupancy_probe_target), 256, 32, 0);

        EXPECT_TRUE(results.empty()) << "a range whose minimum exceeds its maximum contains nothing";
    }

    TEST(OccupancyQueryTest, NonExistentDeviceDegradesInsteadOfThrowing) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A device index that does not exist must not crash the analysis; the
        // queries fail and the analyser reports what it knows (nothing).
        OccupancyAnalyzer analyzer(9999);
        EXPECT_EQ(analyzer.sm_count(), 0) << "no limits could be read";

        const auto analysis = analyzer.analyze(nullptr, 128, 0);
        EXPECT_DOUBLE_EQ(analysis.theoretical_occupancy, 0.0);
        (void)cudaGetLastError();  // clear the sticky error the failed queries set
    }

    TEST(OccupancyQueryTest, BlockSizeFeedbackFormatsItself) {
        BlockSizeFeedback feedback;
        feedback.block_size = 256;
        feedback.occupancy = 0.75;
        feedback.is_optimal = false;

        const std::string text = feedback.to_string();
        EXPECT_NE(text.find("256"), std::string::npos) << text;
        EXPECT_NE(text.find("75"), std::string::npos) << text;
        EXPECT_EQ(text.find("[OPTIMAL]"), std::string::npos) << "a non-optimal entry must not be labelled optimal: " << text;

        feedback.is_optimal = true;
        EXPECT_NE(feedback.to_string().find("[OPTIMAL]"), std::string::npos);
    }

    TEST(OccupancyQueryTest, AnalyzeBlockSizesMarksTheBestCandidate) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const auto feedback = analyze_block_sizes(reinterpret_cast<const void*>(&occupancy_probe_target), 0, 0);

        ASSERT_FALSE(feedback.empty()) << "the sweep 32..1024 should produce entries";

        int optimal_count = 0;
        for (const auto& entry : feedback) {
            EXPECT_GE(entry.block_size, 32);
            EXPECT_LE(entry.block_size, 1024);
            EXPECT_GE(entry.occupancy, 0.0);
            EXPECT_LE(entry.occupancy, 1.0);
            if (entry.is_optimal) {
                ++optimal_count;
            }
        }
        EXPECT_LE(optimal_count, 1) << "at most one entry can be marked optimal";
    }

}  // namespace cuda::observability::test
