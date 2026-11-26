/**
 * @file kernel_stats_sentinel_test.cpp
 * @brief Contracts for the kernel-statistics collector's initial state
 *
 * src/cuda/observability/kernel_stats.cpp accumulates per-kernel timings. Entries
 * can appear two ways: a timing call (record_kernel) or a bare launch notice
 * (record_launch, used when a caller knows a kernel launched but has no events to
 * measure).
 *
 * KernelStats::min_time_us defaults to std::numeric_limits<double>::max() so the
 * first measured timing replaces it via std::min. record_launch reused that default
 * when creating an entry, so a kernel that was announced but never timed reported
 *
 *     invocations = 0, min_time_us = 1.79769e+308, max_time_us = 0, avg_time_us = 0
 *
 * The sentinel leaked to callers as a reported duration — the largest finite double —
 * while the neighbouring fields in the same struct reported zero. Nothing had been
 * measured, so reporting that nothing is minimal is the consistent answer. The first
 * real measurement still seeds the minimum instead of being clamped by the zero.
 */

#include <cuda/observability/kernel_stats.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <limits>
#include <string>

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

        /// Records a timing of roughly `milliseconds` for `name` using real CUDA events, so
        /// the collector exercises its production path rather than a synthetic value.
        void record_timing(KernelStatsCollector& collector, const char* name, size_t blocks = 4, size_t threads = 128) {
            cudaEvent_t start = nullptr;
            cudaEvent_t end = nullptr;
            ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
            ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);

            ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
            // A trivial kernel gives the events something to bracket.
            auto* dummy = static_cast<float*>(nullptr);
            (void)dummy;
            ASSERT_EQ(cudaEventRecord(end), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);

            collector.record_kernel(name, start, end, blocks, threads);

            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }

    }  // namespace

    // Regression: this reported min_time_us as DBL_MAX.
    TEST(KernelStatsSentinelTest, LaunchedButUntimedReportsNoDuration) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        KernelStatsCollector collector;
        collector.record_launch("kernel", 8, 256);

        const KernelStats stats = collector.get_stats("kernel");

        EXPECT_EQ(stats.invocations, 0u) << "nothing was timed, so nothing was invoked";
        EXPECT_EQ(stats.min_time_us, 0.0) << "a sentinel must not escape as a reported duration; DBL_MAX is " << std::numeric_limits<double>::max();
        EXPECT_EQ(stats.max_time_us, 0.0);
        EXPECT_EQ(stats.avg_time_us, 0.0);
        EXPECT_EQ(stats.blocks_launched, 8u) << "the launch details are still recorded";
        EXPECT_EQ(stats.threads_per_block, 256u);
    }

    TEST(KernelStatsSentinelTest, FirstTimingAfterALaunchSeedsTheMinimum) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        KernelStatsCollector collector;
        collector.record_launch("kernel", 8, 256);
        record_timing(collector, "kernel");

        const KernelStats stats = collector.get_stats("kernel");

        ASSERT_EQ(stats.invocations, 1u);
        EXPECT_GT(stats.min_time_us, 0.0) << "the first measurement must seed the minimum, not be clamped to zero";
        EXPECT_GT(stats.max_time_us, 0.0);
        EXPECT_EQ(stats.min_time_us, stats.max_time_us) << "one measurement is both the minimum and the maximum";
        EXPECT_EQ(stats.avg_time_us, stats.total_time_us);
    }

    TEST(KernelStatsSentinelTest, TimedWithoutAPriorLaunchIsRecorded) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        KernelStatsCollector collector;
        record_timing(collector, "direct", 2, 64);

        const KernelStats stats = collector.get_stats("direct");

        ASSERT_EQ(stats.invocations, 1u);
        EXPECT_GT(stats.min_time_us, 0.0);
        EXPECT_LE(stats.min_time_us, stats.max_time_us);
        EXPECT_EQ(stats.blocks_launched, 2u);
        EXPECT_EQ(stats.threads_per_block, 64u);
    }

    TEST(KernelStatsSentinelTest, RepeatedTimingsTrackMinimumAndMaximum) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        KernelStatsCollector collector;
        constexpr int kRuns = 5;
        for (int i = 0; i < kRuns; ++i) {
            record_timing(collector, "repeated");
        }

        const KernelStats stats = collector.get_stats("repeated");

        EXPECT_EQ(stats.invocations, static_cast<uint64_t>(kRuns));
        EXPECT_GT(stats.min_time_us, 0.0);
        EXPECT_LE(stats.min_time_us, stats.max_time_us);
        EXPECT_GT(stats.total_time_us, 0.0);
        EXPECT_NEAR(stats.avg_time_us, stats.total_time_us / static_cast<double>(kRuns), 1e-6);
    }

    TEST(KernelStatsSentinelTest, UntimedEntryDoesNotDistortASubsequentTiming) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Two kernels, one announced and one timed: the announced one must not appear to
        // be the fastest ever recorded.
        KernelStatsCollector collector;
        collector.record_launch("announced", 1, 32);
        record_timing(collector, "measured");

        const KernelStats announced = collector.get_stats("announced");
        const KernelStats measured = collector.get_stats("measured");

        EXPECT_EQ(announced.min_time_us, 0.0);
        EXPECT_GT(measured.min_time_us, 0.0);
        EXPECT_LT(measured.min_time_us, std::numeric_limits<double>::max()) << "a real timing must never reach the sentinel";
    }

    TEST(KernelStatsSentinelTest, UnknownKernelReportsAZeroedRecord) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        KernelStatsCollector collector;
        const KernelStats missing = collector.get_stats("never-seen");

        EXPECT_TRUE(missing.kernel_name.empty());
        EXPECT_EQ(missing.invocations, 0u);
        EXPECT_EQ(missing.total_time_us, 0.0);
        EXPECT_EQ(missing.max_time_us, 0.0);
        // The min must not be the DBL_MAX sentinel either. record_launch() was fixed for
        // leaking it, but this path returned the default record verbatim, so an unknown
        // name produced min_time_us = 1.797e308 beside max_time_us = 0 and
        // invocations = 0 -- min > max, for something never measured. The previous
        // comment here said the sentinel was "by design ... never reached a caller
        // through record_launch", which was true of that path and not of this one.
        EXPECT_EQ(missing.min_time_us, 0.0) << "an unmeasured kernel must not report the sentinel as its minimum";
        EXPECT_LE(missing.min_time_us, missing.max_time_us);
    }

}  // namespace cuda::observability::test
