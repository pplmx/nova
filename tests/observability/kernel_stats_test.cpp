#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

#include "cuda_alloc_injection.h"
#include "cuda/observability/kernel_stats.h"

namespace cuda::observability::test {

    class KernelStatsTest : public ::testing::Test {
    protected:
        void SetUp() override { collector_.reset(); }

        KernelStatsCollector collector_;
    };

    TEST_F(KernelStatsTest, RecordSingleKernel) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        cudaEventSynchronize(start);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        collector_.record_kernel("test_kernel", start, end, 10, 256);

        auto stats = collector_.get_stats("test_kernel");
        EXPECT_EQ(stats.invocations, 1);
        EXPECT_GT(stats.total_time_us, 0.0);
        EXPECT_EQ(stats.blocks_launched, 10);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    // ScopedKernelTiming owns two cudaEvent_t and destroys both in its destructor, and declared no
    // copy or move operations. The compiler-generated copy therefore made two objects own the same
    // two events, and both destructors destroyed them -- the second on a handle that no longer
    // exists. Its scope semantics also make a copy meaningless: the timing is recorded when the
    // object dies, and two copies cannot both record it.
    TEST_F(KernelStatsTest, ScopedTimingIsExclusiveToItsScope) {
        static_assert(!std::is_copy_constructible_v<ScopedKernelTiming>, "a copy would destroy the same events twice");
        static_assert(!std::is_copy_assignable_v<ScopedKernelTiming>, "a copy would destroy the same events twice");
        static_assert(!std::is_move_constructible_v<ScopedKernelTiming>, "the timing records on destruction, so it must not be relocatable");
        static_assert(!std::is_move_assignable_v<ScopedKernelTiming>, "the timing records on destruction, so it must not be relocatable");

        // A scope still works normally.
        cuda_test::reset_event_count();
        { ScopedKernelTiming timing("exclusive-probe", collector_); }
        EXPECT_EQ(cuda_test::live_event_count(), 0) << "both events must be released when the scope ends";
    }

    TEST_F(KernelStatsTest, MultipleInvocationsAccumulate) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        for (int i = 0; i < 5; ++i) {
            cudaEventRecord(start);
            cudaEventSynchronize(start);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            collector_.record_kernel("test_kernel", start, end, 10, 256);
        }

        auto stats = collector_.get_stats("test_kernel");
        EXPECT_EQ(stats.invocations, 5);
        EXPECT_EQ(stats.blocks_launched, 50);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    TEST_F(KernelStatsTest, DifferentKernelsTrackedSeparately) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        cudaEventSynchronize(start);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        collector_.record_kernel("kernel_a", start, end, 10, 256);
        collector_.record_kernel("kernel_b", start, end, 20, 128);

        auto stats_a = collector_.get_stats("kernel_a");
        auto stats_b = collector_.get_stats("kernel_b");

        EXPECT_EQ(stats_a.invocations, 1);
        EXPECT_EQ(stats_b.invocations, 1);
        EXPECT_EQ(collector_.stats().size(), 2);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    TEST_F(KernelStatsTest, MinMaxAvgCalculatedCorrectly) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        std::vector<int> times_us = {100, 200, 150};

        for (int t : times_us) {
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            cudaEventSynchronize(start);
            cudaEvent_t dummy_end;
            cudaEventCreate(&dummy_end);
            std::this_thread::sleep_for(std::chrono::microseconds(t));
            cudaEventRecord(dummy_end);
            cudaEventSynchronize(dummy_end);
            collector_.record_kernel("test_kernel", start, dummy_end, 1, 256);
            cudaEventDestroy(dummy_end);
        }

        auto stats = collector_.get_stats("test_kernel");
        EXPECT_EQ(stats.invocations, 3);
        EXPECT_GE(stats.min_time_us, 0.0);
        EXPECT_GE(stats.max_time_us, stats.min_time_us);
        EXPECT_GE(stats.avg_time_us, stats.min_time_us);
        EXPECT_LE(stats.avg_time_us, stats.max_time_us);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    TEST_F(KernelStatsTest, ResetClearsAllStats) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        cudaEventSynchronize(start);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        collector_.record_kernel("test_kernel", start, end, 10, 256);
        EXPECT_EQ(collector_.stats().size(), 1);

        collector_.reset();
        EXPECT_EQ(collector_.stats().size(), 0);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    // ScopedKernelTiming's constructor creates two events and then records the first. A throwing
    // constructor does not run its destructor, so a failure on the second create leaves the first
    // event with no owner -- and end_ is not initialised in the member list, so there is no safe
    // value to destroy even if something tried.
    //
    // Reachable only through fail_event_after(): the event create count is what has to fail, and the
    // wrapper ignored the allocation arm before that was added.
    TEST_F(KernelStatsTest, AFailedSecondEventCreateDoesNotLeakTheFirst) {
        cuda_test::reset_event_count();

        // Let the first event through, fail the second.
        cuda_test::fail_event_after(1);
        EXPECT_THROW((void)ScopedKernelTiming("k", collector_, 1, 1), std::exception);
        cuda_test::stop_failing_events();

        EXPECT_EQ(cuda_test::live_event_count(), 0) << "the event created before the failure must be destroyed";
    }

    TEST_F(KernelStatsTest, AFailedFirstEventCreateLeavesNothingBehind) {
        cuda_test::reset_event_count();

        cuda_test::fail_event_after(0);
        EXPECT_THROW((void)ScopedKernelTiming("k", collector_, 1, 1), std::exception);
        cuda_test::stop_failing_events();

        EXPECT_EQ(cuda_test::live_event_count(), 0);
    }

    // The default-constructed struct carries min_time_us = DBL_MAX so that the first real timing
    // replaces it. get_stats() returned that default verbatim for a name it has never seen, so the
    // caller received a record claiming a minimum time of 1.797e308 next to max_time_us == 0 and
    // invocations == 0 -- min > max, for a kernel nobody measured.
    //
    // record_launch() eighteen lines up was already fixed for exactly this ("reporting DBL_MAX
    // leaked that sentinel to callers ... Nothing has been measured, so nothing is minimal"); the
    // unknown-name path kept it.
    TEST_F(KernelStatsTest, GetStatsReportsNoTimingForUnknown) {
        auto stats = collector_.get_stats("nonexistent");
        EXPECT_EQ(stats.invocations, 0);
        EXPECT_EQ(stats.min_time_us, 0.0) << "an unmeasured kernel must not report the DBL_MAX sentinel as its minimum";
        EXPECT_EQ(stats.max_time_us, 0.0);
        EXPECT_EQ(stats.avg_time_us, 0.0);
        EXPECT_LE(stats.min_time_us, stats.max_time_us) << "min > max is not a record any measurement could produce";
    }

    TEST_F(KernelStatsTest, OccupancyMetricsMeasure) {
        auto metrics = measure_occupancy(nullptr, 0, 0);

        EXPECT_GE(metrics.theoretical_occupancy, 0.0);
        EXPECT_GE(metrics.achieved_occupancy, 0.0);
    }

}  // namespace cuda::observability::test
