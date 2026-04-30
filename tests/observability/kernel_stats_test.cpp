#include <gtest/gtest.h>

#include "cuda/observability/kernel_stats.h"

namespace cuda::observability::test {

class KernelStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        collector_.reset();
    }

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
        cudaSleep(0, t);
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

TEST_F(KernelStatsTest, GetStatsReturnsEmptyForUnknown) {
    auto stats = collector_.get_stats("nonexistent");
    EXPECT_EQ(stats.invocations, 0);
}

TEST_F(KernelStatsTest, OccupancyMetricsMeasure) {
    auto metrics = measure_occupancy(nullptr, 0, 0);

    EXPECT_GE(metrics.theoretical_occupancy, 0.0);
    EXPECT_GE(metrics.achieved_occupancy, 0.0);
}

}  // namespace cuda::observability::test
