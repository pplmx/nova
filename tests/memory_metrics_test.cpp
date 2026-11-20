#include <gtest/gtest.h>
#include "cuda/performance/memory_metrics.h"

class MemoryMetricsTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

// Compares two separate device-wide snapshots, so it lives in a ...SerialTest suite
// and CMake registers it with RUN_SERIAL. get_metrics() and used() each call
// cudaMemGetInfo, and another process allocating between the two calls is
// indistinguishable from the function disagreeing with itself: the case failed with
// "metrics.used_bytes: 3570991104 vs used(): 3573088256" under -j4 and passed alone.
class MemoryMetricsSerialTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(MemoryMetricsTest, UsedReturnsNonNegativeValue) {
    size_t used = cuda::memory::used();
    EXPECT_GE(used, 0);
}

TEST_F(MemoryMetricsTest, AvailableReturnsPositiveValue) {
    size_t available = cuda::memory::available();
    EXPECT_GT(available, 0);
}

TEST_F(MemoryMetricsTest, TotalReturnsPositiveValue) {
    size_t total = cuda::memory::total();
    EXPECT_GT(total, 0);
}

TEST_F(MemoryMetricsTest, UsedIsNotGreaterThanTotal) {
    size_t used = cuda::memory::used();
    size_t total = cuda::memory::total();
    EXPECT_LE(used, total);
}

TEST_F(MemoryMetricsTest, GetMetricsReturnsValidUtilization) {
    auto metrics = cuda::memory::get_metrics();

    EXPECT_GE(metrics.utilization_percent, 0.0);
    EXPECT_LE(metrics.utilization_percent, 100.0);
    EXPECT_EQ(metrics.used_bytes, metrics.total_bytes - metrics.available_bytes);
}

TEST_F(MemoryMetricsSerialTest, GetMetricsMatchesIndividualFunctions) {
    auto metrics = cuda::memory::get_metrics();

    EXPECT_EQ(metrics.used_bytes, cuda::memory::used());
    EXPECT_EQ(metrics.available_bytes, cuda::memory::available());
    EXPECT_EQ(metrics.total_bytes, cuda::memory::total());
}
