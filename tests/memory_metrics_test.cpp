#include <gtest/gtest.h>
#include "cuda/performance/memory_metrics.h"

class MemoryMetricsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
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

TEST_F(MemoryMetricsTest, GetMetricsMatchesIndividualFunctions) {
    auto metrics = cuda::memory::get_metrics();

    EXPECT_EQ(metrics.used_bytes, cuda::memory::used());
    EXPECT_EQ(metrics.available_bytes, cuda::memory::available());
    EXPECT_EQ(metrics.total_bytes, cuda::memory::total());
}
