#include <gtest/gtest.h>

#include "cuda/production/async_error_tracker.h"
#include "cuda/production/health_metrics.h"
#include <fstream>

namespace {

void reset() {
    cudaDeviceReset();
}

}

class AsyncErrorTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(AsyncErrorTrackerTest, DefaultConstruction) {
    cuda::production::AsyncErrorTracker tracker;
    EXPECT_TRUE(tracker.is_enabled());
    EXPECT_EQ(tracker.error_count(), 0u);
}

TEST_F(AsyncErrorTrackerTest, RecordError) {
    cuda::production::AsyncErrorTracker tracker;

    tracker.record(cudaErrorMemoryAllocation, "Test allocation");
    EXPECT_EQ(tracker.error_count(), 1u);
}

TEST_F(AsyncErrorTrackerTest, RecordSuccessNoOp) {
    cuda::production::AsyncErrorTracker tracker;

    tracker.record(cudaSuccess);
    EXPECT_EQ(tracker.error_count(), 0u);
}

TEST_F(AsyncErrorTrackerTest, GetLastError) {
    cuda::production::AsyncErrorTracker tracker;

    tracker.record(cudaErrorInvalidValue, "Test error");
    auto last = tracker.get_last_error();

    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(last->error, cudaErrorInvalidValue);
}

TEST_F(AsyncErrorTrackerTest, GetErrors) {
    cuda::production::AsyncErrorTracker tracker;

    tracker.record(cudaErrorMemoryAllocation, "Error 1");
    tracker.record(cudaErrorInvalidValue, "Error 2");

    auto errors = tracker.get_errors();
    EXPECT_EQ(errors.size(), 2u);
}

TEST_F(AsyncErrorTrackerTest, Clear) {
    cuda::production::AsyncErrorTracker tracker;

    tracker.record(cudaErrorMemoryAllocation);
    EXPECT_EQ(tracker.error_count(), 1u);

    tracker.clear();
    EXPECT_EQ(tracker.error_count(), 0u);
}

TEST_F(AsyncErrorTrackerTest, SetEnabled) {
    cuda::production::AsyncErrorTracker tracker;

    tracker.set_enabled(false);
    EXPECT_FALSE(tracker.is_enabled());

    tracker.record(cudaErrorMemoryAllocation);
    EXPECT_EQ(tracker.error_count(), 0u);

    tracker.set_enabled(true);
    tracker.record(cudaErrorMemoryAllocation);
    EXPECT_EQ(tracker.error_count(), 1u);
}

TEST_F(AsyncErrorTrackerTest, ScopedErrorTracking) {
    cuda::production::AsyncErrorTracker tracker;

    {
        cuda::production::ScopedErrorTracking scope(tracker);
        int* p = nullptr;
        cudaMalloc(&p, 1);
    }

    EXPECT_GT(tracker.error_count(), 0u);
}

class HealthMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(HealthMonitorTest, GetHealthSnapshot) {
    cuda::production::HealthMonitor monitor;

    auto metrics = monitor.get_health_snapshot();

    EXPECT_GE(metrics.device_id, 0u);
    EXPECT_GE(metrics.memory_total_mb, 0.0f);
}

TEST_F(HealthMonitorTest, GetMemorySnapshot) {
    cuda::production::HealthMonitor monitor;

    auto mem = monitor.get_memory_snapshot();

    EXPECT_GE(mem.allocated_bytes, 0u);
    EXPECT_GE(mem.reserved_bytes, 0u);
}

TEST_F(HealthMonitorTest, ToJSON) {
    cuda::production::HealthMonitor monitor;

    auto json = monitor.to_json();

    EXPECT_TRUE(json.find("device_id") != std::string::npos);
    EXPECT_TRUE(json.find("memory_total_mb") != std::string::npos);
    EXPECT_TRUE(json.find("timestamp_ns") != std::string::npos);
}

TEST_F(HealthMonitorTest, ToCSV) {
    cuda::production::HealthMonitor monitor;

    auto csv = monitor.to_csv();

    EXPECT_TRUE(csv.find(",") != std::string::npos);
}

TEST_F(HealthMonitorTest, RecordError) {
    cuda::production::HealthMonitor monitor;

    EXPECT_EQ(monitor.error_count_24h(), 0u);

    monitor.record_error();
    EXPECT_EQ(monitor.error_count_24h(), 1u);

    monitor.reset_error_count();
    EXPECT_EQ(monitor.error_count_24h(), 0u);
}
