#include <gtest/gtest.h>
#include <cmath>

#include "cuda/observability/bandwidth_tracker.h"

namespace cuda::observability::test {

class BandwidthTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(BandwidthTrackerTest, MeasureHostToDevice) {
    BandwidthTracker tracker;
    auto result = tracker.measure_host_to_device(1024 * 1024);

    EXPECT_GT(result.bandwidth_gbps, 0.0);
    EXPECT_EQ(result.bytes_transferred, 1024 * 1024);
    EXPECT_GT(result.elapsed_ms, 0.0);
    EXPECT_EQ(result.type, MemoryTransferType::HostToDevice);
}

TEST_F(BandwidthTrackerTest, MeasureDeviceToHost) {
    BandwidthTracker tracker;
    auto result = tracker.measure_device_to_host(1024 * 1024);

    EXPECT_GT(result.bandwidth_gbps, 0.0);
    EXPECT_EQ(result.bytes_transferred, 1024 * 1024);
    EXPECT_EQ(result.type, MemoryTransferType::DeviceToHost);
}

TEST_F(BandwidthTrackerTest, MeasureDeviceToDevice) {
    BandwidthTracker tracker;
    auto result = tracker.measure_device_to_device(1024 * 1024);

    EXPECT_GT(result.bandwidth_gbps, 0.0);
    EXPECT_EQ(result.bytes_transferred, 1024 * 1024);
    EXPECT_EQ(result.type, MemoryTransferType::DeviceToDevice);
}

TEST_F(BandwidthTrackerTest, BandwidthDecreasesWithSize) {
    BandwidthTracker tracker;

    auto small = tracker.measure_host_to_device(64 * 1024);
    auto large = tracker.measure_host_to_device(16 * 1024 * 1024);

    EXPECT_GT(large.bandwidth_gbps, small.bandwidth_gbps * 0.5);
}

TEST_F(BandwidthTrackerTest, DeviceMemoryBandwidthQuery) {
    auto bandwidth = DeviceMemoryBandwidth::query(0);

    EXPECT_GE(bandwidth.h2d_gbps, 0.0);
    EXPECT_GE(bandwidth.d2h_gbps, 0.0);
    EXPECT_GE(bandwidth.d2d_gbps, 0.0);
}

TEST_F(BandwidthTrackerTest, TrackerAccumulation) {
    BandwidthTracker tracker;

    tracker.measure_host_to_device(1024);
    tracker.measure_device_to_host(2048);
    tracker.measure_device_to_device(4096);

    EXPECT_EQ(tracker.total_bytes_transferred(), 1024 + 2048 + 4096);
    EXPECT_GT(tracker.total_time_ns(), 0);
}

TEST_F(BandwidthTrackerTest, ResetClearsAccumulation) {
    BandwidthTracker tracker;

    tracker.measure_host_to_device(1024);
    EXPECT_GT(tracker.total_bytes_transferred(), 0);

    tracker.reset();
    EXPECT_EQ(tracker.total_bytes_transferred(), 0);
    EXPECT_EQ(tracker.total_time_ns(), 0);
}

}  // namespace cuda::observability::test
