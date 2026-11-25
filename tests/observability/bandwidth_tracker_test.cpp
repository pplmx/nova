#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

#include "cuda_alloc_injection.h"
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

    // measure_transfer() takes its two buffers, creates two CUDA events, records and runs the copy,
    // and only then destroys the events and frees the buffers. Every step in between is a CUDA_CHECK,
    // so a failure there exits without reaching the frees: the buffers leak, and a failure on the
    // second event leaks the first event too.
    //
    // Reachable through fail_event_after(), which the wrapper could not do before -- it counted
    // event creations but ignored fail_allocation_after(), so no case could make one fail.
    TEST_F(BandwidthTrackerTest, AFailedEventCreateReleasesTheBuffersAndTheFirstEvent) {
        BandwidthTracker tracker;

        cuda_test::reset_allocation_count();
        cuda_test::reset_event_count();

        // Let the buffers and the first event through, fail the second event.
        cuda_test::fail_event_after(1);
        EXPECT_THROW((void)tracker.measure_host_to_device(1024 * 1024), std::exception);
        cuda_test::stop_failing_events();

        EXPECT_EQ(cuda_test::live_allocation_count(), 0) << "the buffers taken before the failure must be released";
        EXPECT_EQ(cuda_test::live_event_count(), 0) << "the event created before the failure must be destroyed";
    }

    TEST_F(BandwidthTrackerTest, AFailedFirstEventCreateLeavesNothingBehind) {
        BandwidthTracker tracker;

        cuda_test::reset_allocation_count();
        cuda_test::reset_event_count();

        cuda_test::fail_event_after(0);
        EXPECT_THROW((void)tracker.measure_host_to_device(1024 * 1024), std::exception);
        cuda_test::stop_failing_events();

        EXPECT_EQ(cuda_test::live_allocation_count(), 0) << "the buffers taken before the failure must be released";
        EXPECT_EQ(cuda_test::live_event_count(), 0);
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
