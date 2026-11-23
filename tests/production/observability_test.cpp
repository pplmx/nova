#include <gtest/gtest.h>

#include "cuda/production/async_error_tracker.h"
#include "cuda/production/health_metrics.h"
#include <fstream>

namespace {

    void reset() {
        (void)cudaDeviceSynchronize();
    }

}  // namespace

class AsyncErrorTrackerTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
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
        // Force a real CUDA error inside the scope. The previous version relied
        // on some stray sticky error left by an earlier test, which made this
        // test pass or fail depending on process order (it failed standalone).
        int* p = nullptr;
        (void)cudaMalloc(&p, static_cast<size_t>(-1));
    }  // scope dtor -> tracker.check() records the failed allocation

    EXPECT_GT(tracker.error_count(), 0u);

    // Drain the sticky error left by the deliberately failed allocation so this
    // negative-path test cannot poison later GPU work in the same process.
    (void)cudaGetLastError();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}

class HealthMonitorTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

class HealthMonitorSerialTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(HealthMonitorTest, GetHealthSnapshot) {
    cuda::production::HealthMonitor monitor;

    auto metrics = monitor.get_health_snapshot();

    EXPECT_GE(metrics.device_id, 0u);
    EXPECT_GE(metrics.memory_total_mb, 0.0f);
    // The case never looked at memory_used_mb, which is how it stayed wrong.
    EXPECT_GE(metrics.memory_used_mb, 0.0f);
    EXPECT_LE(metrics.memory_used_mb, metrics.memory_total_mb) << "used memory cannot exceed the device total";
}

// Regression: memory_used_mb was assigned three times -- (total - free), then total, then
// `memory_total_mb - memory_used_mb`, which evaluates to FREE. The field reported free memory
// under a name that says used. Measured on an 8xA100 host holding a 2 GiB allocation: the driver
// reported used=2472.8 MiB / free=78679.9 MiB, and the snapshot returned 78680.0 -- matching free
// to 0.1 MiB and differing from used by 76207.
//
// Two earlier versions of this case tried to compare against absolute figures and both failed in
// full-suite runs while passing alone. The first used a fixed 64 MiB tolerance; the second
// asserted the reported value was under half the device total, which breaks down because other
// GPU cases hold memory concurrently -- measured, six concurrent 6 GiB allocations put the real
// used figure at 46% of the device, close enough to the threshold that the seventh crosses it.
//
// This version measures the thing the defect actually gets wrong. It records the reported figure,
// takes a large allocation, records it again, and requires the reported figure to have moved by
// roughly the amount allocated. A snapshot reporting FREE memory moves the wrong way -- it goes
// down as memory is consumed -- so the sign of the change distinguishes the two implementations
// regardless of what the rest of the machine is doing.
// Moved out of HealthMonitorTest into a serial suite: this case reads device-wide memory
// snapshots, and cudaMemGetInfo aggregates every process on the device, so a concurrent test's
// allocation is indistinguishable from the counter it is checking. See the RUN_SERIAL
// registration in tests/CMakeLists.txt.
TEST_F(HealthMonitorSerialTest, MemoryUsedTracksAllocationRatherThanMovingWithFreeMemory) {
    cuda::production::HealthMonitor monitor;

    // Baseline. Nothing can make the difference between two readings of the same field as large
    // as the allocation below unless the field is actually tracking allocation.
    const size_t hold_bytes = 2ull * 1024 * 1024 * 1024;  // 2 GiB
    const float hold_mb = static_cast<float>(hold_bytes / (1024 * 1024));

    const float before = monitor.get_health_snapshot().memory_used_mb;

    void* hog = nullptr;
    if (cudaMalloc(&hog, hold_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        GTEST_SKIP() << "could not reserve 2 GiB to measure the delta";
    }
    const float after = monitor.get_health_snapshot().memory_used_mb;
    cudaFree(hog);

    const float delta = after - before;
    RecordProperty("used_before_mb", std::to_string(before));
    RecordProperty("used_after_mb", std::to_string(after));
    RecordProperty("delta_mb", std::to_string(delta));

    // A correct implementation reports this allocation, so the figure must rise by about its
    // size. Other processes may also allocate or free meanwhile, which is why the window is wide
    // rather than exact -- but no amount of unrelated churn makes the figure rise by ~2 GiB
    // unless it is tracking allocation.
    EXPECT_GT(delta, hold_mb * 0.5f) << "memory_used_mb went from " << before << " to " << after << " MiB while this case allocated " << hold_mb
                                     << " MiB; a field reporting USED memory must rise by about that much";

    // And the direction is the giveaway: reporting free memory means the figure FALLS when memory
    // is consumed. This assertion is the one that fails against the original bug.
    EXPECT_GE(delta, 0.0f) << "memory_used_mb fell from " << before << " to " << after << " MiB when memory was consumed, which is what a field reporting FREE memory does";
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
