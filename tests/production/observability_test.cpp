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
// under a name that says used. Measured on an 8xA100 host holding a 2 GiB allocation: the
// driver reported used=2472.8 MiB / free=78679.9 MiB, and the snapshot returned 78680.0 --
// matching free to 0.1 MiB and differing from used by 76207.
//
// This asserts the relationship against the driver directly rather than a fixed number, so it
// holds whether the GPU is idle or busy.
TEST_F(HealthMonitorTest, MemoryUsedMatchesTheDriverRatherThanTheFreeFigure) {
    // Occupy a known slice so used and free are unmistakably different.
    void* hog = nullptr;
    const bool reserved = cudaMalloc(&hog, 512ull * 1024 * 1024) == cudaSuccess;
    if (!reserved) {
        (void)cudaGetLastError();
        GTEST_SKIP() << "could not reserve scratch memory";
    }

    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        cudaFree(hog);
        GTEST_SKIP() << "cudaMemGetInfo unavailable";
    }
    const float driver_used_mb = static_cast<float>(total_bytes - free_bytes) / (1024.0f * 1024.0f);
    const float driver_free_mb = static_cast<float>(free_bytes) / (1024.0f * 1024.0f);

    cuda::production::HealthMonitor monitor;
    const auto metrics = monitor.get_health_snapshot();
    // Read BOTH accessors before releasing the scratch allocation: freeing in between moves the
    // device figure by the size of the allocation and makes the two disagree for a reason that
    // has nothing to do with the defect under test.
    const auto mem = monitor.get_memory_snapshot();
    cudaFree(hog);

    if (driver_free_mb < driver_used_mb + 64.0f) {
        GTEST_SKIP() << "used and free are too close on this host for the two to be distinguishable";
    }

    const float to_used = std::fabs(metrics.memory_used_mb - driver_used_mb);
    const float to_free = std::fabs(metrics.memory_used_mb - driver_free_mb);
    EXPECT_LT(to_used, to_free) << "memory_used_mb=" << metrics.memory_used_mb << " is closer to the FREE figure (" << driver_free_mb << " MiB) than to the used figure ("
                                << driver_used_mb << " MiB)";

    // And the two accessors in this class must agree: get_memory_snapshot() computes
    // allocated_bytes as total - free, which is the same quantity.
    EXPECT_NEAR(static_cast<float>(mem.allocated_bytes) / (1024.0f * 1024.0f), metrics.memory_used_mb, 64.0f)
        << "to_json() emits both of these, so they cannot contradict each other";
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
