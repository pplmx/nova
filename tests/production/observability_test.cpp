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
// under a name that says used. Measured on an 8xA100 host holding a 2 GiB allocation: the driver
// reported used=2472.8 MiB / free=78679.9 MiB, and the snapshot returned 78680.0 -- matching free
// to 0.1 MiB and differing from used by 76207.
//
// Earlier versions of this case compared two live cudaMemGetInfo readings with a tolerance, and
// failed in full-suite runs under load while passing alone: other GPU cases and other tenants
// allocate between any two readings. Widening the tolerance did not fix that, because the drift
// was not bounded. This version asserts a property that no amount of drift can violate -- the
// buggy implementation reported the FREE figure, which for any device that is not essentially
// full is close to the TOTAL, whereas a correct one reports a small fraction of it.
TEST_F(HealthMonitorTest, MemoryUsedMatchesTheDriverRatherThanTheFreeFigure) {
    // Hold a known slice so the true used figure is at least this large.
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

    cuda::production::HealthMonitor monitor;
    const auto metrics = monitor.get_health_snapshot();
    const auto mem = monitor.get_memory_snapshot();
    cudaFree(hog);

    const float total_mb = static_cast<float>(total_bytes / (1024 * 1024));
    const float reported = metrics.memory_used_mb;

    // Reporting free memory puts the figure near the device total.
    EXPECT_LT(reported, total_mb * 0.5f) << "memory_used_mb=" << reported << " MiB is at least half of the " << total_mb
                                         << " MiB device total, which is what reporting FREE memory looks like";
    // Reporting correctly puts it at least above the 512 MiB this case holds.
    EXPECT_GE(reported, 256.0f) << "memory_used_mb=" << reported << " MiB is below the 512 MiB held by this case, so the figure is not tracking allocation";

    // The two accessors describe the same quantity; compare as a ratio so absolute drift cancels.
    const double allocated_mb = static_cast<double>(mem.allocated_bytes) / (1024.0 * 1024.0);
    if (allocated_mb > 256.0) {
        const double ratio = static_cast<double>(reported) / allocated_mb;
        EXPECT_GT(ratio, 0.25) << "memory_used_mb=" << reported << " MiB against get_memory_snapshot's " << allocated_mb << " MiB allocated";
        EXPECT_LT(ratio, 4.0) << "memory_used_mb=" << reported << " MiB against get_memory_snapshot's " << allocated_mb << " MiB allocated";
    }
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
