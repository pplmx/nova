#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cuda_alloc_injection.h"
#include "cuda/production/error_injection.h"
#include "cuda/device/error.h"

namespace {

    void reset() {
        // Was cudaDeviceReset(): that tears down the whole process context and leaves
        // every singleton's cached handles stale, which libcuda dereferences rather than
        // reporting when they are used again (measured: cuEventDestroy on a handle created
        // before the reset segfaults). Synchronising gives these fixtures what they
        // actually want -- no work still in flight from the previous case.
        (void)cudaDeviceSynchronize();
    }

}  // namespace

class ErrorInjectionTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(ErrorInjectionTest, DefaultConstruction) {
    cuda::production::ErrorInjector injector;
    EXPECT_EQ(injector.total_injection_count(), 0u);
}

TEST_F(ErrorInjectionTest, InjectAlways) {
    cuda::production::ErrorInjector injector;

    injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);

    EXPECT_TRUE(injector.should_inject(cuda::production::ErrorTarget::Allocation));
    EXPECT_EQ(injector.get_error(cuda::production::ErrorTarget::Allocation), cudaErrorMemoryAllocation);
}

TEST_F(ErrorInjectionTest, InjectOnce) {
    cuda::production::ErrorInjector injector;

    injector.inject_once(cuda::production::ErrorTarget::Launch, cudaErrorLaunchFailure);

    EXPECT_TRUE(injector.should_inject(cuda::production::ErrorTarget::Launch));
}

TEST_F(ErrorInjectionTest, Disable) {
    cuda::production::ErrorInjector injector;

    injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);
    injector.disable();

    EXPECT_FALSE(injector.should_inject(cuda::production::ErrorTarget::Allocation));
}

TEST_F(ErrorInjectionTest, Reset) {
    cuda::production::ErrorInjector injector;

    injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);
    injector.inject_always(cuda::production::ErrorTarget::Launch, cudaErrorLaunchFailure);

    injector.reset();

    EXPECT_FALSE(injector.should_inject(cuda::production::ErrorTarget::Allocation));
    EXPECT_FALSE(injector.should_inject(cuda::production::ErrorTarget::Launch));
}

TEST_F(ErrorInjectionTest, ScopedErrorInjection) {
    cuda::production::ErrorInjector injector;

    injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);

    EXPECT_THROW({ cuda::production::ScopedErrorInjection scope(injector, cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation); }, cuda::device::CudaException);
}

TEST_F(ErrorInjectionTest, CountTracking) {
    cuda::production::ErrorInjector injector;

    injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);

    // Called for the side effect (each query bumps injection_count), not for the answer.
    (void)injector.should_inject(cuda::production::ErrorTarget::Allocation);
    (void)injector.should_inject(cuda::production::ErrorTarget::Allocation);

    EXPECT_GE(injector.injection_count(cuda::production::ErrorTarget::Allocation), 1u);
}

class MemoryPressureTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(MemoryPressureTest, DefaultConstruction) {
    cuda::production::MemoryPressureTest pressure(1024 * 1024);
    EXPECT_EQ(pressure.remaining(), 1024 * 1024u);
}

TEST_F(MemoryPressureTest, Allocate) {
    cuda::production::MemoryPressureTest pressure(1024 * 1024);

    EXPECT_TRUE(pressure.allocate(512 * 1024));
    EXPECT_EQ(pressure.remaining(), 512 * 1024u);
}

TEST_F(MemoryPressureTest, AllocationLimit) {
    cuda::production::MemoryPressureTest pressure(1024);

    EXPECT_TRUE(pressure.allocate(512));
    EXPECT_TRUE(pressure.allocate(512));
    EXPECT_FALSE(pressure.allocate(1));
}

// Regression: allocate() was a load followed by an unconditional fetch_add, so concurrent
// callers could all read the same used_bytes_, all pass the limit check, and all reserve. The
// existing cases here are all single-threaded and therefore cannot see it. Measured with 128
// threads released from a barrier, a 1000-byte limit and 100 bytes each: one run granted 56
// allocations where at most 10 fit.
//
// The threads are released together deliberately -- staggering them lets each finish its
// check-and-add before the next starts, and the defect disappears.
TEST_F(MemoryPressureTest, ConcurrentAllocationNeverOverCommitsTheLimit) {
    constexpr size_t kLimit = 1000;
    constexpr size_t kChunk = 100;  // at most 10 can fit
    constexpr size_t kMaxGrants = kLimit / kChunk;
    constexpr int kThreads = 128;

    for (int trial = 0; trial < 8; ++trial) {
        cuda::production::MemoryPressureTest pressure(kLimit);
        std::atomic<int> granted {0};
        std::atomic<int> ready {0};
        std::atomic<bool> go {false};

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&] {
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                if (pressure.allocate(kChunk)) {
                    granted.fetch_add(1);
                }
            });
        }

        while (ready.load() < kThreads) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);
        for (auto& thread : threads) {
            thread.join();
        }

        EXPECT_LE(static_cast<size_t>(granted.load()), kMaxGrants)
            << "trial " << trial << ": the budget admits at most " << kMaxGrants << " allocations of " << kChunk << " bytes, but " << granted.load() << " were granted";
        EXPECT_EQ(pressure.remaining(), kLimit - kChunk * static_cast<size_t>(granted.load())) << "trial " << trial << ": remaining must match what was granted";
    }
}

// The overflow guard: a request larger than the whole budget must be refused even though
// `used + bytes` would wrap.
TEST_F(MemoryPressureTest, AllocationRefusesASizeThatWouldOverflowTheLimit) {
    cuda::production::MemoryPressureTest pressure(1024);
    EXPECT_TRUE(pressure.allocate(512));

    EXPECT_FALSE(pressure.allocate(std::numeric_limits<size_t>::max())) << "a wrapping size must be refused, not accepted";
    EXPECT_FALSE(pressure.allocate(std::numeric_limits<size_t>::max() - 100 + 1));
    EXPECT_EQ(pressure.remaining(), 512u) << "a refused allocation must not change the accounting";
}

TEST_F(MemoryPressureTest, IsUnderPressure) {
    cuda::production::MemoryPressureTest pressure(1000);

    EXPECT_FALSE(pressure.is_under_pressure());

    EXPECT_TRUE(pressure.allocate(900)) << "the allocation fits under the limit";
    EXPECT_TRUE(pressure.is_under_pressure());
}

TEST_F(MemoryPressureTest, Release) {
    cuda::production::MemoryPressureTest pressure(1024);

    EXPECT_TRUE(pressure.allocate(512));
    EXPECT_EQ(pressure.remaining(), 512u);

    pressure.release();
    EXPECT_EQ(pressure.remaining(), 1024u);
}

TEST_F(MemoryPressureTest, SetLimit) {
    cuda::production::MemoryPressureTest pressure(1024);

    EXPECT_TRUE(pressure.allocate(512));
    EXPECT_EQ(pressure.remaining(), 512u);

    pressure.set_limit(2048);
    EXPECT_EQ(pressure.remaining(), 1536u);
}

class StressTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(StressTest, MemoryPressureTest) {
    cuda::production::StressTestConfig config;
    config.max_allocations = 100;
    config.allocation_size = 1024 * 1024;

    EXPECT_TRUE(cuda::production::run_memory_pressure_test(config));
}

TEST_F(StressTest, ConcurrentStreamTest) {
    cuda::production::StressTestConfig config;
    config.max_concurrent_streams = 8;
    config.allocation_size = 1024 * 1024;

    EXPECT_TRUE(cuda::production::run_concurrent_stream_test(config));
}

// run_concurrent_stream_test allocates one buffer per stream in a loop, each guarded by CUDA_CHECK,
// and frees them in a separate loop after every stream has been synchronised. A cudaMalloc that
// fails part way through -- or a synchronize that throws -- leaves the buffers taken before it with
// no owner, because the free loop is never reached.
TEST_F(StressTest, AFailedStreamAllocationReleasesTheBuffersAlreadyTaken) {
    cuda::production::StressTestConfig config;
    config.max_concurrent_streams = 8;
    config.allocation_size = 1024 * 1024;

    cuda_test::reset_allocation_count();

    // Let two buffers through, fail the third.
    cuda_test::fail_allocation_after(2);
    EXPECT_THROW((void)cuda::production::run_concurrent_stream_test(config), std::exception);
    cuda_test::stop_failing_allocations();

    EXPECT_EQ(cuda_test::live_allocation_count(), 0) << "the buffers taken before the failure must be released, not left with no owner";
}
