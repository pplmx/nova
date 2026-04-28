#include <gtest/gtest.h>

#include "cuda/production/error_injection.h"

namespace {

void reset() {
    cudaDeviceReset();
}

}

class ErrorInjectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
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

    EXPECT_THROW(
        {
            cuda::production::ScopedErrorInjection scope(
                injector,
                cuda::production::ErrorTarget::Allocation,
                cudaErrorMemoryAllocation);
        },
        cuda::production::device::CudaException);
}

TEST_F(ErrorInjectionTest, CountTracking) {
    cuda::production::ErrorInjector injector;

    injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);

    injector.should_inject(cuda::production::ErrorTarget::Allocation);
    injector.should_inject(cuda::production::ErrorTarget::Allocation);

    EXPECT_GE(injector.injection_count(cuda::production::ErrorTarget::Allocation), 1u);
}

class MemoryPressureTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
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

TEST_F(MemoryPressureTest, IsUnderPressure) {
    cuda::production::MemoryPressureTest pressure(1000);

    EXPECT_FALSE(pressure.is_under_pressure());

    pressure.allocate(900);
    EXPECT_TRUE(pressure.is_under_pressure());
}

TEST_F(MemoryPressureTest, Release) {
    cuda::production::MemoryPressureTest pressure(1024);

    pressure.allocate(512);
    EXPECT_EQ(pressure.remaining(), 512u);

    pressure.release();
    EXPECT_EQ(pressure.remaining(), 1024u);
}

TEST_F(MemoryPressureTest, SetLimit) {
    cuda::production::MemoryPressureTest pressure(1024);

    pressure.allocate(512);
    EXPECT_EQ(pressure.remaining(), 512u);

    pressure.set_limit(2048);
    EXPECT_EQ(pressure.remaining(), 1536u);
}

class StressTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
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
