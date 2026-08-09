#include <gtest/gtest.h>
#include <fstream>

#include "cuda/testing/integration.h"
#include "cuda/testing/memory_safety.h"
#include "cuda/testing/fp_determinism.h"
#include "cuda/testing/boundary_testing.h"
#include "cuda/observability/timeline.h"
#include "cuda/observability/bandwidth_tracker.h"

namespace cuda::testing::test {

class ProfilingIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

TEST_F(ProfilingIntegrationTest, E2ERobustnessWithProfiling) {
    auto result = run_e2e_robustness_with_profiling();

    EXPECT_TRUE(result.robust) << "E2E robustness test failed for: "
                                << result.failed_tests.size() << " tests";
    EXPECT_GE(result.tests_passed, 0u);
}

TEST_F(ProfilingIntegrationTest, MemorySafetyValidation) {
    auto result = validate_all_algorithm_memory_safety();

    EXPECT_TRUE(result.all_safe);
    EXPECT_GE(result.algorithms_validated, 0u);
}

TEST_F(ProfilingIntegrationTest, TimelineExportWithMemoryTest) {
    NOVA_TIMELINE_SCOPED("integration_test", "test");

    void* ptr = nullptr;
    cudaMalloc(&ptr, 1024);

    NOVA_TIMELINE_EXPORT("/tmp/integration_trace.json");

    cudaFree(ptr);

    std::ifstream file("/tmp/integration_trace.json");
    EXPECT_TRUE(file.good());

    std::remove("/tmp/integration_trace.json");
}

TEST_F(ProfilingIntegrationTest, BoundaryTestsWithBandwidth) {
    auto bw_result = cuda::observability::BandwidthTracker().measure_device_to_device(1024);

    EXPECT_GT(bw_result.bandwidth_gbps, 0.0);

    EXPECT_TRUE(is_warp_aligned(CUDA_WARP_SIZE));
    EXPECT_TRUE(is_memory_aligned(reinterpret_cast<const void*>(256)));
}

TEST_F(ProfilingIntegrationTest, FPDeterminismWithIntegration) {
    FPDeterminismControl::instance().set_level(DeterminismLevel::RunToRun);

    EXPECT_EQ(FPDeterminismControl::instance().level(),
              DeterminismLevel::RunToRun);

    FPDeterminismControl::instance().enable_flush_to_zero();

    EXPECT_TRUE(FPDeterminismControl::instance().is_flush_to_zero_enabled());

    FPDeterminismControl::instance().reset();
}

TEST_F(ProfilingIntegrationTest, IntegrationTestRunner) {
    IntegrationTestRunner runner;

    int counter = 0;
    runner.add_test({"test1", "Test 1", [&counter]() {
        counter = 1;
        return true;
    }});

    runner.add_test({"test2", "Test 2", [&counter]() {
        counter = 2;
        return true;
    }});

    runner.add_test({"test3", "Test 3", [&counter]() {
        counter = 3;
        return false;
    }, true});

    auto results = runner.run_all();

    EXPECT_EQ(results.size(), 3u);
    EXPECT_EQ(runner.total_tests(), 3u);

    int passed = 0;
    for (const auto& r : results) {
        if (r.passed) passed++;
    }
    EXPECT_EQ(passed, 2);
}

TEST_F(ProfilingIntegrationTest, MemorySafetyWithTimeline) {
    MemorySafetyValidator::instance().enable();

    void* ptr = nullptr;
    cudaMalloc(&ptr, 1024);

    auto valid = MemorySafetyValidator::instance().validate_allocation(ptr, 1024);
    EXPECT_TRUE(valid);

    NOVA_TIMELINE_SCOPED("memory_test", "memory");

    cudaFree(ptr);

    EXPECT_EQ(MemorySafetyValidator::instance().validation_count(), 1u);
}

}  // namespace cuda::testing::test
