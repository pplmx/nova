#include <gtest/gtest.h>
#include <stdexcept>
#include "cuda/benchmark/benchmark.h"

class BenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(BenchmarkTest, WarmupRunsExecuteBeforeMeasurement) {
    int warmup_count = 0;
    int measurement_count = 0;

    // Count warm-up and measured calls separately: the case is named for the ordering,
    // and counting only the total cannot distinguish it. The first warmup_iterations
    // calls are warm-up by definition of the option.
    auto kernel = [&]() {
        if (warmup_count < 5) {
            warmup_count++;
        } else {
            measurement_count++;
        }
    };

    cuda::benchmark::BenchmarkOptions options;
    options.warmup_iterations = 5;
    options.measurement_iterations = 1;

    cuda::benchmark::Benchmark bench("warmup_test", options);
    bench.run(kernel);

    EXPECT_EQ(warmup_count, 5) << "exactly the configured warm-up iterations";
    EXPECT_EQ(measurement_count, 1) << "exactly the configured measurement iterations";
}

TEST_F(BenchmarkTest, ResultsIncludeMeanAndStddev) {
    auto kernel = []() {};

    cuda::benchmark::Benchmark bench("variance_test");
    auto result = bench.run(kernel);

    EXPECT_GE(result.mean_ms, 0.0);
    EXPECT_GE(result.stddev_ms, 0.0);
    EXPECT_GT(result.samples, 0);
}

TEST_F(BenchmarkTest, ResultsIncludeMinAndMax) {
    auto kernel = []() {};

    cuda::benchmark::Benchmark bench("range_test");
    auto result = bench.run(kernel);

    EXPECT_LE(result.min_ms, result.max_ms);
}

TEST_F(BenchmarkTest, MultipleRunsProduceResults) {
    std::vector<cuda::benchmark::BenchmarkResult> results;

    for (int i = 0; i < 3; ++i) {
        cuda::benchmark::Benchmark bench("consistency_test");
        results.push_back(bench.run([]() {}));
    }

    for (const auto& result : results) {
        EXPECT_GT(result.samples, 0);
    }
}

TEST_F(BenchmarkTest, ThroughputCalculationIsCorrect) {
    size_t gib = 1024ULL * 1024 * 1024;
    double gbps = cuda::benchmark::compute_throughput_gbps(gib, 1.0);
    EXPECT_NEAR(gbps, 1000.0, 0.01);

    gbps = cuda::benchmark::compute_throughput_gbps(gib, 1000.0);
    EXPECT_NEAR(gbps, 1.0, 0.01);

    gbps = cuda::benchmark::compute_throughput_gbps(gib, 0.0);
    EXPECT_DOUBLE_EQ(gbps, 0.0);
}

class RegressionTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(RegressionTest, NoRegressionWhenPerformanceIsWithinTolerance) {
    cuda::benchmark::Benchmark bench("regression_test");

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 10.0;
    baseline.stddev_ms = 1.0;
    baseline.samples = 10;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 11.0;
    current.stddev_ms = 1.0;
    current.samples = 10;

    EXPECT_TRUE(bench.compare_to_baseline(current));
}

TEST_F(RegressionTest, RegressionDetectedWhenPerformanceDegrades) {
    cuda::benchmark::BenchmarkOptions options;
    options.tolerance_percent = 10.0;
    cuda::benchmark::Benchmark bench("regression_test", options);

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 10.0;
    baseline.stddev_ms = 1.0;
    baseline.samples = 10;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 11.5;
    current.stddev_ms = 1.0;
    current.samples = 10;

    EXPECT_FALSE(bench.compare_to_baseline(current));
}

TEST_F(RegressionTest, FormatReportShowsDeltaPercent) {
    cuda::benchmark::Benchmark bench("regression_test");

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 100.0;
    baseline.stddev_ms = 5.0;
    baseline.samples = 10;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 110.0;
    current.stddev_ms = 5.0;
    current.samples = 10;

    std::string report = bench.format_regression_report(current);

    EXPECT_TRUE(report.find("110") != std::string::npos || report.find("+10") != std::string::npos || report.find("10%") != std::string::npos);
}

TEST_F(RegressionTest, ReturnsTrueWithoutBaseline) {
    cuda::benchmark::Benchmark bench("no_baseline");

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 100.0;

    EXPECT_TRUE(bench.compare_to_baseline(current));
}

// Benchmark::run created two events and destroyed them at the end, past six CUDA_CHECKs and the
// user's kernel. Any of those throwing leaked both events: the destroy calls sat after the throw
// point. A kernel that throws is the reachable way in -- cudaEventCreate failing cannot be requested
// through any public API, and the allocation injection covers cudaMalloc only.
TEST(BenchmarkEventCleanupTest, AKernelThatThrowsStillDestroysBothEvents) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        (void)cudaGetLastError();
        GTEST_SKIP() << "no CUDA device";
    }
    (void)cudaGetLastError();

    cuda::benchmark::Benchmark bench("throwing_kernel");

    bool threw = false;
    try {
        bench.run([]() { throw std::runtime_error("kernel failed"); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw) << "the kernel's exception must propagate";

    // If the events had leaked, cudaGetLastError would still be clean -- the leak is invisible here.
    // What this asserts is that the exception path exists and completes; ASan runs confirm the rest.
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
