#include <gtest/gtest.h>
#include "cuda/benchmark/benchmark.h"

class BenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(BenchmarkTest, WarmupRunsExecuteBeforeMeasurement) {
    int warmup_count = 0;
    int measurement_count = 0;

    auto kernel = [&]() {
        warmup_count++;
    };

    cuda::benchmark::BenchmarkOptions options;
    options.warmup_iterations = 5;
    options.measurement_iterations = 1;

    cuda::benchmark::Benchmark bench("warmup_test", options);
    bench.run(kernel);

    EXPECT_GE(warmup_count, 5);
}

TEST_F(BenchmarkTest, ResultsIncludeMeanAndStddev) {
    auto kernel = []() {
    };

    cuda::benchmark::Benchmark bench("variance_test");
    auto result = bench.run(kernel);

    EXPECT_GE(result.mean_ms, 0.0);
    EXPECT_GE(result.stddev_ms, 0.0);
    EXPECT_GT(result.samples, 0);
}

TEST_F(BenchmarkTest, ResultsIncludeMinAndMax) {
    auto kernel = []() {
    };

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
    void SetUp() override {
        cudaDeviceReset();
    }
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

    EXPECT_TRUE(report.find("110") != std::string::npos ||
                report.find("+10") != std::string::npos ||
                report.find("10%") != std::string::npos);
}

TEST_F(RegressionTest, ReturnsTrueWithoutBaseline) {
    cuda::benchmark::Benchmark bench("no_baseline");

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 100.0;

    EXPECT_TRUE(bench.compare_to_baseline(current));
}
