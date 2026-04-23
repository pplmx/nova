#include <gtest/gtest.h>
#include "cuda/benchmark/benchmark.h"

class RegressionDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(RegressionDetectionTest, NoRegressionWhenPerformanceMatchesBaseline) {
    cuda::benchmark::Benchmark bench("exact_match");

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 10.0;
    baseline.stddev_ms = 0.5;
    baseline.samples = 10;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 10.0;
    current.stddev_ms = 0.5;
    current.samples = 10;

    EXPECT_TRUE(bench.compare_to_baseline(current));
}

TEST_F(RegressionDetectionTest, RegressionDetectedAtExactTolerance) {
    cuda::benchmark::BenchmarkOptions options;
    options.tolerance_percent = 10.0;
    cuda::benchmark::Benchmark bench("at_tolerance", options);

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 100.0;
    baseline.samples = 10;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 110.0;
    current.samples = 10;

    EXPECT_TRUE(bench.compare_to_baseline(current));
}

TEST_F(RegressionDetectionTest, RegressionDetectedBeyondTolerance) {
    cuda::benchmark::BenchmarkOptions options;
    options.tolerance_percent = 5.0;
    cuda::benchmark::Benchmark bench("beyond_tolerance", options);

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 100.0;
    baseline.samples = 10;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 106.0;
    current.samples = 10;

    EXPECT_FALSE(bench.compare_to_baseline(current));
}

TEST_F(RegressionDetectionTest, FormatReportShowsPositiveDelta) {
    cuda::benchmark::Benchmark bench("positive_delta");

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 100.0;
    baseline.stddev_ms = 5.0;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 120.0;
    current.stddev_ms = 5.0;

    std::string report = bench.format_regression_report(current);

    EXPECT_TRUE(report.find("+") != std::string::npos);
    EXPECT_TRUE(report.find("120") != std::string::npos);
}

TEST_F(RegressionDetectionTest, FormatReportShowsNegativeDelta) {
    cuda::benchmark::Benchmark bench("negative_delta");

    cuda::benchmark::BenchmarkResult baseline;
    baseline.mean_ms = 100.0;
    baseline.stddev_ms = 5.0;
    bench.set_baseline(baseline);

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 80.0;
    current.stddev_ms = 5.0;

    std::string report = bench.format_regression_report(current);

    EXPECT_TRUE(report.find("-20") != std::string::npos);
}

TEST_F(RegressionDetectionTest, FormatReportWithoutBaseline) {
    cuda::benchmark::Benchmark bench("no_baseline");

    cuda::benchmark::BenchmarkResult current;
    current.mean_ms = 100.0;

    std::string report = bench.format_regression_report(current);

    EXPECT_TRUE(report.find("No baseline set") != std::string::npos);
}
