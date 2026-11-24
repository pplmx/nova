#include <cuda/quantize/benchmark.hpp>

#include "cuda/device/error.h"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <cuda_runtime.h>

namespace nova {
    namespace quantize {
        namespace benchmark {
            namespace test {

                class BenchmarkResultTest : public ::testing::Test {};

                TEST_F(BenchmarkResultTest, DefaultInitialization) {
                    BenchmarkResult result;
                    EXPECT_EQ(result.name, "");
                    EXPECT_EQ(result.throughput_gbps, 0.0f);
                    EXPECT_EQ(result.latency_us, 0.0f);
                    EXPECT_EQ(result.accuracy_loss, 0.0f);
                    EXPECT_EQ(result.relative_error, 0.0f);
                    EXPECT_EQ(result.batch_size, 0);
                    EXPECT_EQ(result.num_samples, 0);
                }

                TEST_F(BenchmarkResultTest, PrintFormatsOutput) {
                    BenchmarkResult result;
                    result.name = "Test Benchmark";
                    result.throughput_gbps = 100.5f;
                    result.latency_us = 25.3f;
                    result.relative_error = 0.001f;

                    testing::internal::CaptureStdout();
                    result.print();
                    std::string output = testing::internal::GetCapturedStdout();

                    EXPECT_NE(output.find("Test Benchmark"), std::string::npos);
                    EXPECT_NE(output.find("100.50"), std::string::npos);
                    EXPECT_NE(output.find("25.30"), std::string::npos);
                }

                TEST_F(BenchmarkResultTest, SaveJsonValidFormat) {
                    BenchmarkResult result;
                    result.name = "Test";
                    result.throughput_gbps = 100.0f;
                    result.latency_us = 50.0f;
                    result.relative_error = 0.01f;
                    result.batch_size = 1024;
                    result.num_samples = 100;

                    std::ostringstream oss;
                    result.save_json(oss);

                    std::string json = oss.str();
                    EXPECT_NE(json.find("\"name\": \"Test\""), std::string::npos);
                    EXPECT_NE(json.find("\"throughput_gbps\": 100"), std::string::npos);
                    EXPECT_NE(json.find("\"latency_us\": 50"), std::string::npos);
                    EXPECT_NE(json.find("\"relative_error\": 0.01"), std::string::npos);
                    EXPECT_NE(json.find("\"batch_size\": 1024"), std::string::npos);
                    EXPECT_NE(json.find("\"num_samples\": 100"), std::string::npos);
                }

                class ComputeErrorTest : public ::testing::Test {};

                TEST_F(ComputeErrorTest, L2ErrorIdenticalVectorsReturnsZero) {
                    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
                    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};

                    QuantizationBenchmark bench;
                    float l2 = bench.compute_l2_error(a, b);

                    EXPECT_NEAR(l2, 0.0f, 1e-6f);
                }

                TEST_F(ComputeErrorTest, L2ErrorDifferentVectorsPositive) {
                    std::vector<float> a = {1.0f, 2.0f, 3.0f};
                    std::vector<float> b = {1.1f, 2.1f, 3.1f};

                    QuantizationBenchmark bench;
                    float l2 = bench.compute_l2_error(a, b);

                    EXPECT_GT(l2, 0.0f);
                    EXPECT_NEAR(l2, std::sqrt(0.03f), 0.01f);
                }

                TEST_F(ComputeErrorTest, L2ErrorMismatchedSizesUsesMin) {
                    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                    std::vector<float> b = {1.0f, 2.0f, 3.0f};

                    QuantizationBenchmark bench;
                    float l2 = bench.compute_l2_error(a, b);

                    EXPECT_NEAR(l2, 0.0f, 1e-6f);
                }

                TEST_F(ComputeErrorTest, KLDivergenceIdenticalReturnsZero) {
                    std::vector<float> a = {1.0f, 1.0f, 1.0f, 1.0f};
                    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f};

                    QuantizationBenchmark bench;
                    float kl = bench.compute_kl_divergence(a, b);

                    EXPECT_NEAR(kl, 0.0f, 1e-6f);
                }

                TEST_F(ComputeErrorTest, KLDivergencePositiveValues) {
                    std::vector<float> a = {0.5f, 0.5f};
                    std::vector<float> b = {0.3f, 0.3f};

                    QuantizationBenchmark bench;
                    float kl = bench.compute_kl_divergence(a, b);

                    EXPECT_GT(kl, 0.0f);
                }

                // Regression: the reported accuracy divided the L2 norm by the element COUNT, so the
                // number shrank as the batch grew -- a uniform 100% error read 0.5 at n=4 and 0.25
                // at n=16. A benchmark exists to compare configurations, and this one could not
                // compare two batch sizes. Dividing by sqrt(n) makes it the RMSE, which is
                // size-invariant.
                TEST_F(ComputeErrorTest, L2AccuracyDoesNotDependOnTheBatchSize) {
                    QuantizationBenchmark bench;

                    const std::vector<float> small_a(4, 1.0f), small_b(4, 2.0f);
                    const std::vector<float> large_a(16, 1.0f), large_b(16, 2.0f);

                    const auto small = bench.benchmark_accuracy_comparison(small_a, small_b, "l2");
                    const auto large = bench.benchmark_accuracy_comparison(large_a, large_b, "l2");

                    EXPECT_NEAR(small.relative_error, large.relative_error, 1e-5f)
                        << "the same 100% error must report the same number at any size; got " << small.relative_error << " vs " << large.relative_error;

                    // And the number has meaning: RMSE of a uniform doubling of 1.0 is 1.0.
                    EXPECT_NEAR(small.relative_error, 1.0f, 1e-5f) << "root-mean-square error of a 100% uniform error is 1.0";
                }

                // Regression: the mean relative error accumulated only over elements above the
                // 1e-6 magnitude guard but divided by the FULL element count, so padding a batch
                // with near-zero elements divided the answer by the padding factor: one
                // measurable element of ten reported a tenth of that element's error.
                TEST_F(ComputeErrorTest, MeanRelativeErrorIgnoresPaddingZeros) {
                    std::vector<float> padded {0.7f, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    std::vector<float> alone {0.7f};

                    const auto padded_result = benchmark_fp8_roundtrip(padded.data(), padded.size());
                    const auto alone_result = benchmark_fp8_roundtrip(alone.data(), alone.size());

                    EXPECT_GT(alone_result.relative_error, 0.0f) << "0.7 is not exactly representable in E4M3, so this must be non-zero";

                    EXPECT_NEAR(padded_result.relative_error, alone_result.relative_error, 1e-5f)
                        << "adding nine near-zero elements must not change the reported error; got " << padded_result.relative_error << " vs " << alone_result.relative_error;
                }

                TEST_F(ComputeErrorTest, KLDivergenceHandlesZeroBins) {
                    std::vector<float> a = {1.0f, 0.0f, 1.0f};
                    std::vector<float> b = {1.0f, 1.0f, 1.0f};

                    QuantizationBenchmark bench;
                    float kl = bench.compute_kl_divergence(a, b);

                    // EXPECT_GE(kl, 0.0f) was tautological for a sum of non-negative terms. For
                    // this input every measured term is log(1/1) = 0, so the exact answer is 0.
                    EXPECT_NEAR(kl, 0.0f, 1e-6f) << "a == b wherever a is non-zero, so the divergence is exactly zero";
                }

                class QuantizationBenchmarkConfigTest : public ::testing::Test {};

                TEST_F(QuantizationBenchmarkConfigTest, ConfigDefaults) {
                    QuantizationBenchmark::Config config;

                    EXPECT_EQ(config.warmup_runs, 10);
                    EXPECT_EQ(config.benchmark_runs, 100);
                    EXPECT_EQ(config.batch_size, 1024);
                    EXPECT_TRUE(config.verify_accuracy);
                    EXPECT_FALSE(config.verbose);
                }

                TEST_F(QuantizationBenchmarkConfigTest, ConfigCustomization) {
                    QuantizationBenchmark::Config config;
                    config.warmup_runs = 5;
                    config.benchmark_runs = 50;
                    config.batch_size = 512;

                    QuantizationBenchmark bench(config);

                    EXPECT_EQ(config.warmup_runs, 5);
                    EXPECT_EQ(config.benchmark_runs, 50);
                    EXPECT_EQ(config.batch_size, 512);
                }

                TEST_F(QuantizationBenchmarkConfigTest, GenerateRandomDataSize) {
                    QuantizationBenchmark bench;
                    auto data = bench.generate_random_data(1000);

                    EXPECT_EQ(data.size(), 1000u);
                }

                TEST_F(QuantizationBenchmarkConfigTest, GenerateRandomDataScale) {
                    QuantizationBenchmark bench;
                    auto data = bench.generate_random_data(100, 2.0f);

                    for (float v : data) {
                        EXPECT_GE(std::abs(v), 0.0f);
                    }
                }

                TEST_F(QuantizationBenchmarkConfigTest, SetStreamRoundtrips) {
                    // The old test called set_stream(0) and asserted nothing (construct-only).
                    // set_stream must actually store the caller's stream.
                    QuantizationBenchmark bench;
                    EXPECT_EQ(bench.get_stream(), (cudaStream_t) nullptr);

                    cudaStream_t s = nullptr;
                    CUDA_CHECK(cudaStreamCreate(&s));
                    bench.set_stream(s);
                    EXPECT_EQ(bench.get_stream(), s);
                    CUDA_CHECK(cudaStreamDestroy(s));
                }

                class QuantizationBenchmarkIntegrationTest : public ::testing::Test {
                protected:
                    void SetUp() override {
                        cudaError_t err = cudaSetDevice(0);
                        if (err != cudaSuccess) {
                            GTEST_SKIP() << "CUDA not available";
                        }
                    }
                };

                TEST_F(QuantizationBenchmarkIntegrationTest, AccuracyComparisonL2Metric) {
                    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f};
                    std::vector<float> quantized = {1.0f, 2.0f, 3.0f, 4.0f};

                    QuantizationBenchmark bench;
                    auto result = bench.benchmark_accuracy_comparison(original, quantized, "l2");

                    EXPECT_EQ(result.name, "Accuracy (l2)");
                    EXPECT_EQ(result.batch_size, 4);
                    EXPECT_LE(result.relative_error, 1e-6f);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, AccuracyComparisonKLMetric) {
                    std::vector<float> original = {1.0f, 1.0f};
                    std::vector<float> quantized = {1.0f, 1.0f};

                    QuantizationBenchmark bench;
                    auto result = bench.benchmark_accuracy_comparison(original, quantized, "kl");

                    EXPECT_EQ(result.name, "Accuracy (kl)");
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, SaveResultsJson) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 1, 64, false, false});
                    auto data = bench.generate_random_data(64);

                    bench.benchmark_int8_quantization(data);
                    auto results = bench.get_results();

                    EXPECT_GT(results.size(), 0);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, PrintSummaryDoesNotCrash) {
                    QuantizationBenchmark bench;
                    testing::internal::CaptureStdout();
                    bench.print_summary();
                    std::string output = testing::internal::GetCapturedStdout();
                    EXPECT_NE(output.find("=== Quantization Benchmark Summary ==="), std::string::npos);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, BenchmarkFP8QuantizationProducesResult) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 1, 256, false, false});
                    auto data = bench.generate_random_data(256);

                    auto result = bench.benchmark_fp8_quantization(data, Precision::FP8_E4M3);

                    EXPECT_EQ(result.name, "FP8 Quantization");
                    EXPECT_EQ(result.batch_size, 256);
                    EXPECT_GT(result.latency_us, 0.0f);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, BenchmarkFP8QuantizationMeasuresThroughput) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 10, 1024, false, false});
                    auto data = bench.generate_random_data(1024);

                    auto result = bench.benchmark_fp8_quantization(data);

                    EXPECT_GE(result.throughput_gbps, 0.0f);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, BenchmarkFP8QuantizationTracksRelativeError) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 5, 512, true, false});
                    auto data = bench.generate_random_data(512);

                    auto result = bench.benchmark_fp8_quantization(data);

                    EXPECT_GE(result.relative_error, 0.0f);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, BenchmarkINT8QuantizationProducesResult) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 1, 256, false, false});
                    auto data = bench.generate_random_data(256);

                    auto result = bench.benchmark_int8_quantization(data);

                    EXPECT_EQ(result.name, "INT8 Quantization");
                    EXPECT_EQ(result.batch_size, 256);
                    EXPECT_GT(result.latency_us, 0.0f);
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, BenchmarkFP8GEMMProducesResult) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 1, 256, false, false});

                    auto result = bench.benchmark_fp8_gemm(16, 32, 16);

                    EXPECT_NE(result.name.find("FP8 GEMM"), std::string::npos);
                    EXPECT_GT(result.batch_size, 0);
                }

                // The element counts were int products widened afterwards, so they wrapped before
                // the widening. With m = k = 65536, `m * k` is exactly 0: cudaMalloc was asked for
                // zero bytes while the kernel ran over the full extent, and the failure surfaced as
                // "CUDA error: an illegal memory access was encountered" from a later, unrelated
                // line. Measured before the fix: benchmark_fp8_gemm(65536, 65536, 8) terminated with
                // that CUDA error; after it, the call is refused up front with the element total in
                // the message.
                TEST_F(QuantizationBenchmarkIntegrationTest, Fp8GemmRejectsShapesThatOverflowTheElementCount) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 0, 1, false, false});

                    EXPECT_THROW(bench.benchmark_fp8_gemm(65536, 65536, 8), std::invalid_argument) << "m * k wraps to 0 as an int; the shape must be refused, not allocated";
                    EXPECT_THROW(bench.benchmark_fp8_gemm(0, 8, 8), std::invalid_argument) << "a zero dimension must be refused";
                    EXPECT_THROW(bench.benchmark_fp8_gemm(-4, 8, 8), std::invalid_argument) << "a negative dimension must be refused";
                }

                TEST_F(QuantizationBenchmarkIntegrationTest, BenchmarkCalibrationProducesResult) {
                    QuantizationBenchmark bench(QuantizationBenchmark::Config {1, 1, 256, false, false});
                    auto data = bench.generate_random_data(256);

                    MinMaxCalibrator calibrator;
                    auto result = bench.benchmark_calibration(data, calibrator);

                    EXPECT_EQ(result.name, "Calibration (MinMax)");
                    EXPECT_GT(result.latency_us, 0.0f);
                }

                class StandaloneBenchmarkTest : public ::testing::Test {
                protected:
                    void SetUp() override {
                        cudaError_t err = cudaSetDevice(0);
                        if (err != cudaSuccess) {
                            GTEST_SKIP() << "CUDA not available";
                        }
                    }
                };

                TEST_F(StandaloneBenchmarkTest, FP8RoundtripComputesError) {
                    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

                    auto result = benchmark_fp8_roundtrip(data.data(), data.size(), Precision::FP8_E4M3);

                    EXPECT_EQ(result.name, "FP8 Roundtrip");
                    EXPECT_EQ(result.batch_size, 4);
                    EXPECT_GE(result.relative_error, 0.0f);
                }

                TEST_F(StandaloneBenchmarkTest, FP8GEMMThroughputMeasuresThroughput) {
                    auto result = benchmark_fp8_gemm_throughput(32, 64, 32, 10);

                    EXPECT_NE(result.name.find("FP8 GEMM"), std::string::npos);
                    EXPECT_GT(result.batch_size, 0);
                    EXPECT_GE(result.throughput_gbps, 0.0f);
                }

                TEST_F(StandaloneBenchmarkTest, GenerateReportCreatesMarkdown) {
                    std::vector<BenchmarkResult> results;
                    BenchmarkResult r1;
                    r1.name = "Test 1";
                    r1.throughput_gbps = 100.0f;
                    r1.latency_us = 50.0f;
                    r1.relative_error = 0.01f;
                    results.push_back(r1);

                    generate_report(results, "/tmp/benchmark_report.md");

                    std::ifstream file("/tmp/benchmark_report.md");
                    ASSERT_TRUE(file.is_open());

                    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

                    EXPECT_NE(content.find("Quantization Benchmark Report"), std::string::npos);
                    EXPECT_NE(content.find("Test 1"), std::string::npos);
                }

            }  // namespace test
        }  // namespace benchmark
    }  // namespace quantize
}  // namespace nova
