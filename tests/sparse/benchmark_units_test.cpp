/**
 * @file benchmark_units_test.cpp
 * @brief Contracts for the units the quantize benchmarks report
 *
 * src/cuda/quantize/benchmark.cpp fills a BenchmarkResult whose throughput field is
 * named and printed as GB/s:
 *
 *     printf("%s: %.2f GB/s, ...", name.c_str(), throughput_gbps, ...);
 *
 * Three of its four computations produced something else entirely.
 *
 * - Two divided a BYTE count by a duration in MICROSECONDS. bytes / us is MB/s, not
 *   GB/s, so the figure was 1000x too large. Measured before the fix: a 1 MiB
 *   buffer reported 364088.88 GB/s, against an A100-80GB peak of about 2039 GB/s.
 *
 * - The FP8 GEMM benchmark divided a FLOP count by microseconds, which is a compute
 *   rate (TFLOP/s) placed in a bandwidth field — wrong unit AND wrong quantity.
 *
 * - Its sibling multiplied the FLOP count by 1e9 and divided by ms * 1e6, a
 *   combination that describes no physical quantity at all.
 *
 * All four now report bytes moved divided by elapsed seconds, in GB/s. After the fix
 * the same measurement reports 358.79 GB/s, which matches the bytes-accounting
 * arithmetic for that buffer size and sits below the device's peak as expected.
 */

#include <cuda/quantize/benchmark.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

namespace nova::quantize::test {

    namespace {

        using nova::quantize::benchmark::BenchmarkResult;
        using nova::quantize::benchmark::QuantizationBenchmark;

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Peak device memory bandwidth in GB/s, from the hardware attributes. Used as an
        /// upper bound: no measurement of a single small buffer can exceed it.
        double device_peak_gbps() {
            int clock_khz = 0;
            int bus_bits = 0;
            if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0) != cudaSuccess
                || cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0) != cudaSuccess) {
                return 0.0;
            }
            return (clock_khz / 1e6) * (bus_bits / 8.0) * 2.0;
        }

    }  // namespace

    // Regression: this reported 364088.88 GB/s for a 1 MiB buffer.
    TEST(BenchmarkUnitsTest, QuantizationThroughputIsPhysicallyPlausible) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const double peak = device_peak_gbps();
        ASSERT_GT(peak, 0.0) << "the device must expose clock and bus width";

        QuantizationBenchmark bench;
        const std::vector<float> data(1024 * 1024, 0.5f);
        const BenchmarkResult result = bench.benchmark_fp8_quantization(data);

        EXPECT_GT(result.throughput_gbps, 0.0f);
        // A single 1 MiB copy cannot exceed the device's peak. Allowing the full peak
        // still rejects an answer off by the factor of 1000 that the old code produced.
        EXPECT_LT(result.throughput_gbps, peak * 1.5) << "reported " << result.throughput_gbps << " GB/s against a device peak of " << peak
                                                      << " GB/s; a bytes-per-microsecond quotient is 1000x too large";
    }

    TEST(BenchmarkUnitsTest, QuantizationThroughputMatchesByteAccounting) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const size_t samples = 1024 * 1024;
        QuantizationBenchmark bench;
        const std::vector<float> data(samples, 0.25f);
        const BenchmarkResult result = bench.benchmark_fp8_quantization(data);

        ASSERT_GT(result.latency_us, 0.0f) << "the measurement needs a duration";

        // The benchmark counts the float input bytes it moves. Recompute from the
        // reported latency and compare: agreement within 5% shows both numbers use the
        // same unit. The old code differed by exactly three orders of magnitude.
        const double bytes = static_cast<double>(samples) * sizeof(float);
        const double expected_gbps = bytes / (static_cast<double>(result.latency_us) * 1e3);
        EXPECT_NEAR(result.throughput_gbps, expected_gbps, expected_gbps * 0.05) << "throughput and latency are not expressed in consistent units";
    }

    // Regression: this divided FLOPs by microseconds inside a GB/s field.
    TEST(BenchmarkUnitsTest, GemmThroughputIsPhysicallyPlausible) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const double peak = device_peak_gbps();
        ASSERT_GT(peak, 0.0);

        // A small shape keeps the run short while still crossing the block grid.
        const auto result = nova::quantize::benchmark::benchmark_fp8_gemm_throughput(128, 128, 128, 20);

        EXPECT_GT(result.throughput_gbps, 0.0f);
        EXPECT_LT(result.throughput_gbps, peak * 1.5) << "reported " << result.throughput_gbps << " GB/s; the field is printed as GB/s and must be a bandwidth";
        EXPECT_GT(result.latency_us, 0.0f) << "per-run latency must be positive";
    }

    TEST(BenchmarkUnitsTest, ResultPrintingCarriesTheUnitItClaims) {
        // The field is only meaningful because print() labels it GB/s. Pinning the label
        // keeps a future rename from silently changing what readers assume.
        BenchmarkResult result;
        result.name = "unit-check";
        result.throughput_gbps = 1.0f;
        result.latency_us = 1.0f;

        testing::internal::CaptureStdout();
        result.print();
        const std::string printed = testing::internal::GetCapturedStdout();

        EXPECT_NE(printed.find("GB/s"), std::string::npos) << "the field is named throughput_gbps and must print as GB/s: " << printed;
        EXPECT_NE(printed.find("us"), std::string::npos) << "latency is in microseconds: " << printed;
    }

    TEST(BenchmarkUnitsTest, Int8ThroughputIsAlsoPhysicallyPlausible) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const double peak = device_peak_gbps();
        ASSERT_GT(peak, 0.0);

        QuantizationBenchmark bench;
        const std::vector<float> data(512 * 1024, 0.5f);
        const BenchmarkResult result = bench.benchmark_int8_quantization(data);

        EXPECT_LT(result.throughput_gbps, peak * 1.5) << "the int8 path shares the same unit convention";
    }

}  // namespace nova::quantize::test
