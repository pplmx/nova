/**
 * @file benchmark_dequantization_test.cpp
 * @brief QuantizationBenchmark::benchmark_dequantization must exist
 *
 * The header declared six BenchmarkResult measurements and the .cpp defined five:
 * benchmark_dequantization was missing. Taking its address and linking produced
 *
 *   undefined reference to
 *   `nova::quantize::benchmark::QuantizationBenchmark::benchmark_dequantization(
 *        std::vector<signed char> const&, nova::quantize::Precision)'
 *
 * so the class advertised a measurement that could not be called at all. It is now
 * implemented alongside its siblings, using dequantize_int8_to_f32 with the same warmup /
 * timed-run structure and the same bytes-per-microsecond unit convention.
 *
 * Calling it from this translation unit is the regression check: the test binary would not
 * link if the definition disappeared again.
 */

#include <cuda/quantize/benchmark.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace benchmark_dequantization_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        nova::quantize::benchmark::QuantizationBenchmark::Config small_config() {
            nova::quantize::benchmark::QuantizationBenchmark::Config config;
            config.warmup_runs = 1;
            config.benchmark_runs = 2;
            config.verify_accuracy = false;
            return config;
        }

    }  // namespace

    TEST(BenchmarkDequantizationTest, TheMeasurementRunsAndReportsItsName) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        nova::quantize::benchmark::QuantizationBenchmark bench(small_config());
        const std::vector<int8_t> data(256, 4);

        const auto result = bench.benchmark_dequantization(data);

        EXPECT_FALSE(result.name.empty());
        EXPECT_NE(result.name.find("Dequantization"), std::string::npos);
        EXPECT_EQ(result.batch_size, static_cast<int>(data.size()));
        EXPECT_EQ(result.num_samples, 2);
        EXPECT_GE(result.latency_us, 0.0F) << "a duration cannot be negative";
        EXPECT_GE(result.throughput_gbps, 0.0F);
    }

    TEST(BenchmarkDequantizationTest, ItAcceptsAPrecisionArgument) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        nova::quantize::benchmark::QuantizationBenchmark bench(small_config());
        const std::vector<int8_t> data(128, -3);

        const auto fp32 = bench.benchmark_dequantization(data, nova::quantize::Precision::FP32);
        const auto mixed = bench.benchmark_dequantization(data, nova::quantize::Precision::FP16);

        EXPECT_NE(fp32.name, mixed.name) << "the requested precision should be visible in the result";
    }

    TEST(BenchmarkDequantizationTest, EmptyInputDoesNotCrash) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        nova::quantize::benchmark::QuantizationBenchmark bench(small_config());
        const std::vector<int8_t> empty;

        // dequantize_int8_to_f32 guards n == 0, and cudaMalloc(0) is legal, so this should
        // return a result with no work in it rather than faulting.
        EXPECT_NO_THROW({
            const auto result = bench.benchmark_dequantization(empty);
            EXPECT_EQ(result.batch_size, 0);
        });
    }

}  // namespace benchmark_dequantization_test
