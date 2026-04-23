#pragma once

/**
 * @file benchmark.h
 * @brief Benchmark framework with warm-up, statistics, and regression detection
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::benchmark {

struct BenchmarkResult {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double throughput_gbps = 0.0;
    size_t samples = 0;

    [[nodiscard]] std::string format() const {
        return std::to_string(mean_ms) + " ± " + std::to_string(stddev_ms) + " ms";
    }
};

struct BenchmarkOptions {
    int warmup_iterations = 3;
    int measurement_iterations = 10;
    bool compute_throughput = false;
    size_t data_size_bytes = 0;
    double tolerance_percent = 10.0;
};

inline double compute_throughput_gbps(size_t bytes, double elapsed_ms) {
    if (elapsed_ms <= 0.0) {
        return 0.0;
    }
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) /
           (elapsed_ms / 1000.0);
}

class Benchmark {
public:
    explicit Benchmark(std::string name, BenchmarkOptions options = {})
        : name_(std::move(name)),
          options_(options) {}

    ~Benchmark() = default;

    template <typename Kernel, typename... Args>
    BenchmarkResult run(Kernel&& kernel, Args&&... /* args */) {
        warmup(kernel);

        std::vector<double> measurements;
        measurements.reserve(static_cast<size_t>(options_.measurement_iterations));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < options_.measurement_iterations; ++i) {
            CUDA_CHECK(cudaEventRecord(start, nullptr));

            kernel();

            CUDA_CHECK(cudaEventRecord(stop, nullptr));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            measurements.push_back(static_cast<double>(elapsed_ms));
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return compute_statistics(std::move(measurements));
    }

    void set_baseline(const BenchmarkResult& baseline) {
        baseline_ = baseline;
    }

    [[nodiscard]] bool compare_to_baseline(const BenchmarkResult& current) const {
        if (!baseline_.has_value()) {
            return true;
        }

        double baseline_mean = baseline_.value().mean_ms;
        double current_mean = current.mean_ms;

        double delta_percent = std::abs((current_mean - baseline_mean) / baseline_mean) * 100.0;
        return delta_percent <= options_.tolerance_percent;
    }

    [[nodiscard]] std::string format_regression_report(const BenchmarkResult& current) const {
        if (!baseline_.has_value()) {
            return "No baseline set";
        }

        double baseline_mean = baseline_.value().mean_ms;
        double current_mean = current.mean_ms;
        double delta_percent = ((current_mean - baseline_mean) / baseline_mean) * 100.0;

        std::string prefix = delta_percent >= 0 ? "+" : "";
        return "Current: " + current.format() + "\n" +
               "Baseline: " + baseline_.value().format() + "\n" +
               "Delta: " + prefix + std::to_string(delta_percent) + "%";
    }

private:
    std::string name_;
    BenchmarkOptions options_;
    std::optional<BenchmarkResult> baseline_;
    std::vector<double> measurements_;

    template <typename Kernel>
    void warmup(Kernel&& kernel) {
        for (int i = 0; i < options_.warmup_iterations; ++i) {
            kernel();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    BenchmarkResult compute_statistics(std::vector<double> measurements) {
        BenchmarkResult result;
        result.samples = measurements.size();

        if (measurements.empty()) {
            return result;
        }

        result.min_ms = *std::min_element(measurements.begin(), measurements.end());
        result.max_ms = *std::max_element(measurements.begin(), measurements.end());

        double sum = 0.0;
        for (double m : measurements) {
            sum += m;
        }
        result.mean_ms = sum / static_cast<double>(measurements.size());

        double sq_sum = 0.0;
        for (double m : measurements) {
            sq_sum += (m - result.mean_ms) * (m - result.mean_ms);
        }
        result.stddev_ms = std::sqrt(sq_sum / static_cast<double>(measurements.size()));

        if (options_.compute_throughput && options_.data_size_bytes > 0) {
            result.throughput_gbps = compute_throughput_gbps(options_.data_size_bytes, result.mean_ms);
        }

        return result;
    }
};

}  // namespace cuda::benchmark