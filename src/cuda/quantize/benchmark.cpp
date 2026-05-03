#include <cuda/quantize/benchmark.hpp>
#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda/quantize/calibrator.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

namespace nova {
namespace quantize {
namespace benchmark {

BenchmarkResult QuantizationBenchmark::benchmark_fp8_quantization(
    const std::vector<float>& data,
    Precision precision) {

    BenchmarkResult result;
    result.name = "FP8 Quantization";
    result.batch_size = static_cast<int>(data.size());
    result.num_samples = config_.benchmark_runs;

    float* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(float));

    std::vector<FP8E4M3> quantized(data.size());
    float* d_quantized;
    cudaMalloc(&d_quantized, data.size() * sizeof(FP8E4M3));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < config_.warmup_runs; ++i) {
        cuda::quantize_f32_to_fp8e4m3(d_data, quantized.data(), data.size(), stream_);
    }
    cudaStreamSynchronize(stream_);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config_.benchmark_runs; ++i) {
        cuda::quantize_f32_to_fp8e4m3(d_data, quantized.data(), data.size(), stream_);
    }
    cudaStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float total_bytes = static_cast<float>(data.size()) * sizeof(float) * config_.benchmark_runs;
    result.throughput_gbps = total_bytes / duration.count();
    result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

    std::vector<float> recovered(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        recovered[i] = static_cast<float>(quantized[i]);
    }

    result.relative_error = compute_l2_error(data, recovered) / data.size();

    cudaFree(d_data);
    cudaFree(d_quantized);

    return result;
}

BenchmarkResult QuantizationBenchmark::benchmark_int8_quantization(
    const std::vector<float>& data) {

    BenchmarkResult result;
    result.name = "INT8 Quantization";
    result.batch_size = static_cast<int>(data.size());
    result.num_samples = config_.benchmark_runs;

    float* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(float));
    int8_t* d_quantized;
    cudaMalloc(&d_quantized, data.size() * sizeof(int8_t));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    cuda::QuantizationParams params(0.1f);

    for (int i = 0; i < config_.warmup_runs; ++i) {
        cuda::quantize_f32_to_int8(d_data, d_quantized, data.size(), params, stream_);
    }
    cudaStreamSynchronize(stream_);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config_.benchmark_runs; ++i) {
        cuda::quantize_f32_to_int8(d_data, d_quantized, data.size(), params, stream_);
    }
    cudaStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float total_bytes = static_cast<float>(data.size()) * sizeof(float) * config_.benchmark_runs;
    result.throughput_gbps = total_bytes / duration.count();
    result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

    cudaFree(d_data);
    cudaFree(d_quantized);

    return result;
}

BenchmarkResult QuantizationBenchmark::benchmark_fp8_gemm(
    int m, int k, int n) {

    BenchmarkResult result;
    result.name = "FP8 GEMM " + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
    result.batch_size = m * k + k * n + m * n;
    result.num_samples = config_.benchmark_runs;

    std::vector<FP8E4M3> a(m * k);
    std::vector<FP8E4M3> b(k * n);
    std::vector<float> c(m * n);

    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
    cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    FP8GEMM::Config gemm_config;

    for (int i = 0; i < config_.warmup_runs; ++i) {
        FP8GEMM::forward(
            reinterpret_cast<const FP8E4M3*>(d_a),
            reinterpret_cast<const FP8E4M3*>(d_b),
            d_c, m, k, n, gemm_config, stream_);
    }
    cudaStreamSynchronize(stream_);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config_.benchmark_runs; ++i) {
        FP8GEMM::forward(
            reinterpret_cast<const FP8E4M3*>(d_a),
            reinterpret_cast<const FP8E4M3*>(d_b),
            d_c, m, k, n, gemm_config, stream_);
    }
    cudaStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    size_t flops = 2ULL * m * k * n;
    float total_flops = static_cast<float>(flops) * config_.benchmark_runs;
    result.throughput_gbps = total_flops / duration.count();
    result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

BenchmarkResult QuantizationBenchmark::benchmark_calibration(
    const std::vector<float>& data,
    Calibrator& calibrator) {

    BenchmarkResult result;
    result.name = "Calibration (" + calibrator.name() + ")";
    result.batch_size = static_cast<int>(data.size());
    result.num_samples = config_.benchmark_runs;

    float* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(float));
    cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < config_.warmup_runs; ++i) {
        calibrator.calibrate(d_data, data.size(), stream_);
    }
    cudaStreamSynchronize(stream_);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config_.benchmark_runs; ++i) {
        calibrator.calibrate(d_data, data.size(), stream_);
    }
    cudaStreamSynchronize(stream_);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

    cudaFree(d_data);

    return result;
}

BenchmarkResult QuantizationBenchmark::benchmark_accuracy_comparison(
    const std::vector<float>& original,
    const std::vector<float>& quantized,
    const std::string& metric) {

    BenchmarkResult result;
    result.name = "Accuracy (" + metric + ")";
    result.batch_size = static_cast<int>(original.size());

    if (metric == "l2") {
        result.relative_error = compute_l2_error(original, quantized) / original.size();
    } else if (metric == "kl") {
        result.relative_error = compute_kl_divergence(original, quantized);
    }

    return result;
}

std::vector<float> QuantizationBenchmark::generate_random_data(size_t n, float scale) {
    std::vector<float> data(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

float QuantizationBenchmark::compute_l2_error(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    float sum = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float QuantizationBenchmark::compute_kl_divergence(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    float kl = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] > 0.0f && b[i] > 0.0f) {
            kl += a[i] * std::log(a[i] / b[i]);
        }
    }
    return kl;
}

void QuantizationBenchmark::run_all_benchmarks() {
    results_.clear();

    auto data = generate_random_data(config_.batch_size, 1.0f);

    results_.push_back(benchmark_fp8_quantization(data));
    results_.push_back(benchmark_int8_quantization(data));

    MinMaxCalibrator minmax_cal;
    results_.push_back(benchmark_calibration(data, minmax_cal));

    results_.push_back(benchmark_fp8_gemm(128, 256, 128));
    results_.push_back(benchmark_fp8_gemm(256, 512, 256));

    if (config_.verify_accuracy) {
        auto quantized = generate_random_data(config_.batch_size, 0.9f);
        results_.push_back(benchmark_accuracy_comparison(data, quantized, "l2"));
    }
}

void QuantizationBenchmark::save_results_json(const std::string& path) const {
    std::ofstream file(path);
    if (file.is_open()) {
        file << "[\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            results_[i].save_json(file);
            if (i < results_.size() - 1) {
                file << ",\n";
            }
        }
        file << "]\n";
    }
}

void QuantizationBenchmark::print_summary() const {
    printf("\n=== Quantization Benchmark Summary ===\n");
    printf("%-40s %12s %12s\n", "Benchmark", "Throughput", "Latency");
    printf("%-40s %12s %12s\n", "", "(GB/s)", "(us)");
    printf("%-40s %12s %12s\n", "----------------------------------------",
           "------------", "------------");

    for (const auto& result : results_) {
        printf("%-40s %11.2f %11.2f\n",
               result.name.c_str(),
               result.throughput_gbps,
               result.latency_us);
    }
}

BenchmarkResult benchmark_fp8_roundtrip(
    const float* data, size_t n,
    Precision precision) {

    BenchmarkResult result;
    result.name = "FP8 Roundtrip";
    result.batch_size = static_cast<int>(n);

    std::vector<FP8E4M3> quantized(n);
    for (size_t i = 0; i < n; ++i) {
        quantized[i] = FP8E4M3(data[i]);
    }

    std::vector<float> recovered(n);
    for (size_t i = 0; i < n; ++i) {
        recovered[i] = static_cast<float>(quantized[i]);
    }

    float sum_rel_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(data[i]) > 1e-6f) {
            sum_rel_err += std::abs(data[i] - recovered[i]) / std::abs(data[i]);
        }
    }
    result.relative_error = sum_rel_err / n;

    return result;
}

BenchmarkResult benchmark_fp8_gemm_throughput(
    int m, int k, int n,
    int num_runs) {

    BenchmarkResult result;
    result.name = "FP8 GEMM " + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
    result.batch_size = m * n;
    result.num_samples = num_runs;

    std::vector<FP8E4M3> a(m * k);
    std::vector<FP8E4M3> b(k * n);
    std::vector<float> c(m * n, 0.0f);

    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
    cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    FP8GEMM::Config config;

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; ++i) {
        FP8GEMM::forward(
            reinterpret_cast<const FP8E4M3*>(d_a),
            reinterpret_cast<const FP8E4M3*>(d_b),
            d_c, m, k, n, config, 0);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    size_t flops = 2ULL * m * k * n;
    float total_flops = static_cast<float>(flops) * num_runs * 1e9f;
    result.throughput_gbps = total_flops / (ms * 1e6f);
    result.latency_us = ms * 1000.0f / num_runs;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

void generate_report(
    const std::vector<BenchmarkResult>& results,
    const std::string& output_path) {

    std::ofstream file(output_path);
    if (file.is_open()) {
        file << "# Quantization Benchmark Report\n\n";

        file << "| Benchmark | Throughput (GB/s) | Latency (us) | Relative Error |\n";
        file << "|-----------|-------------------|--------------|----------------|\n";

        for (const auto& result : results) {
            file << "| " << result.name
                 << " | " << result.throughput_gbps
                 << " | " << result.latency_us
                 << " | " << result.relative_error
                 << " |\n";
        }
    }
}

} // namespace benchmark
} // namespace quantize
} // namespace nova
