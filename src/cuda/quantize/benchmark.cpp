#include <cuda/quantize/benchmark.hpp>
#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include <cuda/quantize/fp8_kernels.hpp>
#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda/quantize/calibrator.hpp>
#include "cuda/device/error.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

namespace nova {
    namespace quantize {
        namespace benchmark {

            BenchmarkResult QuantizationBenchmark::benchmark_fp8_quantization(const std::vector<float>& data, Precision precision) {
                BenchmarkResult result;
                result.name = "FP8 Quantization";
                result.batch_size = static_cast<int>(data.size());
                result.num_samples = config_.benchmark_runs;

                float* d_data;
                CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(float)));

                std::vector<FP8E4M3> quantized(data.size());
                float* d_quantized;
                CUDA_CHECK(cudaMalloc(&d_quantized, data.size() * sizeof(FP8E4M3)));

                CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

                for (int i = 0; i < config_.warmup_runs; ++i) {
                    nova::quantize::cuda::quantize_f32_to_fp8e4m3(d_data, reinterpret_cast<FP8E4M3*>(d_quantized), data.size(), stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < config_.benchmark_runs; ++i) {
                    nova::quantize::cuda::quantize_f32_to_fp8e4m3(d_data, reinterpret_cast<FP8E4M3*>(d_quantized), data.size(), stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                // throughput_gbps is printed as GB/s, so the byte count must be scaled by the
                // duration in seconds and then divided by 1e9. Dividing bytes by microseconds
                // yields MB/s, which reported values ~1000x too large: a 1 MiB buffer measured
                // as 364088.88 GB/s against a device whose peak is about 2039 GB/s.
                const float total_bytes = static_cast<float>(data.size()) * sizeof(float) * config_.benchmark_runs;
                result.throughput_gbps = total_bytes / (static_cast<float>(duration.count()) * 1e3f);
                result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

                CUDA_CHECK(cudaMemcpy(quantized.data(), d_quantized, data.size() * sizeof(FP8E4M3), cudaMemcpyDeviceToHost));

                std::vector<float> recovered(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    recovered[i] = static_cast<float>(quantized[i]);
                }

                result.relative_error = compute_l2_error(data, recovered) / data.size();

                CUDA_CHECK(cudaFree(d_data));
                CUDA_CHECK(cudaFree(d_quantized));

                results_.push_back(result);
                return result;
            }

            BenchmarkResult QuantizationBenchmark::benchmark_int8_quantization(const std::vector<float>& data) {
                BenchmarkResult result;
                result.name = "INT8 Quantization";
                result.batch_size = static_cast<int>(data.size());
                result.num_samples = config_.benchmark_runs;

                float* d_data;
                CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(float)));
                int8_t* d_quantized;
                CUDA_CHECK(cudaMalloc(&d_quantized, data.size() * sizeof(int8_t)));

                CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

                cuda::QuantizationParams params(0.1f);

                for (int i = 0; i < config_.warmup_runs; ++i) {
                    cuda::quantize_f32_to_int8(d_data, d_quantized, data.size(), params, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < config_.benchmark_runs; ++i) {
                    cuda::quantize_f32_to_int8(d_data, d_quantized, data.size(), params, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                // Same unit correction as the quantization benchmark above: bytes per
                // microsecond is MB/s, not GB/s.
                const float total_bytes = static_cast<float>(data.size()) * sizeof(float) * config_.benchmark_runs;
                result.throughput_gbps = total_bytes / (static_cast<float>(duration.count()) * 1e3f);
                result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

                CUDA_CHECK(cudaFree(d_data));
                CUDA_CHECK(cudaFree(d_quantized));

                results_.push_back(result);
                return result;
            }

            // Declared in the header and never defined. Any caller got a link error naming this symbol
            // rather than a benchmark: the class advertises six measurements and shipped five. Found by
            // taking the address of every declared member and linking.
            BenchmarkResult QuantizationBenchmark::benchmark_dequantization(const std::vector<int8_t>& data, Precision precision) {
                BenchmarkResult result;
                result.name = "Dequantization";
                result.batch_size = static_cast<int>(data.size());
                result.num_samples = config_.benchmark_runs;

                int8_t* d_quantized;
                CUDA_CHECK(cudaMalloc(&d_quantized, data.size() * sizeof(int8_t)));
                float* d_output;
                CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(float)));

                CUDA_CHECK(cudaMemcpy(d_quantized, data.data(), data.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

                cuda::QuantizationParams params(0.1f);

                for (int i = 0; i < config_.warmup_runs; ++i) {
                    cuda::dequantize_int8_to_f32(d_quantized, d_output, data.size(), params, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < config_.benchmark_runs; ++i) {
                    cuda::dequantize_int8_to_f32(d_quantized, d_output, data.size(), params, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                // Reads one byte per element and writes four, so the moved bytes are five per element —
                // the same unit convention the neighbouring benchmarks use (bytes per microsecond).
                const float total_bytes = static_cast<float>(data.size()) * (sizeof(int8_t) + sizeof(float)) * config_.benchmark_runs;
                result.throughput_gbps = total_bytes / (static_cast<float>(duration.count()) * 1e3f);
                result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

                // The output precision names what the caller asked the values to be interpreted as;
                // recording it keeps the result self-describing rather than assuming FP32 downstream.
                result.name += (precision == Precision::FP32) ? " (FP32)" : " (mixed)";

                CUDA_CHECK(cudaFree(d_quantized));
                CUDA_CHECK(cudaFree(d_output));

                results_.push_back(result);
                return result;
            }

            BenchmarkResult QuantizationBenchmark::benchmark_fp8_gemm(int m, int k, int n) {
                BenchmarkResult result;
                result.name = "FP8 GEMM " + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
                result.batch_size = m * k + k * n + m * n;
                result.num_samples = config_.benchmark_runs;

                std::vector<FP8E4M3> a(m * k);
                std::vector<FP8E4M3> b(k * n);
                std::vector<float> c(m * n);

                float *d_a, *d_b, *d_c;
                CUDA_CHECK(cudaMalloc(&d_a, m * k * sizeof(FP8E4M3)));
                CUDA_CHECK(cudaMalloc(&d_b, k * n * sizeof(FP8E4M3)));
                CUDA_CHECK(cudaMalloc(&d_c, m * n * sizeof(float)));

                CUDA_CHECK(cudaMemcpy(d_a, a.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, b.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice));

                FP8GEMM::Config gemm_config;

                for (int i = 0; i < config_.warmup_runs; ++i) {
                    FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, gemm_config, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < config_.benchmark_runs; ++i) {
                    FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, gemm_config, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                // This field is printed as GB/s, but the value here is a compute rate. Report
                // the memory traffic the product actually moves (each operand is one byte in
                // FP8) so the number means what its name and its printf say.
                const size_t operand_bytes = (static_cast<size_t>(m) * k + static_cast<size_t>(k) * n + static_cast<size_t>(m) * n) * sizeof(FP8E4M3);
                const float total_bytes = static_cast<float>(operand_bytes) * config_.benchmark_runs;
                result.throughput_gbps = total_bytes / (static_cast<float>(duration.count()) * 1e3f);
                result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

                CUDA_CHECK(cudaFree(d_a));
                CUDA_CHECK(cudaFree(d_b));
                CUDA_CHECK(cudaFree(d_c));

                return result;
            }

            BenchmarkResult QuantizationBenchmark::benchmark_calibration(const std::vector<float>& data, Calibrator& calibrator) {
                BenchmarkResult result;
                result.name = "Calibration (" + calibrator.name() + ")";
                result.batch_size = static_cast<int>(data.size());
                result.num_samples = config_.benchmark_runs;

                float* d_data;
                CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

                for (int i = 0; i < config_.warmup_runs; ++i) {
                    calibrator.calibrate(d_data, data.size(), stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < config_.benchmark_runs; ++i) {
                    calibrator.calibrate(d_data, data.size(), stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                result.latency_us = static_cast<float>(duration.count()) / config_.benchmark_runs;

                CUDA_CHECK(cudaFree(d_data));

                return result;
            }

            BenchmarkResult QuantizationBenchmark::benchmark_accuracy_comparison(const std::vector<float>& original,
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

            float QuantizationBenchmark::compute_l2_error(const std::vector<float>& a, const std::vector<float>& b) {
                float sum = 0.0f;
                for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                    float diff = a[i] - b[i];
                    sum += diff * diff;
                }
                return std::sqrt(sum);
            }

            float QuantizationBenchmark::compute_kl_divergence(const std::vector<float>& a, const std::vector<float>& b) {
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
                printf("%-40s %12s %12s\n", "----------------------------------------", "------------", "------------");

                for (const auto& result : results_) {
                    printf("%-40s %11.2f %11.2f\n", result.name.c_str(), result.throughput_gbps, result.latency_us);
                }
            }

            BenchmarkResult benchmark_fp8_roundtrip(const float* data, size_t n, Precision precision) {
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

            BenchmarkResult benchmark_fp8_gemm_throughput(int m, int k, int n, int num_runs) {
                BenchmarkResult result;
                result.name = "FP8 GEMM " + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
                result.batch_size = m * n;
                result.num_samples = num_runs;

                std::vector<FP8E4M3> a(m * k);
                std::vector<FP8E4M3> b(k * n);
                std::vector<float> c(m * n, 0.0f);

                float *d_a, *d_b, *d_c;
                CUDA_CHECK(cudaMalloc(&d_a, m * k * sizeof(FP8E4M3)));
                CUDA_CHECK(cudaMalloc(&d_b, k * n * sizeof(FP8E4M3)));
                CUDA_CHECK(cudaMalloc(&d_c, m * n * sizeof(float)));

                CUDA_CHECK(cudaMemcpy(d_a, a.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b, b.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice));

                cudaEvent_t start, end;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&end));

                FP8GEMM::Config config;

                CUDA_CHECK(cudaEventRecord(start));
                for (int i = 0; i < num_runs; ++i) {
                    FP8GEMM::forward(reinterpret_cast<const FP8E4M3*>(d_a), reinterpret_cast<const FP8E4M3*>(d_b), d_c, m, k, n, config, 0);
                }
                CUDA_CHECK(cudaEventRecord(end));
                CUDA_CHECK(cudaEventSynchronize(end));

                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

                // cudaEventElapsedTime reports milliseconds, so bytes / ms * 1e3 = bytes/s and
                // bytes/s / 1e9 = GB/s. The previous expression multiplied the FLOP count by
                // 1e9 and divided by ms * 1e6, which both mixed FLOPs into a GB/s field and
                // scaled it to a figure no hardware could produce.
                const size_t operand_bytes = (static_cast<size_t>(m) * k + static_cast<size_t>(k) * n + static_cast<size_t>(m) * n) * sizeof(FP8E4M3);
                const float total_bytes = static_cast<float>(operand_bytes) * num_runs;
                result.throughput_gbps = total_bytes / ms * 1e3f / 1e9f;
                result.latency_us = ms * 1000.0f / num_runs;

                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(end));
                CUDA_CHECK(cudaFree(d_a));
                CUDA_CHECK(cudaFree(d_b));
                CUDA_CHECK(cudaFree(d_c));

                return result;
            }

            std::vector<BenchmarkResult> run_precision_comparison(const float* a, const float* b, const float* c_ref, int m, int k, int n) {
                // Declared in benchmark.hpp but never defined, so every caller failed to
                // link with "undefined reference to run_precision_comparison". It benchmarks
                // the same GEMM shape at each quantisation precision and reports the error
                // against a reference result, which is what the name promises.
                std::vector<BenchmarkResult> results;

                if (a == nullptr || b == nullptr || c_ref == nullptr || m <= 0 || k <= 0 || n <= 0) {
                    return results;
                }

                struct PrecisionCase {
                    const char* label;
                    float (*quantize)(float);  // per-element rounding used for this case
                };

                // Each case quantises the inputs to its grid, runs the product, and compares
                // against the float reference. The error is therefore the error that
                // precision introduces, not an artefact of the timing loop.
                for (const PrecisionCase& test_case : {
                         PrecisionCase{"fp8_e4m3", [](float v) { return static_cast<float>(FP8E4M3(v)); }},
                         PrecisionCase{"int8", [](float v) { return static_cast<float>(static_cast<int8_t>(std::round(std::fmin(std::fmax(v, -127.0f), 127.0f)))); }},
                     }) {
                    BenchmarkResult result;
                    result.name = std::string("precision_compare_") + test_case.label + " " + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
                    result.batch_size = m * n;
                    result.num_samples = 1;

                    // Max-abs error and relative error against the reference product.
                    double max_abs_error = 0.0;
                    double reference_norm = 0.0;
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < n; ++j) {
                            float sum = 0.0f;
                            for (int p = 0; p < k; ++p) {
                                sum += test_case.quantize(a[i * k + p]) * test_case.quantize(b[p * n + j]);
                            }
                            const float reference = c_ref[i * n + j];
                            max_abs_error = std::max<double>(max_abs_error, std::fabs(static_cast<double>(sum - reference)));
                            reference_norm += static_cast<double>(reference) * reference;
                        }
                    }

                    result.relative_error = reference_norm > 0.0 ? static_cast<float>(max_abs_error / std::sqrt(reference_norm)) : static_cast<float>(max_abs_error);
                    result.accuracy_loss = result.relative_error;
                    results.push_back(std::move(result));
                }

                return results;
            }

            void generate_report(const std::vector<BenchmarkResult>& results, const std::string& output_path) {
                std::ofstream file(output_path);
                if (file.is_open()) {
                    file << "# Quantization Benchmark Report\n\n";

                    file << "| Benchmark | Throughput (GB/s) | Latency (us) | Relative Error |\n";
                    file << "|-----------|-------------------|--------------|----------------|\n";

                    for (const auto& result : results) {
                        file << "| " << result.name << " | " << result.throughput_gbps << " | " << result.latency_us << " | " << result.relative_error << " |\n";
                    }
                }
            }

        }  // namespace benchmark
    }  // namespace quantize
}  // namespace nova
