#include <cuda/quantize/qat.hpp>
#include <cuda/quantize/calibrator.hpp>
#include <cuda/quantize/int8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>
#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda/quantize/fp8_activation.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>

#ifndef NOVA_CUDA_QUANTIZE_BENCHMARK_HPP
#define NOVA_CUDA_QUANTIZE_BENCHMARK_HPP

namespace nova {
namespace quantize {
namespace benchmark {

struct BenchmarkResult {
    std::string name;
    float throughput_gbps{0.0f};
    float latency_us{0.0f};
    float accuracy_loss{0.0f};
    float relative_error{0.0f};
    int batch_size{0};
    int num_samples{0};

    void print() const {
        printf("%s: %.2f GB/s, %.2f us latency, %.4f rel_error\n",
               name.c_str(), throughput_gbps, latency_us, relative_error);
    }

    void save_json(std::ostream& os) const {
        os << "{\n";
        os << "  \"name\": \"" << name << "\",\n";
        os << "  \"throughput_gbps\": " << throughput_gbps << ",\n";
        os << "  \"latency_us\": " << latency_us << ",\n";
        os << "  \"relative_error\": " << relative_error << ",\n";
        os << "  \"batch_size\": " << batch_size << ",\n";
        os << "  \"num_samples\": " << num_samples << "\n";
        os << "}\n";
    }
};

class QuantizationBenchmark {
public:
    struct Config {
        int warmup_runs{10};
        int benchmark_runs{100};
        int batch_size{1024};
        bool verify_accuracy{true};
        bool verbose{false};
    };

    explicit QuantizationBenchmark(const Config& config = Config{})
        : config_(config), stream_(0) {}

    void set_stream(cudaStream_t stream) { stream_ = stream; }

    BenchmarkResult benchmark_fp8_quantization(
        const std::vector<float>& data, Precision precision = Precision::FP8_E4M3);

    BenchmarkResult benchmark_int8_quantization(
        const std::vector<float>& data);

    BenchmarkResult benchmark_dequantization(
        const std::vector<int8_t>& data, Precision precision = Precision::INT8);

    BenchmarkResult benchmark_fp8_gemm(
        int m, int k, int n);

    BenchmarkResult benchmark_calibration(
        const std::vector<float>& data, Calibrator& calibrator);

    BenchmarkResult benchmark_accuracy_comparison(
        const std::vector<float>& original,
        const std::vector<float>& quantized,
        const std::string& metric = "l2");

    void run_all_benchmarks();

    const std::vector<BenchmarkResult>& get_results() const { return results_; }
    void save_results_json(const std::string& path) const;
    void print_summary() const;

private:
    Config config_;
    cudaStream_t stream_;
    std::vector<BenchmarkResult> results_;

    std::vector<float> generate_random_data(size_t n, float scale = 1.0f);
    float compute_l2_error(const std::vector<float>& a, const std::vector<float>& b);
    float compute_kl_divergence(const std::vector<float>& a, const std::vector<float>& b);
};

BenchmarkResult benchmark_fp8_roundtrip(
    const float* data, size_t n,
    Precision precision = Precision::FP8_E4M3);

BenchmarkResult benchmark_fp8_gemm_throughput(
    int m, int k, int n,
    int num_runs = 100);

std::vector<BenchmarkResult> run_precision_comparison(
    const float* a, const float* b, const float* c_ref,
    int m, int k, int n);

void generate_report(
    const std::vector<BenchmarkResult>& results,
    const std::string& output_path);

} // namespace benchmark
} // namespace quantize
} // namespace nova

#endif // NOVA_CUDA_QUANTIZE_BENCHMARK_HPP
