#ifndef NOVA_CUDA_SPARSE_ROOFLINE_HPP
#define NOVA_CUDA_SPARSE_ROOFLINE_HPP

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

namespace nova {
namespace sparse {

enum class Precision { FP64, FP32, FP16 };

enum class PerformanceBound {
    COMPUTE_BOUND,
    MEMORY_BOUND,
    BALANCED,
    UNKNOWN
};

struct DevicePeaks {
    double fp64_peak_gflops = 0.0;
    double fp32_peak_gflops = 0.0;
    double fp16_peak_gflops = 0.0;
    double memory_bandwidth_gbps = 0.0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multiprocessor_count = 0;
    int clock_rate_khz = 0;
};

inline DevicePeaks get_device_peaks(int device_id = 0) {
    DevicePeaks peaks;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    peaks.compute_capability_major = device_prop.major;
    peaks.compute_capability_minor = device_prop.minor;
    peaks.multiprocessor_count = device_prop.multiProcessorCount;
    peaks.clock_rate_khz = device_prop.clockRate;

    int cc = device_prop.major * 10 + device_prop.minor;

    size_t bandwidth_bps = static_cast<size_t>(2) *
                          static_cast<size_t>(device_prop.memoryClockRate) * 1000 *
                          static_cast<size_t>(device_prop.memoryBusWidth) / 8;
    peaks.memory_bandwidth_gbps = static_cast<double>(bandwidth_bps) / (1024.0 * 1024.0 * 1024.0);

    double clock_ghz = static_cast<double>(device_prop.clockRate) / 1e6;

    double ops_per_sm_per_clock = 2.0;

    switch (cc) {
        case 90:
        case 86:
        case 89:
        case 80:
        case 75:
            peaks.fp64_peak_gflops = peaks.multiprocessor_count * clock_ghz * 16 * 2;
            peaks.fp32_peak_gflops = peaks.multiprocessor_count * clock_ghz * 128;
            peaks.fp16_peak_gflops = peaks.multiprocessor_count * clock_ghz * 512;
            break;
        case 70:
            peaks.fp64_peak_gflops = peaks.multiprocessor_count * clock_ghz * 8 * 2;
            peaks.fp32_peak_gflops = peaks.multiprocessor_count * clock_ghz * 64;
            peaks.fp16_peak_gflops = peaks.multiprocessor_count * clock_ghz * 256;
            break;
        case 60:
            peaks.fp64_peak_gflops = peaks.multiprocessor_count * clock_ghz * 8 * 2;
            peaks.fp32_peak_gflops = peaks.multiprocessor_count * clock_ghz * 32;
            peaks.fp16_peak_gflops = peaks.multiprocessor_count * clock_ghz * 64;
            break;
        default:
            peaks.fp64_peak_gflops = peaks.multiprocessor_count * clock_ghz * 8 * 2;
            peaks.fp32_peak_gflops = peaks.multiprocessor_count * clock_ghz * 32;
            peaks.fp16_peak_gflops = peaks.multiprocessor_count * clock_ghz * 64;
            break;
    }

    return peaks;
}

struct RooflineMetrics {
    double arithmetic_intensity = 0.0;
    double achieved_gflops = 0.0;
    double peak_gflops = 0.0;
    double achieved_bandwidth_gbps = 0.0;
    double peak_bandwidth_gbps = 0.0;
    PerformanceBound bound = PerformanceBound::UNKNOWN;
    const char* kernel_name = nullptr;
};

template<typename T>
inline double arithmetic_intensity(long long flops, size_t bytes_accessed) {
    if (bytes_accessed == 0) return 0.0;
    return static_cast<double>(flops) / static_cast<double>(bytes_accessed);
}

template<typename T>
inline double spmv_arithmetic_intensity(int nnz, int n) {
    long long flops = 2LL * nnz;
    size_t bytes = static_cast<size_t>(nnz) * sizeof(T) +
                   static_cast<size_t>(nnz) * sizeof(int) +
                   static_cast<size_t>(n) * sizeof(T) * 2;
    return arithmetic_intensity<T>(flops, bytes);
}

class RooflineAnalyzer {
public:
    RooflineAnalyzer() : peaks_(get_device_peaks()) {}

    explicit RooflineAnalyzer(int device_id) : peaks_(get_device_peaks(device_id)) {}

    double compute_peak_performance(Precision p) const {
        switch (p) {
            case Precision::FP64:
                return peaks_.fp64_peak_gflops;
            case Precision::FP32:
                return peaks_.fp32_peak_gflops;
            case Precision::FP16:
                return peaks_.fp16_peak_gflops;
            default:
                return peaks_.fp32_peak_gflops;
        }
    }

    double compute_arithmetic_intensity(long long flops, size_t bytes) const {
        return ::nova::sparse::arithmetic_intensity<double>(flops, bytes);
    }

    template<typename T>
    double compute_arithmetic_intensity_spmv(const SparseMatrixCSR<T>& A) const {
        return spmv_arithmetic_intensity<T>(A.nnz(), A.num_rows());
    }

    PerformanceBound classify(double arithmetic_intensity, double peak_gflops) const {
        double bandwidth_ceiling = arithmetic_intensity * peaks_.memory_bandwidth_gbps;

        if (bandwidth_ceiling < peak_gflops * 0.9) {
            return PerformanceBound::MEMORY_BOUND;
        } else if (bandwidth_ceiling > peak_gflops * 1.1) {
            return PerformanceBound::COMPUTE_BOUND;
        } else {
            return PerformanceBound::BALANCED;
        }
    }

    PerformanceBound classify(const RooflineMetrics& m) const {
        return classify(m.arithmetic_intensity, m.peak_gflops);
    }

    struct ClassificationConfidence {
        PerformanceBound bound;
        double confidence_percent;
        double bandwidth_ceiling;
        double compute_ceiling;
    };

    ClassificationConfidence classify_with_confidence(double ai, double peak_gflops) const {
        ClassificationConfidence result;
        double bandwidth_ceiling = ai * peaks_.memory_bandwidth_gbps;
        result.bandwidth_ceiling = bandwidth_ceiling;
        result.compute_ceiling = peak_gflops;

        double ratio = bandwidth_ceiling / peak_gflops;

        if (ratio < 0.85) {
            result.bound = PerformanceBound::MEMORY_BOUND;
            result.confidence_percent = (1.0 - ratio) * 100.0;
        } else if (ratio > 1.15) {
            result.bound = PerformanceBound::COMPUTE_BOUND;
            result.confidence_percent = (ratio - 1.0) * 50.0 + 50.0;
        } else {
            result.bound = PerformanceBound::BALANCED;
            result.confidence_percent = 100.0 - std::abs(ratio - 1.0) * 200.0;
        }

        return result;
    }

    static const char* bound_to_string(PerformanceBound b) {
        switch (b) {
            case PerformanceBound::COMPUTE_BOUND: return "COMPUTE_BOUND";
            case PerformanceBound::MEMORY_BOUND: return "MEMORY_BOUND";
            case PerformanceBound::BALANCED: return "BALANCED";
            default: return "UNKNOWN";
        }
    }

    std::string to_json(const RooflineMetrics& m, Precision p = Precision::FP32) const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"kernel_name\": \"" << (m.kernel_name ? m.kernel_name : "unknown") << "\",\n";
        oss << "  \"arithmetic_intensity\": " << m.arithmetic_intensity << ",\n";
        oss << "  \"achieved_gflops\": " << m.achieved_gflops << ",\n";
        oss << "  \"peak_gflops\": " << m.peak_gflops << ",\n";
        oss << "  \"achieved_bandwidth_gbps\": " << m.achieved_bandwidth_gbps << ",\n";
        oss << "  \"peak_bandwidth_gbps\": " << m.peak_bandwidth_gbps << ",\n";
        oss << "  \"performance_bound\": \"" << bound_to_string(m.bound) << "\",\n";
        double efficiency = m.peak_gflops > 0 ? (m.achieved_gflops / m.peak_gflops * 100.0) : 0.0;
        oss << "  \"efficiency_percent\": " << efficiency << "\n";
        oss << "}";
        return oss.str();
    }

    std::string to_json_device_info(Precision p = Precision::FP32) const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"peak_gflops\": " << compute_peak_performance(p) << ",\n";
        oss << "  \"peak_bandwidth_gbps\": " << peaks_.memory_bandwidth_gbps << ",\n";
        oss << "  \"fp64_peak_gflops\": " << peaks_.fp64_peak_gflops << ",\n";
        oss << "  \"fp32_peak_gflops\": " << peaks_.fp32_peak_gflops << ",\n";
        oss << "  \"fp16_peak_gflops\": " << peaks_.fp16_peak_gflops << ",\n";
        oss << "  \"compute_capability\": \"" << peaks_.compute_capability_major
            << "." << peaks_.compute_capability_minor << "\"\n";
        oss << "}";
        return oss.str();
    }

    RooflineMetrics analyze_kernel(const char* name, long long flops,
                                   double measured_time_ms, size_t bytes,
                                   Precision p = Precision::FP32) {
        RooflineMetrics metrics;
        metrics.kernel_name = name;
        metrics.arithmetic_intensity = compute_arithmetic_intensity(flops, bytes);
        metrics.peak_gflops = compute_peak_performance(p);
        metrics.peak_bandwidth_gbps = peaks_.memory_bandwidth_gbps;

        if (measured_time_ms > 0) {
            metrics.achieved_gflops = flops / (measured_time_ms * 1e6);
            double bandwidth_bytes_per_ms = bytes / measured_time_ms;
            metrics.achieved_bandwidth_gbps = bandwidth_bytes_per_ms * 1e-6;
        }

        metrics.bound = classify(metrics);

        return metrics;
    }

    const DevicePeaks& device_peaks() const { return peaks_; }

private:
    DevicePeaks peaks_;
};

template<typename T>
struct KernelMetrics {
    long long flops = 0;
    size_t bytes_read = 0;
    size_t bytes_written = 0;
    double time_ms = 0.0;
    T achieved_gflops = T{0};
    T achieved_bandwidth_gbps = T{0};
};

class ScopedTimer {
public:
    ScopedTimer(cudaEvent_t* start, cudaEvent_t* stop)
        : start_event_(start), stop_event_(stop) {
        cudaEventRecord(*start_event_);
    }

    float elapsed_ms() {
        cudaEventRecord(*stop_event_);
        cudaEventSynchronize(*stop_event_);
        float elapsed;
        cudaEventElapsedTime(&elapsed, *start_event_, *stop_event_);
        return elapsed;
    }

private:
    cudaEvent_t* start_event_;
    cudaEvent_t* stop_event_;
};

inline double measure_bandwidth(int device_id, size_t size_bytes, int iterations = 10) {
    cudaSetDevice(device_id);

    char* d_src;
    char* d_dst;
    cudaMalloc(&d_src, size_bytes);
    cudaMalloc(&d_dst, size_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpy(d_dst, d_src, size_bytes, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double bandwidth_gbps = (static_cast<double>(size_bytes) * iterations) /
                            (elapsed_ms * 1e6);
    return bandwidth_gbps;
}

class RooflineAnalysis {
public:
    void add_kernel(const RooflineMetrics& metrics) {
        kernels_.push_back(metrics);
    }

    void set_device_info(const DevicePeaks& peaks, Precision p, const char* device_name = "Unknown") {
        device_peaks_ = peaks;
        precision_ = p;
        device_name_ = device_name;
    }

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"metadata\": {\n";
        oss << "    \"device_name\": \"" << device_name_ << "\",\n";
        oss << "    \"compute_capability\": \"" << device_peaks_.compute_capability_major
            << "." << device_peaks_.compute_capability_minor << "\",\n";
        oss << "    \"precision\": \"" << precision_to_string(precision_) << "\",\n";
        oss << "    \"timestamp\": \"" << get_timestamp() << "\"\n";
        oss << "  },\n";
        oss << "  \"device_peaks\": {\n";
        oss << "    \"peak_gflops\": " << get_peak_gflops() << ",\n";
        oss << "    \"peak_bandwidth_gbps\": " << device_peaks_.memory_bandwidth_gbps << ",\n";
        oss << "    \"fp64_peak_gflops\": " << device_peaks_.fp64_peak_gflops << ",\n";
        oss << "    \"fp32_peak_gflops\": " << device_peaks_.fp32_peak_gflops << ",\n";
        oss << "    \"fp16_peak_gflops\": " << device_peaks_.fp16_peak_gflops << "\n";
        oss << "  },\n";
        oss << "  \"kernels\": [\n";

        for (size_t i = 0; i < kernels_.size(); ++i) {
            const auto& m = kernels_[i];
            oss << "    {\n";
            oss << "      \"name\": \"" << (m.kernel_name ? m.kernel_name : "unknown") << "\",\n";
            oss << "      \"arithmetic_intensity\": " << m.arithmetic_intensity << ",\n";
            oss << "      \"achieved_gflops\": " << m.achieved_gflops << ",\n";
            oss << "      \"peak_gflops\": " << m.peak_gflops << ",\n";
            oss << "      \"achieved_bandwidth_gbps\": " << m.achieved_bandwidth_gbps << ",\n";
            oss << "      \"peak_bandwidth_gbps\": " << m.peak_bandwidth_gbps << ",\n";
            oss << "      \"performance_bound\": \"" << RooflineAnalyzer::bound_to_string(m.bound) << "\",\n";
            double eff = m.peak_gflops > 0 ? (m.achieved_gflops / m.peak_gflops * 100.0) : 0.0;
            oss << "      \"efficiency_percent\": " << eff << "\n";
            oss << "    }";
            if (i < kernels_.size() - 1) oss << ",";
            oss << "\n";
        }

        oss << "  ]\n";
        oss << "}";
        return oss.str();
    }

private:
    std::vector<RooflineMetrics> kernels_;
    DevicePeaks device_peaks_;
    Precision precision_ = Precision::FP32;
    std::string device_name_;

    double get_peak_gflops() const {
        switch (precision_) {
            case Precision::FP64: return device_peaks_.fp64_peak_gflops;
            case Precision::FP32: return device_peaks_.fp32_peak_gflops;
            case Precision::FP16: return device_peaks_.fp16_peak_gflops;
            default: return device_peaks_.fp32_peak_gflops;
        }
    }

    static const char* precision_to_string(Precision p) {
        switch (p) {
            case Precision::FP64: return "FP64";
            case Precision::FP32: return "FP32";
            case Precision::FP16: return "FP16";
            default: return "FP32";
        }
    }

    static std::string get_timestamp() {
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
        return std::string(buf);
    }
};

}
}

#endif
