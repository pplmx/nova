#include <cuda/performance/bandwidth/roofline_model.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace cuda::performance::bandwidth {

    DevicePeaks DevicePeaks::query(int device_id) {
        DevicePeaks peaks;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
            return peaks;
        }

        peaks.hbm_bandwidth_gbs = static_cast<double>(prop.memoryClockRate) * 1e3 * static_cast<double>(prop.memoryBusWidth) / 8.0 / 1e9;

        int major = prop.major;
        int minor = prop.minor;

        if (major >= 9) {
            peaks.fp64_gflops = static_cast<double>(prop.multiProcessorCount) * 32.0 * prop.clockRate * 1e3;
            peaks.fp32_gflops = peaks.fp64_gflops * 2.0;
            peaks.fp16_gflops = peaks.fp64_gflops * 2.0;
        } else if (major >= 7) {
            peaks.fp64_gflops = static_cast<double>(prop.multiProcessorCount) * 8.0 * prop.clockRate * 1e3;
            peaks.fp32_gflops = peaks.fp64_gflops * 2.0;
            peaks.fp16_gflops = peaks.fp64_gflops * 2.0;
        } else {
            peaks.fp64_gflops = static_cast<double>(prop.multiProcessorCount) * 64.0 * prop.clockRate * 1e3;
            peaks.fp32_gflops = peaks.fp64_gflops * 2.0;
            peaks.fp16_gflops = peaks.fp64_gflops * 2.0;
        }

        return peaks;
    }

    RooflineModel::RooflineModel()
        : peaks_(DevicePeaks::query(0)) {}

    void RooflineModel::add_point(const RooflinePoint& point) {
        points_.push_back(point);
    }

    void RooflineModel::add_point(const std::string& name, uint64_t flops, size_t bytes, uint64_t elapsed_ns) {
        RooflinePoint point;
        point.kernel_name = name;
        point.flops = flops;
        point.bytes_accessed = bytes;
        point.latency_ns = elapsed_ns;
        point.arithmetic_intensity = compute_arithmetic_intensity(flops, bytes);
        point.performance_gflops = compute_performance_gflops(flops, elapsed_ns);
        add_point(point);
    }

    void RooflineModel::clear() {
        points_.clear();
    }

    size_t RooflineModel::point_count() const {
        return points_.size();
    }

    double RooflineModel::compute_arithmetic_intensity(uint64_t flops, size_t bytes) const {
        if (bytes == 0) {
            return 0.0;
        }
        return static_cast<double>(flops) / static_cast<double>(bytes);
    }

    double RooflineModel::compute_performance_gflops(uint64_t flops, uint64_t elapsed_ns) const {
        if (elapsed_ns == 0) {
            return 0.0;
        }
        return (static_cast<double>(flops) / 1e9) / (static_cast<double>(elapsed_ns) / 1e9);
    }

    double RooflineModel::ridge_point() const {
        if (points_.empty()) {
            return 0.0;
        }
        if (peaks_.hbm_bandwidth_gbs <= 0.0) {
            return 0.0;
        }
        if (peaks_.fp64_gflops <= 0.0) {
            return 0.0;
        }
        return peaks_.fp64_gflops / peaks_.hbm_bandwidth_gbs;
    }

    std::vector<RooflinePoint> RooflineModel::get_points() const {
        return points_;
    }

    std::vector<RooflinePoint> RooflineModel::get_memory_bound_points() const {
        std::vector<RooflinePoint> result;
        double ridge = ridge_point();
        for (const auto& p : points_) {
            if (p.arithmetic_intensity < ridge) {
                result.push_back(p);
            }
        }
        return result;
    }

    std::vector<RooflinePoint> RooflineModel::get_compute_bound_points() const {
        std::vector<RooflinePoint> result;
        double ridge = ridge_point();
        for (const auto& p : points_) {
            if (p.arithmetic_intensity >= ridge) {
                result.push_back(p);
            }
        }
        return result;
    }

    RooflineModel::BoundType RooflineModel::classify_point(double arithmetic_intensity) const {
        double ridge = ridge_point();
        if (arithmetic_intensity < ridge * 0.9) {
            return BoundType::Memory;
        }
        if (arithmetic_intensity > ridge * 1.1) {
            return BoundType::Compute;
        }
        return BoundType::Unknown;
    }

    std::string RooflineModel::to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"peaks\": {\n";
        oss << "    \"fp64_gflops\": " << std::fixed << std::setprecision(1) << peaks_.fp64_gflops << ",\n";
        oss << "    \"fp32_gflops\": " << peaks_.fp32_gflops << ",\n";
        oss << "    \"fp16_gflops\": " << peaks_.fp16_gflops << ",\n";
        oss << "    \"hbm_bandwidth_gbs\": " << peaks_.hbm_bandwidth_gbs << "\n";
        oss << "  },\n";
        oss << "  \"ridge_point\": " << std::fixed << std::setprecision(2) << ridge_point() << ",\n";
        oss << "  \"points\": [\n";

        for (size_t i = 0; i < points_.size(); ++i) {
            const auto& p = points_[i];
            if (i > 0) {
                oss << ",\n";
            }
            oss << "    {\"name\": \"" << p.kernel_name << "\", " << "\"arithmetic_intensity\": " << std::fixed << std::setprecision(4) << p.arithmetic_intensity << ", "
                << "\"performance_gflops\": " << p.performance_gflops << "}";
        }

        oss << "\n  ]\n";
        oss << "}\n";
        return oss.str();
    }

    std::string RooflineModel::to_csv() const {
        std::ostringstream oss;
        oss << "kernel_name,arithmetic_intensity,performance_gflops,flops,bytes,latency_us\n";

        for (const auto& p : points_) {
            oss << "\"" << p.kernel_name << "\"," << std::fixed << std::setprecision(4) << p.arithmetic_intensity << "," << p.performance_gflops << "," << p.flops << ","
                << p.bytes_accessed << "," << (p.latency_ns / 1000.0) << "\n";
        }

        return oss.str();
    }

    BandwidthUtilizationTracker::BandwidthUtilizationTracker() {}

    void BandwidthUtilizationTracker::add_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, false});
    }

    void BandwidthUtilizationTracker::add_h2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, true});
    }

    void BandwidthUtilizationTracker::add_d2h_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, true});
    }

    void BandwidthUtilizationTracker::add_d2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, true});
    }

    void BandwidthUtilizationTracker::reset() {
        samples_.clear();
    }

    double BandwidthUtilizationTracker::average_bandwidth_gbs() const {
        if (samples_.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (const auto& s : samples_) {
            sum += s.bandwidth_gbs;
        }
        return sum / static_cast<double>(samples_.size());
    }

    double BandwidthUtilizationTracker::peak_bandwidth_gbs() const {
        if (samples_.empty()) {
            return 0.0;
        }
        double peak = 0.0;
        for (const auto& s : samples_) {
            peak = std::max(peak, s.bandwidth_gbs);
        }
        return peak;
    }

    double BandwidthUtilizationTracker::utilization_percent() const {
        if (samples_.empty()) {
            return 0.0;
        }
        return (average_bandwidth_gbs() / peak_d2d_) * 100.0;
    }

    double BandwidthUtilizationTracker::h2d_utilization_percent() const {
        if (peak_h2d_ <= 0.0) {
            return 0.0;
        }
        double avg = average_bandwidth_gbs();
        return (avg / peak_h2d_) * 100.0;
    }

    double BandwidthUtilizationTracker::d2h_utilization_percent() const {
        if (peak_d2h_ <= 0.0) {
            return 0.0;
        }
        double avg = average_bandwidth_gbs();
        return (avg / peak_d2h_) * 100.0;
    }

    double BandwidthUtilizationTracker::d2d_utilization_percent() const {
        if (peak_d2d_ <= 0.0) {
            return 0.0;
        }
        double avg = average_bandwidth_gbs();
        return (avg / peak_d2d_) * 100.0;
    }

    void BandwidthUtilizationTracker::set_peak_bandwidth(double h2d, double d2h, double d2d) {
        peak_h2d_ = h2d;
        peak_d2h_ = d2h;
        peak_d2d_ = d2d;
    }

    size_t BandwidthUtilizationTracker::sample_count() const {
        return samples_.size();
    }

    bool BandwidthUtilizationTracker::has_low_utilization_warning() const {
        return get_min_utilization() < 50.0;
    }

    double BandwidthUtilizationTracker::get_min_utilization() const {
        if (samples_.empty()) {
            return 0.0;
        }
        double min_util = 100.0;
        for (const auto& s : samples_) {
            double util = (peak_d2d_ > 0.0) ? (s.bandwidth_gbs / peak_d2d_) * 100.0 : 0.0;
            min_util = std::min(min_util, util);
        }
        return min_util;
    }

    std::string BandwidthUtilizationTracker::to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"samples\": " << samples_.size() << ",\n";
        oss << "  \"average_bandwidth_gbs\": " << std::fixed << std::setprecision(2) << average_bandwidth_gbs() << ",\n";
        oss << "  \"peak_bandwidth_gbs\": " << peak_bandwidth_gbs() << ",\n";
        oss << "  \"utilization_percent\": " << utilization_percent() << ",\n";
        oss << "  \"h2d_utilization_percent\": " << h2d_utilization_percent() << ",\n";
        oss << "  \"d2h_utilization_percent\": " << d2h_utilization_percent() << ",\n";
        oss << "  \"d2d_utilization_percent\": " << d2d_utilization_percent() << ",\n";
        oss << "  \"low_utilization_warning\": " << (has_low_utilization_warning() ? "true" : "false") << "\n";
        oss << "}";
        return oss.str();
    }

}  // namespace cuda::performance::bandwidth
