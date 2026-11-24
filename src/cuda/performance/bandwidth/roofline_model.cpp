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

        // Peak FP64 in GFLOPS = SMs * (FP64 cores per SM) * 2 * clock.
        //
        // The factor 2 is the FMA: each core retires one multiply-add, i.e. two
        // floating-point operations, per clock. The previous table omitted it and used
        // core counts that did not match the hardware --
        //
        //     major >= 9: 32      (Hopper H100: 64)
        //     major >= 7:  8      (Volta V100: 32, Ampere A100: 32, Ada consumer: 2)
        //     otherwise:  64      (Pascal P100: 32)
        //
        // so an A100 reported 108 * 8 * 1.41e9 = 1.22 TFLOPS against a published 9.7 --
        // 8x low, measured on this host. Every ridge-point and memory-bound/compute-bound
        // classification was made against that wrong number.
        //
        // The counts below are the datacenter parts' published FP64 cores per SM. Parts
        // whose FP64 is cut down (consumer Ada/Ampere, 2 per SM) will still be
        // overestimated: cudaDeviceProp carries no core-count field, so this table is the
        // best available from the device properties alone, and it is documented as such
        // rather than presented as exact.
        double fp64_cores_per_sm = 32.0;
        if (prop.major >= 9) {
            fp64_cores_per_sm = 64.0;  // Hopper / Blackwell datacenter
        } else if (prop.major == 6) {
            fp64_cores_per_sm = 32.0;  // Pascal P100
        }

        // Divide by 1e9 on purpose. prop.clockRate is in kHz, so clockRate * 1e3 is Hz and
        // the product below is FLOP per second. The field is named *_gflops and
        // ridge_point() divides it by a GB/s bandwidth to get FLOP/Byte, so it has to be
        // GFLOPS. Without this division every peak was 1e9 too large and the ridge point
        // came out around 9.5e9 FLOP/Byte, which no kernel can reach -- measured, and it
        // made get_compute_bound_points() return empty for every workload.
        const double clock_hz = static_cast<double>(prop.clockRate) * 1e3;
        peaks.fp64_gflops = static_cast<double>(prop.multiProcessorCount) * fp64_cores_per_sm * 2.0 * clock_hz / 1e9;
        peaks.fp32_gflops = peaks.fp64_gflops * 2.0;
        peaks.fp16_gflops = peaks.fp64_gflops * 2.0;

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

    // The four adders used to store a single bool, and add_d2h_sample/add_d2d_sample both set it to
    // the same value as add_h2d_sample, so the record could not say which link a sample came from.
    // Nothing read the flag either. Each sample now carries its direction, which is what lets the
    // per-direction accessors below select their own samples.
    void BandwidthUtilizationTracker::add_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, TransferDirection::D2D});
    }

    void BandwidthUtilizationTracker::add_h2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, TransferDirection::H2D});
    }

    void BandwidthUtilizationTracker::add_d2h_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, TransferDirection::D2H});
    }

    void BandwidthUtilizationTracker::add_d2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        samples_.push_back({bandwidth_gbs, bytes, elapsed_ns, TransferDirection::D2D});
    }

    namespace {

        /// Mean bandwidth of the samples in one direction, or 0 when that direction has none.
        double mean_for(const std::vector<BandwidthSample>& samples, TransferDirection direction) {
            double sum = 0.0;
            size_t count = 0;
            for (const auto& s : samples) {
                if (s.direction == direction) {
                    sum += s.bandwidth_gbs;
                    ++count;
                }
            }
            return count == 0 ? 0.0 : sum / static_cast<double>(count);
        }

    }  // namespace

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
        // Only this direction's samples: dividing the mean of everything by one link's peak gives a
        // ratio that is not a utilization. Measured before: one H2D sample at 10 GB/s plus one D2D
        // sample at 450 GB/s made h2d_utilization_percent() report 1150% while the honest answer was
        // 50%.
        return (mean_for(samples_, TransferDirection::H2D) / peak_h2d_) * 100.0;
    }

    double BandwidthUtilizationTracker::d2h_utilization_percent() const {
        if (peak_d2h_ <= 0.0) {
            return 0.0;
        }
        // Only this direction's samples: dividing the mean of everything by one link's peak gives a
        // ratio that is not a utilization. Measured before: one H2D sample at 10 GB/s plus one D2D
        // sample at 450 GB/s made h2d_utilization_percent() report 1150% while the honest answer was
        // 50%.
        return (mean_for(samples_, TransferDirection::D2H) / peak_d2h_) * 100.0;
    }

    double BandwidthUtilizationTracker::d2d_utilization_percent() const {
        if (peak_d2d_ <= 0.0) {
            return 0.0;
        }
        // Only this direction's samples: dividing the mean of everything by one link's peak gives a
        // ratio that is not a utilization. Measured before: one H2D sample at 10 GB/s plus one D2D
        // sample at 450 GB/s made h2d_utilization_percent() report 1150% while the honest answer was
        // 50%.
        return (mean_for(samples_, TransferDirection::D2D) / peak_d2d_) * 100.0;
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
