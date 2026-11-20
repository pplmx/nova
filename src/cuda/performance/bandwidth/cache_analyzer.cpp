#include <cuda/performance/bandwidth/cache_analyzer.h>

#include <cuda_runtime.h>

#include <sstream>
#include <iomanip>

namespace cuda::performance::bandwidth {

    CacheAnalyzer::CacheAnalyzer() {
        available_ = check_cupti_availability();
        if (!available_) {
            error_message_ = "CUPTI not available for cache analysis";
        }
    }

    CacheMetrics CacheAnalyzer::analyze(int device_id) {
        CacheMetrics metrics;

        if (!available_) {
            metrics.error_message = error_message_;
            return metrics;
        }

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
            metrics.error_message = "Failed to get device properties";
            return metrics;
        }

        // The hit rates that used to be set here (0.85, 0.95, 0.99) were literals, not
        // measurements: no counter was read, and analyze_kernel() picked its numbers by
        // substring-matching the kernel name. They were plausible enough to be mistaken for
        // real results by anything that logged them, which is worse than reporting nothing.
        metrics.available = false;
        metrics.error_message = "cache hit rates require CUPTI, which this build does not use";

        return metrics;
    }

    CacheMetrics CacheAnalyzer::analyze_kernel(const std::string& kernel_name, int device_id) {
        CacheMetrics metrics;

        if (!available_) {
            metrics.error_message = error_message_;
            return metrics;
        }

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
            metrics.error_message = "Failed to get device properties";
            return metrics;
        }

        // These numbers were chosen by substring-matching the kernel name against "shared"
        // and "global" — a kernel called "shared_matmul" got 95% L1 and one called "matmul"
        // got 85%, with nothing measured either way. Reporting a fabricated figure under a
        // field named l1_hit_rate is the most damaging shape a stub can take, because the
        // value looks like data.
        (void)kernel_name;
        (void)device_id;
        metrics.available = false;
        metrics.error_message =
            "per-kernel cache hit rates require CUPTI, which this build "
            "does not use";

        return metrics;
    }

    bool CacheAnalyzer::is_available() const {
        return available_;
    }

    std::string CacheAnalyzer::get_error_message() const {
        return error_message_;
    }

    bool CacheAnalyzer::check_cupti_availability() {
        // Nothing in this file links against CUPTI or calls into it, so the honest answer is
        // that cache counters are not available. This used to return true unconditionally,
        // which let analyze() and analyze_kernel() run to completion and hand back invented
        // hit rates (see below).
        return false;
    }

    BandwidthAnalysis::BandwidthAnalysis()
        : roofline_(std::make_unique<RooflineModel>()),
          tracker_(std::make_unique<BandwidthUtilizationTracker>()) {}

    BandwidthAnalysis::~BandwidthAnalysis() = default;

    void BandwidthAnalysis::set_roofline_peak_flops(double fp64, double fp32, double fp16) {
        // Push the values into the model that actually produces the numbers. These two
        // setters used to write peak_fp64_/peak_fp32_/peak_fp16_/peak_bandwidth_ members
        // that nothing ever read: get_report() and generate_roofline_json() both go
        // through roofline_->peaks(), which kept whatever DevicePeaks::query() returned
        // at construction. Configuring a device peak therefore had no effect on any
        // output, and nothing said so. The shadow members are gone rather than kept in
        // sync: a field that looks like state but feeds nothing is how the bug was
        // possible.
        if (roofline_) {
            DevicePeaks peaks = roofline_->peaks();
            peaks.fp64_gflops = fp64;
            peaks.fp32_gflops = fp32;
            peaks.fp16_gflops = fp16;
            roofline_->set_peaks(peaks);
        }
    }

    void BandwidthAnalysis::set_roofline_peak_bandwidth(double hbm_gbs) {
        if (roofline_) {
            DevicePeaks peaks = roofline_->peaks();
            peaks.hbm_bandwidth_gbs = hbm_gbs;
            roofline_->set_peaks(peaks);
        }
        if (tracker_) {
            // The tracker derives utilization from the peaks; leaving it at the 900 GB/s
            // default while the model reported something else made the two halves of a
            // single report describe different devices.
            tracker_->set_peak_bandwidth(hbm_gbs, hbm_gbs, hbm_gbs);
        }
    }

    void BandwidthAnalysis::add_kernel_sample(const std::string& name, uint64_t flops, size_t bytes, uint64_t elapsed_ns) {
        if (roofline_) {
            roofline_->add_point(name, flops, bytes, elapsed_ns);
        }
    }

    void BandwidthAnalysis::add_bandwidth_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns) {
        if (tracker_) {
            tracker_->add_sample(bandwidth_gbs, bytes, elapsed_ns);
        }
    }

    std::string BandwidthAnalysis::generate_roofline_json() const {
        if (!roofline_) {
            return "{}";
        }
        return roofline_->to_json();
    }

    std::string BandwidthAnalysis::generate_bandwidth_json() const {
        if (!tracker_) {
            return "{}";
        }
        return tracker_->to_json();
    }

    std::string BandwidthAnalysis::get_report() const {
        std::ostringstream oss;
        oss << "Bandwidth Analysis Report\n";
        oss << "=========================\n\n";

        oss << "Roofline Model:\n";
        if (roofline_) {
            const auto& peaks = roofline_->peaks();
            oss << "  Peak FP64: " << std::fixed << std::setprecision(1) << peaks.fp64_gflops << " GFLOPS\n";
            oss << "  Peak FP32: " << peaks.fp32_gflops << " GFLOPS\n";
            oss << "  Peak Bandwidth: " << peaks.hbm_bandwidth_gbs << " GB/s\n";
            oss << "  Ridge Point: " << std::fixed << std::setprecision(2) << roofline_->ridge_point() << " FLOP/Byte\n";
            oss << "  Points: " << roofline_->point_count() << "\n";

            auto memory_bound = roofline_->get_memory_bound_points();
            auto compute_bound = roofline_->get_compute_bound_points();
            oss << "  Memory-bound: " << memory_bound.size() << " kernels\n";
            oss << "  Compute-bound: " << compute_bound.size() << " kernels\n";
        }

        oss << "\nBandwidth Utilization:\n";
        if (tracker_) {
            oss << "  Average: " << std::fixed << std::setprecision(2) << tracker_->average_bandwidth_gbs() << " GB/s\n";
            oss << "  Peak: " << tracker_->peak_bandwidth_gbs() << " GB/s\n";
            oss << "  Utilization: " << tracker_->utilization_percent() << "%\n";
            oss << "  Samples: " << tracker_->sample_count() << "\n";

            if (tracker_->has_low_utilization_warning()) {
                oss << "  WARNING: Low utilization detected (" << tracker_->get_min_utilization() << "%)\n";
            }
        }

        return oss.str();
    }

}  // namespace cuda::performance::bandwidth
