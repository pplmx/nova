#include <cuda/performance/metric_aggregators.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cuda::performance {

    void ArithmeticIntensityAggregator::add_sample(uint64_t flops, size_t bytes_accessed) {
        if (bytes_accessed > 0) {
            samples_.push_back(static_cast<double>(flops) / static_cast<double>(bytes_accessed));
        }
    }

    double ArithmeticIntensityAggregator::get_average() const {
        if (samples_.empty()) {
            return 0.0;
        }
        return compute_mean(samples_);
    }

    double ArithmeticIntensityAggregator::get_peak() const {
        if (samples_.empty()) {
            return 0.0;
        }
        return *std::max_element(samples_.begin(), samples_.end());
    }

    AggregatedMetric ArithmeticIntensityAggregator::get_summary() const {
        AggregatedMetric summary;
        summary.name = "arithmetic_intensity";
        summary.sample_count = samples_.size();
        if (!samples_.empty()) {
            summary.min = *std::min_element(samples_.begin(), samples_.end());
            summary.max = *std::max_element(samples_.begin(), samples_.end());
            summary.mean = compute_mean(samples_);
            summary.stddev = compute_stddev(samples_);
        }
        return summary;
    }

    void ArithmeticIntensityAggregator::reset() {
        samples_.clear();
    }

    void FLOPsAggregator::add_sample(uint64_t flops, uint64_t elapsed_ns) {
        if (elapsed_ns > 0) {
            double gflops = (static_cast<double>(flops) / 1e9) / (static_cast<double>(elapsed_ns) / 1e9);
            achieved_samples_.push_back(gflops);
        }
    }

    double FLOPsAggregator::get_theoretical_gflops() const {
        return device_fp64_peak_;
    }

    double FLOPsAggregator::get_achieved_gflops() const {
        return compute_mean(achieved_samples_);
    }

    double FLOPsAggregator::get_efficiency_percent() const {
        double achieved = get_achieved_gflops();
        if (device_fp64_peak_ <= 0.0) {
            return 0.0;
        }
        return (achieved / device_fp64_peak_) * 100.0;
    }

    AggregatedMetric FLOPsAggregator::get_summary() const {
        AggregatedMetric summary;
        summary.name = "throughput_gflops";
        summary.sample_count = achieved_samples_.size();
        if (!achieved_samples_.empty()) {
            summary.min = *std::min_element(achieved_samples_.begin(), achieved_samples_.end());
            summary.max = *std::max_element(achieved_samples_.begin(), achieved_samples_.end());
            summary.mean = compute_mean(achieved_samples_);
            summary.stddev = compute_stddev(achieved_samples_);
        }
        return summary;
    }

    void FLOPsAggregator::reset() {
        achieved_samples_.clear();
    }

    void FLOPsAggregator::set_device_peak_flops(double fp64_gflops, double fp32_gflops, double fp16_gflops) {
        device_fp64_peak_ = fp64_gflops;
        device_fp32_peak_ = fp32_gflops;
        device_fp16_peak_ = fp16_gflops;
    }

    void BandwidthAggregator::add_sample(size_t bytes, uint64_t elapsed_ns, TransferType type) {
        if (elapsed_ns > 0) {
            double gbs = (static_cast<double>(bytes) / 1e9) / (static_cast<double>(elapsed_ns) / 1e9);
            switch (type) {
                case TransferType::HostToDevice:
                    h2d_samples_.push_back(gbs);
                    break;
                case TransferType::DeviceToHost:
                    d2h_samples_.push_back(gbs);
                    break;
                case TransferType::DeviceToDevice:
                    d2d_samples_.push_back(gbs);
                    break;
            }
        }
    }

    double BandwidthAggregator::get_h2d_bandwidth_gbs() const {
        return compute_mean(h2d_samples_);
    }

    double BandwidthAggregator::get_d2h_bandwidth_gbs() const {
        return compute_mean(d2h_samples_);
    }

    double BandwidthAggregator::get_d2d_bandwidth_gbs() const {
        return compute_mean(d2d_samples_);
    }

    double BandwidthAggregator::get_total_bandwidth_gbs() const {
        double total = 0.0;
        if (!h2d_samples_.empty()) {
            total += get_h2d_bandwidth_gbs();
        }
        if (!d2h_samples_.empty()) {
            total += get_d2h_bandwidth_gbs();
        }
        if (!d2d_samples_.empty()) {
            total += get_d2d_bandwidth_gbs();
        }
        return total;
    }

    double BandwidthAggregator::get_h2d_utilization_percent() const {
        if (peak_h2d_ <= 0.0) {
            return 0.0;
        }
        return (get_h2d_bandwidth_gbs() / peak_h2d_) * 100.0;
    }

    double BandwidthAggregator::get_d2h_utilization_percent() const {
        if (peak_d2h_ <= 0.0) {
            return 0.0;
        }
        return (get_d2h_bandwidth_gbs() / peak_d2h_) * 100.0;
    }

    double BandwidthAggregator::get_d2d_utilization_percent() const {
        if (peak_d2d_ <= 0.0) {
            return 0.0;
        }
        return (get_d2d_bandwidth_gbs() / peak_d2d_) * 100.0;
    }

    void BandwidthAggregator::set_peak_bandwidths(double h2d_gbs, double d2h_gbs, double d2d_gbs) {
        peak_h2d_ = h2d_gbs;
        peak_d2h_ = d2h_gbs;
        peak_d2d_ = d2d_gbs;
    }

    AggregatedMetric BandwidthAggregator::get_summary(TransferType type) const {
        AggregatedMetric summary;
        const std::vector<double>* samples = nullptr;

        switch (type) {
            case TransferType::HostToDevice:
                samples = &h2d_samples_;
                summary.name = "h2d_bandwidth_gbs";
                break;
            case TransferType::DeviceToHost:
                samples = &d2h_samples_;
                summary.name = "d2h_bandwidth_gbs";
                break;
            case TransferType::DeviceToDevice:
                samples = &d2d_samples_;
                summary.name = "d2d_bandwidth_gbs";
                break;
        }

        summary.sample_count = samples ? samples->size() : 0;
        if (samples && !samples->empty()) {
            summary.min = *std::min_element(samples->begin(), samples->end());
            summary.max = *std::max_element(samples->begin(), samples->end());
            summary.mean = compute_mean(*samples);
            summary.stddev = compute_stddev(*samples);
        }

        return summary;
    }

    void BandwidthAggregator::reset() {
        h2d_samples_.clear();
        d2h_samples_.clear();
        d2d_samples_.clear();
    }

    void MetricAggregatorPipeline::add_arithmetic_sample(uint64_t flops, size_t bytes) {
        ai_agg_.add_sample(flops, bytes);
    }

    void MetricAggregatorPipeline::add_flops_sample(uint64_t flops, uint64_t elapsed_ns) {
        flops_agg_.add_sample(flops, elapsed_ns);
    }

    void MetricAggregatorPipeline::add_bandwidth_sample(size_t bytes, uint64_t elapsed_ns, BandwidthAggregator::TransferType type) {
        bw_agg_.add_sample(bytes, elapsed_ns, type);
    }

    std::vector<AggregatedMetric> MetricAggregatorPipeline::get_all_summaries() const {
        return {ai_agg_.get_summary(),
                flops_agg_.get_summary(),
                bw_agg_.get_summary(BandwidthAggregator::TransferType::HostToDevice),
                bw_agg_.get_summary(BandwidthAggregator::TransferType::DeviceToHost),
                bw_agg_.get_summary(BandwidthAggregator::TransferType::DeviceToDevice)};
    }

    void MetricAggregatorPipeline::reset() {
        ai_agg_.reset();
        flops_agg_.reset();
        bw_agg_.reset();
    }

    void MetricAggregatorPipeline::set_device_peak_specs(double fp64_gflops, double fp32_gflops, double fp16_gflops, double h2d_gbs, double d2h_gbs, double d2d_gbs) {
        flops_agg_.set_device_peak_flops(fp64_gflops, fp32_gflops, fp16_gflops);
        bw_agg_.set_peak_bandwidths(h2d_gbs, d2h_gbs, d2d_gbs);
    }

    double compute_mean(const std::vector<double>& samples) {
        if (samples.empty()) {
            return 0.0;
        }
        return std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    }

    double compute_stddev(const std::vector<double>& samples) {
        if (samples.size() < 2) {
            return 0.0;
        }
        double mean = compute_mean(samples);
        double sq_sum = 0.0;
        for (double s : samples) {
            sq_sum += (s - mean) * (s - mean);
        }
        return std::sqrt(sq_sum / static_cast<double>(samples.size() - 1));
    }

}  // namespace cuda::performance
