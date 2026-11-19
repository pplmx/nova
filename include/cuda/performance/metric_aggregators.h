#pragma once

#include <cuda/performance/nvblox_metrics.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuda::performance {

    struct AggregatedMetric {
        std::string name;
        double min{0.0};
        double max{0.0};
        double mean{0.0};
        double stddev{0.0};
        size_t sample_count{0};
    };

    class ArithmeticIntensityAggregator {
    public:
        void add_sample(uint64_t flops, size_t bytes_accessed);

        [[nodiscard]] double get_average() const;
        [[nodiscard]] double get_peak() const;
        [[nodiscard]] AggregatedMetric get_summary() const;

        void reset();

    private:
        std::vector<double> samples_;
    };

    class FLOPsAggregator {
    public:
        void add_sample(uint64_t flops, uint64_t elapsed_ns);

        [[nodiscard]] double get_theoretical_gflops() const;
        [[nodiscard]] double get_achieved_gflops() const;
        [[nodiscard]] double get_efficiency_percent() const;
        [[nodiscard]] AggregatedMetric get_summary() const;

        void reset();
        void set_device_peak_flops(double fp64_gflops, double fp32_gflops, double fp16_gflops);

    private:
        std::vector<double> achieved_samples_;
        double device_fp64_peak_{0.0};
        double device_fp32_peak_{0.0};
        double device_fp16_peak_{0.0};
    };

    class BandwidthAggregator {
    public:
        enum class TransferType { HostToDevice, DeviceToHost, DeviceToDevice };

        void add_sample(size_t bytes, uint64_t elapsed_ns, TransferType type);

        [[nodiscard]] double get_h2d_bandwidth_gbs() const;
        [[nodiscard]] double get_d2h_bandwidth_gbs() const;
        [[nodiscard]] double get_d2d_bandwidth_gbs() const;
        [[nodiscard]] double get_total_bandwidth_gbs() const;

        [[nodiscard]] double get_h2d_utilization_percent() const;
        [[nodiscard]] double get_d2h_utilization_percent() const;
        [[nodiscard]] double get_d2d_utilization_percent() const;

        void set_peak_bandwidths(double h2d_gbs, double d2h_gbs, double d2d_gbs);
        [[nodiscard]] AggregatedMetric get_summary(TransferType type) const;

        void reset();

    private:
        std::vector<double> h2d_samples_;
        std::vector<double> d2h_samples_;
        std::vector<double> d2d_samples_;
        double peak_h2d_{0.0};
        double peak_d2h_{0.0};
        double peak_d2d_{90.0 * 1024.0 * 1024.0 * 1024.0 / 1e9};
    };

    class MetricAggregatorPipeline {
    public:
        void add_arithmetic_sample(uint64_t flops, size_t bytes);
        void add_flops_sample(uint64_t flops, uint64_t elapsed_ns);
        void add_bandwidth_sample(size_t bytes, uint64_t elapsed_ns, BandwidthAggregator::TransferType type);

        [[nodiscard]] std::vector<AggregatedMetric> get_all_summaries() const;

        void reset();
        void set_device_peak_specs(double fp64_gflops, double fp32_gflops, double fp16_gflops, double h2d_gbs, double d2h_gbs, double d2d_gbs);

    private:
        ArithmeticIntensityAggregator ai_agg_;
        FLOPsAggregator flops_agg_;
        BandwidthAggregator bw_agg_;
    };

    double compute_mean(const std::vector<double>& samples);
    double compute_stddev(const std::vector<double>& samples);

}  // namespace cuda::performance
