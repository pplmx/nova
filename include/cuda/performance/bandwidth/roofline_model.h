#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cuda::performance::bandwidth {

    struct RooflinePoint {
        double arithmetic_intensity {0.0};
        double performance_gflops {0.0};
        std::string kernel_name;
        uint64_t flops {0};
        size_t bytes_accessed {0};
        uint64_t latency_ns {0};
    };

    struct DevicePeaks {
        double fp64_gflops {0.0};
        double fp32_gflops {0.0};
        double fp16_gflops {0.0};
        double hbm_bandwidth_gbs {0.0};

        static DevicePeaks query(int device_id = 0);
    };

    class RooflineModel {
    public:
        RooflineModel();

        void add_point(const RooflinePoint& point);
        void add_point(const std::string& name, uint64_t flops, size_t bytes, uint64_t elapsed_ns);

        void clear();
        [[nodiscard]] size_t point_count() const;

        [[nodiscard]] double compute_arithmetic_intensity(uint64_t flops, size_t bytes) const;
        [[nodiscard]] double compute_performance_gflops(uint64_t flops, uint64_t elapsed_ns) const;

        /// Override the peaks query() found at construction.
        ///
        /// The constructor calls DevicePeaks::query(), which reports what the driver
        /// says the device can do. A caller that measures its own hardware, or targets
        /// a device other than the one the process is bound to, needs to say so; the
        /// arithmetic-intensity classification and the ridge point are computed from
        /// these values, so replacing them changes the answers.
        void set_peaks(const DevicePeaks& peaks) { peaks_ = peaks; }

        [[nodiscard]] const DevicePeaks& peaks() const { return peaks_; }
        [[nodiscard]] double ridge_point() const;

        [[nodiscard]] std::vector<RooflinePoint> get_points() const;
        [[nodiscard]] std::vector<RooflinePoint> get_memory_bound_points() const;
        [[nodiscard]] std::vector<RooflinePoint> get_compute_bound_points() const;

        [[nodiscard]] std::string to_json() const;
        [[nodiscard]] std::string to_csv() const;

        enum class BoundType { Memory, Compute, Unknown };
        [[nodiscard]] BoundType classify_point(double arithmetic_intensity) const;

    private:
        std::vector<RooflinePoint> points_;
        DevicePeaks peaks_;
    };

    struct BandwidthSample {
        double bandwidth_gbs {0.0};
        size_t bytes {0};
        uint64_t elapsed_ns {0};
        bool is_memory_bound {false};
    };

    class BandwidthUtilizationTracker {
    public:
        BandwidthUtilizationTracker();

        void add_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);
        void add_h2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);
        void add_d2h_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);
        void add_d2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);

        void reset();

        [[nodiscard]] double average_bandwidth_gbs() const;
        [[nodiscard]] double peak_bandwidth_gbs() const;
        [[nodiscard]] double utilization_percent() const;

        [[nodiscard]] double h2d_utilization_percent() const;
        [[nodiscard]] double d2h_utilization_percent() const;
        [[nodiscard]] double d2d_utilization_percent() const;

        void set_peak_bandwidth(double h2d, double d2h, double d2d);

        [[nodiscard]] size_t sample_count() const;
        [[nodiscard]] bool has_low_utilization_warning() const;
        [[nodiscard]] double get_min_utilization() const;

        [[nodiscard]] std::string to_json() const;

    private:
        std::vector<BandwidthSample> samples_;
        double peak_h2d_ {20.0};
        double peak_d2h_ {20.0};
        double peak_d2d_ {900.0};
    };

}  // namespace cuda::performance::bandwidth
