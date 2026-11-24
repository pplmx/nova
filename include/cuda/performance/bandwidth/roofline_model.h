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

    /// Which link a sample measures. The three directions have their own hardware peak, so a
    /// sample is only comparable against the peak of the link it came from.
    enum class TransferDirection {
        H2D,  ///< host to device, bounded by the interconnect (PCIe/NVLink)
        D2H,  ///< device to host, bounded by the same interconnect
        D2D,  ///< device to device, bounded by HBM
    };

    struct BandwidthSample {
        double bandwidth_gbs {0.0};
        size_t bytes {0};
        uint64_t elapsed_ns {0};
        TransferDirection direction {TransferDirection::D2D};
    };

    class BandwidthUtilizationTracker {
    public:
        BandwidthUtilizationTracker();

        /// Adds a sample whose direction the caller does not know or does not care about.
        ///
        /// Recorded as device-to-device, the widest of the three peaks. Prefer the three explicit
        /// add_*_sample overloads when the direction is known.
        void add_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);
        void add_h2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);
        void add_d2h_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);
        void add_d2d_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);

        void reset();

        /// Mean of every recorded sample, regardless of direction.
        [[nodiscard]] double average_bandwidth_gbs() const;
        /// Largest bandwidth among every recorded sample, regardless of direction.
        [[nodiscard]] double peak_bandwidth_gbs() const;
        /// Mean against the device-to-device peak.
        ///
        /// This is an average over ALL samples divided by one direction's peak. When samples of
        /// different directions are mixed the ratio is not a utilization of anything -- measured
        /// 1150% for a pair of samples before the per-direction accessors were separated. Use the
        /// matching *_utilization_percent() below for a ratio that means something.
        [[nodiscard]] double utilization_percent() const;

        /// Utilization of the host-to-device link: the mean of the H2D samples over the H2D peak.
        ///
        /// Returns 0 when no H2D sample has been recorded or when the peak is not positive. These
        /// three accessors used to divide the mean of ALL samples by their own peak, so an H2D
        /// accessor reported a D2D sample's bandwidth against the host link's peak.
        [[nodiscard]] double h2d_utilization_percent() const;
        [[nodiscard]] double d2h_utilization_percent() const;
        [[nodiscard]] double d2d_utilization_percent() const;

        /// Sets the hardware peak of each link, in GB/s, for the machine being measured.
        ///
        /// The defaults below are placeholders for a generic accelerator, not this machine's
        /// numbers; call this before reading any utilization, or the ratio is against fiction.
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
