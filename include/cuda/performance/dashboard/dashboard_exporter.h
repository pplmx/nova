#pragma once

#include <cuda/performance/bandwidth/roofline_model.h>
#include <cuda/performance/fusion/fusion_profitability.h>

#include <string>
#include <vector>
#include <chrono>

namespace cuda::performance::dashboard {

struct DashboardConfig {
    bool include_roofline{true};
    bool include_fusion{true};
    bool include_bandwidth{true};
    bool include_kernel_metrics{true};
    std::string output_format{"json"};
};

struct DashboardData {
    struct Header {
        std::string title{"Nova Performance Dashboard"};
        std::string version;
        std::string timestamp;
        std::string gpu_name;
        int cuda_version{0};
    } header;

    struct RooflineSection {
        double peak_fp64_gflops{0.0};
        double peak_fp32_gflops{0.0};
        double peak_bandwidth_gbs{0.0};
        double ridge_point{0.0};
        size_t point_count{0};
        size_t memory_bound_count{0};
        size_t compute_bound_count{0};
    } roofline;

    struct FusionSection {
        size_t opportunity_count{0};
        size_t high_confidence_count{0};
        size_t medium_confidence_count{0};
        size_t low_confidence_count{0};
        double total_latency_saved_us{0.0};
        double best_speedup{0.0};
    } fusion;

    struct BandwidthSection {
        double average_bandwidth_gbs{0.0};
        double peak_bandwidth_gbs{0.0};
        double utilization_percent{0.0};
        size_t sample_count{0};
        bool has_low_utilization_warning{false};
    } bandwidth;

    struct KernelSection {
        size_t kernel_count{0};
        double total_latency_us{0.0};
        double average_occupancy{0.0};
        double peak_throughput_gflops{0.0};
    } kernels;
};

class DashboardExporter {
public:
    explicit DashboardExporter(const DashboardConfig& config = {});

    void set_config(const DashboardConfig& config);
    [[nodiscard]] const DashboardConfig& get_config() const;

    void add_roofline_data(const bandwidth::RooflineModel& model);
    void add_fusion_data(const std::vector<fusion::FusionRecommendation>& recommendations);
    void add_bandwidth_data(const bandwidth::BandwidthUtilizationTracker& tracker);
    void add_kernel_count(size_t count);

    void clear();
    [[nodiscard]] bool is_empty() const;

    [[nodiscard]] std::string to_json() const;
    [[nodiscard]] std::string to_csv() const;

    [[nodiscard]] DashboardData get_data() const;

private:
    void populate_header(DashboardData& data) const;
    void populate_roofline(DashboardData& data) const;
    void populate_fusion(DashboardData& data) const;
    void populate_bandwidth(DashboardData& data) const;
    void populate_kernels(DashboardData& data) const;

    DashboardConfig config_;
    DashboardData data_;

    std::vector<bandwidth::RooflineModel> roofline_models_;
    std::vector<fusion::FusionRecommendation> fusion_recommendations_;
    std::vector<bandwidth::BandwidthUtilizationTracker> bandwidth_trackers_;
};

class DashboardGenerator {
public:
    DashboardGenerator();

    void add_exporter(const DashboardExporter& exporter);
    void clear();

    [[nodiscard]] std::string generate_html() const;
    [[nodiscard]] std::string generate_json() const;

    /// Not implemented in C++: generate_html() returns a placeholder pointing at
    /// scripts/benchmark/generate_performance_dashboard.py, and this function writes
    /// nothing. It returns false to say so. It previously returned true, which made a
    /// no-op indistinguishable from a successful export.
    [[nodiscard]] bool write_files(const std::string& output_dir) const;

private:
    std::vector<DashboardExporter> exporters_;
};

}  // namespace cuda::performance::dashboard
