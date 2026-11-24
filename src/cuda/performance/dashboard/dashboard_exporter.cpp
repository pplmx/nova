#include <cuda/performance/dashboard/dashboard_exporter.h>

#include <sstream>
#include <iomanip>
#include <ctime>

namespace cuda::performance::dashboard {

    DashboardExporter::DashboardExporter(const DashboardConfig& config)
        : config_(config) {}

    void DashboardExporter::set_config(const DashboardConfig& config) {
        config_ = config;
    }

    const DashboardConfig& DashboardExporter::get_config() const {
        return config_;
    }

    void DashboardExporter::add_roofline_data(const bandwidth::RooflineModel& model) {
        roofline_models_.push_back(model);
    }

    void DashboardExporter::add_fusion_data(const std::vector<fusion::FusionRecommendation>& recommendations) {
        fusion_recommendations_ = recommendations;
    }

    void DashboardExporter::add_bandwidth_data(const bandwidth::BandwidthUtilizationTracker& tracker) {
        bandwidth_trackers_.push_back(tracker);
    }

    void DashboardExporter::add_kernel_count(size_t count) {
        data_.kernels.kernel_count = count;
    }

    void DashboardExporter::clear() {
        roofline_models_.clear();
        fusion_recommendations_.clear();
        bandwidth_trackers_.clear();
        data_ = DashboardData {};
    }

    bool DashboardExporter::is_empty() const {
        return roofline_models_.empty() && fusion_recommendations_.empty() && bandwidth_trackers_.empty() && data_.kernels.kernel_count == 0;
    }

    void DashboardExporter::populate_header(DashboardData& data) const {
        data.header.version = "2.11";
        std::time_t now = std::time(nullptr);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        data.header.timestamp = buf;
    }

    void DashboardExporter::populate_roofline(DashboardData& data) const {
        for (const auto& model : roofline_models_) {
            const auto& peaks = model.peaks();
            data.roofline.peak_fp64_gflops = std::max(data.roofline.peak_fp64_gflops, peaks.fp64_gflops);
            data.roofline.peak_fp32_gflops = std::max(data.roofline.peak_fp32_gflops, peaks.fp32_gflops);
            data.roofline.peak_bandwidth_gbs = std::max(data.roofline.peak_bandwidth_gbs, peaks.hbm_bandwidth_gbs);
            data.roofline.ridge_point = model.ridge_point();
            data.roofline.point_count += model.point_count();
            data.roofline.memory_bound_count += model.get_memory_bound_points().size();
            data.roofline.compute_bound_count += model.get_compute_bound_points().size();
        }
    }

    void DashboardExporter::populate_fusion(DashboardData& data) const {
        data.fusion.opportunity_count = fusion_recommendations_.size();
        for (const auto& rec : fusion_recommendations_) {
            switch (rec.confidence()) {
                case fusion::ConfidenceLevel::HIGH:
                    data.fusion.high_confidence_count++;
                    break;
                case fusion::ConfidenceLevel::MEDIUM:
                    data.fusion.medium_confidence_count++;
                    break;
                case fusion::ConfidenceLevel::LOW:
                    data.fusion.low_confidence_count++;
                    break;
            }
            data.fusion.total_latency_saved_us += static_cast<double>(rec.latency_saved_us());
            data.fusion.best_speedup = std::max(data.fusion.best_speedup, rec.speedup_factor());
        }
    }

    void DashboardExporter::populate_bandwidth(DashboardData& data) const {
        for (const auto& tracker : bandwidth_trackers_) {
            data.bandwidth.average_bandwidth_gbs = std::max(data.bandwidth.average_bandwidth_gbs, tracker.average_bandwidth_gbs());
            data.bandwidth.peak_bandwidth_gbs = std::max(data.bandwidth.peak_bandwidth_gbs, tracker.peak_bandwidth_gbs());
            data.bandwidth.utilization_percent = std::max(data.bandwidth.utilization_percent, tracker.utilization_percent());
            data.bandwidth.sample_count += tracker.sample_count();
            data.bandwidth.has_low_utilization_warning |= tracker.has_low_utilization_warning();
        }
    }

    void DashboardExporter::populate_kernels(DashboardData& data) const {
        // add_kernel_count() writes straight into data_.kernels, and get_data() used to build its
        // result from a fresh DashboardData that never incorporated data_, so the count the caller
        // supplied reached nothing. Measured: add_kernel_count(12345) made is_empty() report false
        // -- the value was stored -- while get_data().kernels.kernel_count stayed 0 and to_json()
        // emitted "count": 0.
        //
        // There is no per-kernel data structure to aggregate here (the other three fields of
        // KernelSection have no source at all), so this only has to not discard what was recorded.
        // Keeping the max rather than assigning makes it safe whichever order the sections are
        // filled in.
        data.kernels.kernel_count = std::max(data.kernels.kernel_count, data_.kernels.kernel_count);
    }

    DashboardData DashboardExporter::get_data() const {
        // Start from the accumulators the add_* accessors write into, then let the populate_*
        // passes fill in the sections computed from the registered models and trackers.
        DashboardData d = data_;
        populate_header(d);
        populate_roofline(d);
        populate_fusion(d);
        populate_bandwidth(d);
        populate_kernels(d);
        return d;
    }

    std::string DashboardExporter::to_json() const {
        auto data = get_data();

        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"header\": {\n";
        oss << "    \"title\": \"" << data.header.title << "\",\n";
        oss << "    \"version\": \"" << data.header.version << "\",\n";
        oss << "    \"timestamp\": \"" << data.header.timestamp << "\",\n";
        oss << "    \"gpu\": \"" << data.header.gpu_name << "\"\n";
        oss << "  },\n";

        oss << "  \"roofline\": {\n";
        oss << "    \"peak_fp64_gflops\": " << std::fixed << std::setprecision(1) << data.roofline.peak_fp64_gflops << ",\n";
        oss << "    \"peak_bandwidth_gbs\": " << data.roofline.peak_bandwidth_gbs << ",\n";
        oss << "    \"ridge_point\": " << std::setprecision(2) << data.roofline.ridge_point << ",\n";
        oss << "    \"points\": " << data.roofline.point_count << ",\n";
        oss << "    \"memory_bound\": " << data.roofline.memory_bound_count << ",\n";
        oss << "    \"compute_bound\": " << data.roofline.compute_bound_count << "\n";
        oss << "  },\n";

        oss << "  \"fusion\": {\n";
        oss << "    \"opportunities\": " << data.fusion.opportunity_count << ",\n";
        oss << "    \"high_confidence\": " << data.fusion.high_confidence_count << ",\n";
        oss << "    \"medium_confidence\": " << data.fusion.medium_confidence_count << ",\n";
        oss << "    \"low_confidence\": " << data.fusion.low_confidence_count << ",\n";
        oss << "    \"total_latency_saved_us\": " << std::setprecision(0) << data.fusion.total_latency_saved_us << ",\n";
        oss << "    \"best_speedup\": " << std::setprecision(2) << data.fusion.best_speedup << "\n";
        oss << "  },\n";

        oss << "  \"bandwidth\": {\n";
        oss << "    \"average_gbs\": " << std::setprecision(2) << data.bandwidth.average_bandwidth_gbs << ",\n";
        oss << "    \"peak_gbs\": " << data.bandwidth.peak_bandwidth_gbs << ",\n";
        oss << "    \"utilization_percent\": " << data.bandwidth.utilization_percent << ",\n";
        oss << "    \"samples\": " << data.bandwidth.sample_count << ",\n";
        oss << "    \"low_utilization_warning\": " << (data.bandwidth.has_low_utilization_warning ? "true" : "false") << "\n";
        oss << "  },\n";

        oss << "  \"kernels\": {\n";
        oss << "    \"count\": " << data.kernels.kernel_count << "\n";
        oss << "  }\n";
        oss << "}\n";

        return oss.str();
    }

    std::string DashboardExporter::to_csv() const {
        std::ostringstream oss;
        oss << "section,metric,value\n";

        auto data = get_data();

        oss << "roofline,peak_fp64_gflops," << std::fixed << std::setprecision(1) << data.roofline.peak_fp64_gflops << "\n";
        oss << "roofline,peak_bandwidth_gbs," << data.roofline.peak_bandwidth_gbs << "\n";
        oss << "roofline,ridge_point," << std::setprecision(2) << data.roofline.ridge_point << "\n";
        oss << "roofline,points," << data.roofline.point_count << "\n";

        oss << "fusion,opportunities," << data.fusion.opportunity_count << "\n";
        oss << "fusion,total_latency_saved_us," << std::setprecision(0) << data.fusion.total_latency_saved_us << "\n";

        oss << "bandwidth,average_gbs," << std::setprecision(2) << data.bandwidth.average_bandwidth_gbs << "\n";
        oss << "bandwidth,utilization_percent," << data.bandwidth.utilization_percent << "\n";

        return oss.str();
    }

    DashboardGenerator::DashboardGenerator() {}

    void DashboardGenerator::add_exporter(const DashboardExporter& exporter) {
        exporters_.push_back(exporter);
    }

    void DashboardGenerator::clear() {
        exporters_.clear();
    }

    std::string DashboardGenerator::generate_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"dashboards\": [\n";

        for (size_t i = 0; i < exporters_.size(); ++i) {
            if (i > 0) {
                oss << ",\n";
            }
            oss << "    " << exporters_[i].to_json();
        }

        oss << "\n  ]\n";
        oss << "}\n";

        return oss.str();
    }

    std::string DashboardGenerator::generate_html() const {
        return "# Extended dashboard generation not implemented in C++\n"
               "# Use Python script: scripts/benchmark/generate_performance_dashboard.py";
    }

    bool DashboardGenerator::write_files(const std::string& output_dir) const {
        // This used to return true without touching the filesystem, and generate_html
        // returns a placeholder saying the C++ implementation does not exist. A caller that
        // checked the return value concluded a dashboard had been written; measured, the
        // output directory was never even created.
        //
        // Reporting false is the honest answer and is what a caller can act on: either use
        // the Python generator the placeholder names, or treat the C++ path as unavailable.
        // Returning true here would keep a silent no-op indistinguishable from success.
        (void)output_dir;
        return false;
    }

}  // namespace cuda::performance::dashboard
