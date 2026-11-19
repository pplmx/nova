#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <memory>

#include "roofline_model.h"

namespace cuda::performance::bandwidth {

    struct CacheMetrics {
        double l1_hit_rate {0.0};
        double l2_hit_rate {0.0};
        double texture_hit_rate {0.0};
        bool available {false};
        std::string error_message;
    };

    class CacheAnalyzer {
    public:
        CacheAnalyzer();

        [[nodiscard]] CacheMetrics analyze(int device_id = 0);
        [[nodiscard]] CacheMetrics analyze_kernel(const std::string& kernel_name, int device_id = 0);

        [[nodiscard]] bool is_available() const;
        [[nodiscard]] std::string get_error_message() const;

        static bool check_cupti_availability();

    private:
        bool available_ {false};
        std::string error_message_;
    };

    class BandwidthAnalysis {
    public:
        BandwidthAnalysis();
        ~BandwidthAnalysis();

        void set_roofline_peak_flops(double fp64, double fp32, double fp16);
        void set_roofline_peak_bandwidth(double hbm_gbs);

        void add_kernel_sample(const std::string& name, uint64_t flops, size_t bytes, uint64_t elapsed_ns);

        void add_bandwidth_sample(double bandwidth_gbs, size_t bytes, uint64_t elapsed_ns);

        [[nodiscard]] std::string generate_roofline_json() const;
        [[nodiscard]] std::string generate_bandwidth_json() const;

        [[nodiscard]] std::string get_report() const;

    private:
        double peak_fp64_ {0.0};
        double peak_fp32_ {0.0};
        double peak_fp16_ {0.0};
        double peak_bandwidth_ {0.0};

        std::unique_ptr<RooflineModel> roofline_;
        std::unique_ptr<BandwidthUtilizationTracker> tracker_;
    };

}  // namespace cuda::performance::bandwidth
