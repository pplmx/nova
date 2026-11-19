#pragma once

#include <cuda/performance/fusion/kernel_fusion_analyzer.h>

#include <cstdint>
#include <string>

namespace cuda::performance::fusion {

    enum class ConfidenceLevel { HIGH, MEDIUM, LOW };

    struct FusionProfitabilityConfig {
        double launch_overhead_us {100.0};
        double memory_coalescing_benefit_multiplier {1.2};
        double register_pressure_cost_multiplier {0.8};
        double min_profitability_threshold {0.1};
    };

    class FusionProfitabilityModel {
    public:
        explicit FusionProfitabilityModel(const FusionProfitabilityConfig& config = {});

        void set_config(const FusionProfitabilityConfig& config);
        [[nodiscard]] const FusionProfitabilityConfig& get_config() const;

        [[nodiscard]] bool is_profitable(const FusionOpportunity& opp) const;
        [[nodiscard]] double profitability_score(const FusionOpportunity& opp) const;
        [[nodiscard]] uint64_t estimated_launch_overhead_saved_us() const;
        [[nodiscard]] uint64_t estimated_memory_benefit_us() const;
        [[nodiscard]] uint64_t estimated_register_cost_us() const;

    private:
        FusionProfitabilityConfig config_;
    };

    class FusionRecommendation {
    public:
        FusionRecommendation() = default;
        explicit FusionRecommendation(const FusionOpportunity& opp, ConfidenceLevel confidence);

        [[nodiscard]] const FusionOpportunity& opportunity() const { return opportunity_; }
        [[nodiscard]] ConfidenceLevel confidence() const { return confidence_; }
        [[nodiscard]] double profitability_score() const { return profitability_score_; }

        [[nodiscard]] uint64_t before_latency_us() const { return before_latency_us_; }
        [[nodiscard]] uint64_t after_latency_us() const { return after_latency_us_; }
        [[nodiscard]] uint64_t latency_saved_us() const { return latency_saved_us_; }
        [[nodiscard]] double speedup_factor() const { return speedup_factor_; }

        [[nodiscard]] std::string pattern_name() const { return pattern_name_; }
        [[nodiscard]] std::string description() const { return description_; }
        [[nodiscard]] std::string suggestion() const { return suggestion_; }

        void set_suggestion(const std::string& s) { suggestion_ = s; }

        [[nodiscard]] std::string to_json() const;

    private:
        FusionOpportunity opportunity_;
        ConfidenceLevel confidence_ {ConfidenceLevel::LOW};
        double profitability_score_ {0.0};

        uint64_t before_latency_us_ {0};
        uint64_t after_latency_us_ {0};
        uint64_t latency_saved_us_ {0};
        double speedup_factor_ {1.0};

        std::string pattern_name_;
        std::string description_;
        std::string suggestion_;
    };

    class FusionRecommendationEngine {
    public:
        explicit FusionRecommendationEngine(const FusionProfitabilityConfig& config = {});

        void set_config(const FusionProfitabilityConfig& config);
        [[nodiscard]] const FusionProfitabilityConfig& get_config() const;

        [[nodiscard]] std::vector<FusionRecommendation> generate_recommendations(const std::vector<FusionOpportunity>& opportunities);

        [[nodiscard]] std::vector<FusionRecommendation> generate_recommendations(const std::vector<FusionOpportunity>& opportunities, double min_profitability_score);

        void add_custom_suggestion(const std::string& pattern_name, const std::string& suggestion);

    private:
        [[nodiscard]] ConfidenceLevel determine_confidence(const FusionOpportunity& opp) const;
        [[nodiscard]] std::string generate_suggestion(const FusionOpportunity& opp, ConfidenceLevel conf) const;

        FusionProfitabilityModel profitability_model_;
        std::unordered_map<std::string, std::string> custom_suggestions_;
    };

    ConfidenceLevel higher_confidence(ConfidenceLevel a, ConfidenceLevel b);
    std::string to_string(ConfidenceLevel level);
    std::string to_string(const FusionRecommendation& rec);

}  // namespace cuda::performance::fusion
