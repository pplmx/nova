/**
 * @file stage_balance.cpp
 * @brief Stage balance validator implementation
 */

#include "cuda/pipeline/stage_balance.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cuda::pipeline {

    StageBalanceValidator::StageBalanceValidator() = default;

    void StageBalanceValidator::profile_stage(int stage, float compute_time_ms) {
        if (stage >= static_cast<int>(compute_times_.size())) {
            compute_times_.resize(stage + 1, 0.0f);
        }
        compute_times_[stage] = compute_time_ms;
    }

    float StageBalanceValidator::variance_percent() const {
        if (compute_times_.empty()) {
            return 0.0f;
        }

        float mean = average_time_ms();
        if (mean < 1e-6f) {
            return 0.0f;
        }

        float sum_sq_diff = 0.0f;
        for (float time : compute_times_) {
            float diff = time - mean;
            sum_sq_diff += diff * diff;
        }

        float stddev = std::sqrt(sum_sq_diff / compute_times_.size());
        return (stddev / mean) * 100.0f;
    }

    bool StageBalanceValidator::is_balanced() const {
        return variance_percent() < BALANCE_THRESHOLD_PERCENT;
    }

    float StageBalanceValidator::average_time_ms() const {
        if (compute_times_.empty()) {
            return 0.0f;
        }

        float sum = std::accumulate(compute_times_.begin(), compute_times_.end(), 0.0f);
        return sum / compute_times_.size();
    }

    int StageBalanceValidator::slowest_stage() const {
        if (compute_times_.empty()) {
            return -1;
        }

        return static_cast<int>(std::max_element(compute_times_.begin(), compute_times_.end()) - compute_times_.begin());
    }

    int StageBalanceValidator::fastest_stage() const {
        if (compute_times_.empty()) {
            return -1;
        }

        return static_cast<int>(std::min_element(compute_times_.begin(), compute_times_.end()) - compute_times_.begin());
    }

    std::vector<int> StageBalanceValidator::suggest_rebalance() const {
        std::vector<int> suggestions;

        if (!is_balanced()) {
            suggestions.push_back(slowest_stage());
            suggestions.push_back(fastest_stage());
        }

        return suggestions;
    }

    void StageBalanceValidator::reset() {
        compute_times_.clear();
    }

    size_t StageBalanceValidator::stage_count() const {
        return compute_times_.size();
    }

    float StageBalanceValidator::get_time(int stage) const {
        if (stage >= 0 && stage < static_cast<int>(compute_times_.size())) {
            return compute_times_[stage];
        }
        return 0.0f;
    }

}  // namespace cuda::pipeline
