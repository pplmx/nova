/**
 * @file stage_balance.cpp
 * @brief Stage balance validator implementation
 */

#include "cuda/pipeline/stage_balance.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace cuda::pipeline {

    StageBalanceValidator::StageBalanceValidator() = default;

    void StageBalanceValidator::profile_stage(int stage, float compute_time_ms) {
        // A negative index passed the `stage >= size()` test -- no resize happened -- and the
        // assignment then wrote before the vector's first element. Measured under ASan:
        // heap-buffer-overflow WRITE (4 bytes before a 12-byte region) inside this function.
        if (stage < 0) {
            throw std::invalid_argument("StageBalanceValidator::profile_stage: stage index must not be negative, got " + std::to_string(stage));
        }
        // A NaN or infinite time made variance_percent() return NaN, and `NaN < 10.0f` is
        // false, so is_balanced() reported "unbalanced" as if it had measured something. A
        // negative time is not a duration either.
        if (!std::isfinite(compute_time_ms)) {
            throw std::invalid_argument("StageBalanceValidator::profile_stage: compute_time_ms must be finite, got " + std::to_string(compute_time_ms));
        }
        if (compute_time_ms < 0.0f) {
            throw std::invalid_argument("StageBalanceValidator::profile_stage: compute_time_ms must not be negative, got " + std::to_string(compute_time_ms));
        }

        const size_t index = static_cast<size_t>(stage);
        if (index >= compute_times_.size()) {
            compute_times_.resize(index + 1, 0.0f);
            profiled_.resize(index + 1, false);
        }
        if (!profiled_[index]) {
            profiled_[index] = true;
            ++profiled_count_;
        }
        compute_times_[index] = compute_time_ms;
    }

    float StageBalanceValidator::variance_percent() const {
        // The statistics run over PROFILED stages only. An unprofiled gap in compute_times_ is
        // not a zero-cost stage: averaging it in made two equal 10 ms stages, profiled as 0
        // and 5, report a mean of 3.33 ms, a variance of 141% and "unbalanced".
        if (profiled_count_ == 0) {
            return 0.0f;
        }

        const float mean = average_time_ms();
        if (mean < 1e-6f) {
            return 0.0f;
        }

        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < compute_times_.size(); ++i) {
            if (!profiled_[i]) {
                continue;
            }
            const float diff = compute_times_[i] - mean;
            sum_sq_diff += diff * diff;
        }

        const float stddev = std::sqrt(sum_sq_diff / static_cast<float>(profiled_count_));
        return (stddev / mean) * 100.0f;
    }

    bool StageBalanceValidator::is_balanced() const {
        return variance_percent() < BALANCE_THRESHOLD_PERCENT;
    }

    float StageBalanceValidator::average_time_ms() const {
        if (profiled_count_ == 0) {
            return 0.0f;
        }

        float sum = 0.0f;
        for (size_t i = 0; i < compute_times_.size(); ++i) {
            if (profiled_[i]) {
                sum += compute_times_[i];
            }
        }
        return sum / static_cast<float>(profiled_count_);
    }

    int StageBalanceValidator::slowest_stage() const {
        // Skips unprofiled gaps for the same reason as the statistics: an unprofiled slot
        // holds 0.0f and would otherwise be reported as the fastest stage of all.
        int best = -1;
        float best_time = 0.0f;
        for (size_t i = 0; i < compute_times_.size(); ++i) {
            if (!profiled_[i]) {
                continue;
            }
            if (best < 0 || compute_times_[i] > best_time) {
                best = static_cast<int>(i);
                best_time = compute_times_[i];
            }
        }
        return best;
    }

    int StageBalanceValidator::fastest_stage() const {
        int best = -1;
        float best_time = 0.0f;
        for (size_t i = 0; i < compute_times_.size(); ++i) {
            if (!profiled_[i]) {
                continue;
            }
            if (best < 0 || compute_times_[i] < best_time) {
                best = static_cast<int>(i);
                best_time = compute_times_[i];
            }
        }
        return best;
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
        profiled_.clear();
        profiled_count_ = 0;
    }

    size_t StageBalanceValidator::stage_count() const {
        return compute_times_.size();
    }

    float StageBalanceValidator::get_time(int stage) const {
        if (stage >= 0 && stage < static_cast<int>(compute_times_.size()) && profiled_[static_cast<size_t>(stage)]) {
            return compute_times_[stage];
        }
        return 0.0f;
    }

}  // namespace cuda::pipeline
