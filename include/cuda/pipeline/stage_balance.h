#pragma once

/**
 * @file stage_balance.h
 * @brief Pipeline stage balance validation
 *
 * Profiles compute time per pipeline stage and validates
 * balance to minimize pipeline bubbles.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuda::pipeline {

    /**
     * @class StageBalanceValidator
     * @brief Validates pipeline stage compute balance
     *
     * Profiles compute times and reports variance across stages.
     * Stages with >10% variance indicate imbalance.
     *
     * @example
     * @code
     * StageBalanceValidator validator;
     * validator.profile_stage(0, 10.5f);  // Stage 0 took 10.5ms
     * validator.profile_stage(1, 12.1f);  // Stage 1 took 12.1ms
     * validator.profile_stage(2, 11.0f);  // Stage 2 took 11.0ms
     *
     * if (!validator.is_balanced()) {
     *     auto suggestions = validator.suggest_rebalance();
     * }
     * @endcode
     */
    class StageBalanceValidator {
    public:
        /**
         * @brief Construct validator
         */
        StageBalanceValidator();

        /**
         * @brief Profile a single stage
         * @param stage Stage index
         * @param compute_time_ms Compute time in milliseconds
         */
        void profile_stage(int stage, float compute_time_ms);

        /**
         * @brief Get variance as percentage
         * @return Coefficient of variation (stddev / mean * 100)
         */
        [[nodiscard]] float variance_percent() const;

        /**
         * @brief Check if stages are balanced (< 10% variance)
         * @return true if balanced
         */
        [[nodiscard]] bool is_balanced() const;

        /**
         * @brief Get average compute time
         * @return Mean time in milliseconds
         */
        [[nodiscard]] float average_time_ms() const;

        /**
         * @brief Get slowest stage
         * @return Stage index with maximum time
         */
        [[nodiscard]] int slowest_stage() const;

        /**
         * @brief Get fastest stage
         * @return Stage index with minimum time
         */
        [[nodiscard]] int fastest_stage() const;

        /**
         * @brief Suggest stages to merge or split for balance
         * @return Vector of stage indices to merge
         */
        [[nodiscard]] std::vector<int> suggest_rebalance() const;

        /**
         * @brief Reset all profiles
         */
        void reset();

        /**
         * @brief Get number of profiled stages
         * @return Stage count
         */
        [[nodiscard]] size_t stage_count() const;

        /**
         * @brief Get compute time for a stage
         * @param stage Stage index
         * @return Compute time in ms (0 if not profiled)
         */
        [[nodiscard]] float get_time(int stage) const;

        static constexpr float BALANCE_THRESHOLD_PERCENT = 10.0f;

    private:
        std::vector<float> compute_times_;
    };

}  // namespace cuda::pipeline
