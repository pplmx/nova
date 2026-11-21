#pragma once

/**
 * @file stage_balance.h
 * @brief Pipeline stage balance validation
 *
 * Profiles compute time per pipeline stage and validates
 * balance to minimize pipeline bubbles.
 */

#include <cuda_runtime.h>

#include <vector>

namespace cuda::pipeline {

    /**
     * @class StageBalanceValidator
     * @brief Validates pipeline stage compute balance
     *
     * Profiles compute times and reports variance across stages.
     * Stages with >10% variance indicate imbalance.
     *
     * @par Only profiled stages take part in the statistics
     * profile_stage(3, t) with no earlier calls leaves stages 0..2 unprofiled. Those gaps are
     * not zero-cost stages and must not be averaged in: treating them as 0 ms made two
     * genuinely equal 10 ms stages, profiled as 0 and 5, report a mean of 3.33 ms, a variance
     * of 141% and "unbalanced" (measured). A stage index is a label, not a slot, so the mean
     * and variance are computed over the entries that were actually profiled.
     *
     * @par Times must be finite and non-negative
     * profile_stage() rejects a negative stage index, a negative time and a non-finite (NaN or
     * inf) time with std::invalid_argument. The index check is a memory-safety fix: a negative
     * stage passed the `stage >= size()` test, so no resize happened and the assignment wrote
     * before the vector's start -- measured as a heap-buffer-overflow WRITE. The value checks
     * exist because a NaN time made variance_percent() return NaN, and NaN < threshold is
     * false, so the validator reported a confident "unbalanced" from an input it never
     * validated.
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
         *
         * Profiling the same stage again replaces the previous time for it.
         *
         * @param stage Stage index; must not be negative
         * @param compute_time_ms Compute time in milliseconds; must be finite and non-negative
         * @throws std::invalid_argument if the index is negative, or the time is negative,
         *         NaN or infinite
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
         * @brief First stage index past the highest one profiled
         *
         * This is the size of the index range, not the number of profiled entries: profiling
         * stages 0 and 5 gives 6. Use profiled_stage_count() for the other question.
         * @return One past the highest profiled index, or 0 if nothing has been profiled
         */
        [[nodiscard]] size_t stage_count() const;

        /**
         * @brief Number of stages actually profiled
         *
         * Unlike stage_count() this ignores gaps, so profiling stages 0 and 5 gives 2. The
         * balance statistics are computed over these entries only.
         */
        [[nodiscard]] size_t profiled_stage_count() const { return profiled_count_; }

        /**
         * @brief Get compute time for a stage
         * @param stage Stage index
         * @return Compute time in ms (0 if not profiled)
         */
        [[nodiscard]] float get_time(int stage) const;

        static constexpr float BALANCE_THRESHOLD_PERCENT = 10.0f;

    private:
        /// Time per stage index, as profiled. A slot is meaningful only where profiled_ is true.
        std::vector<float> compute_times_;
        /// Which stage indices were actually profiled. A gap in compute_times_ is not a
        /// zero-cost stage, so it must not enter the mean or the variance.
        std::vector<bool> profiled_;
        /// Number of true entries in profiled_.
        size_t profiled_count_ = 0;
    };

}  // namespace cuda::pipeline
