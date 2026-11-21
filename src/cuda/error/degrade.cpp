#include "cuda/error/degrade.hpp"
#include <cstring>
#include <unordered_map>

namespace nova::error {

    degradation_manager& degradation_manager::instance() {
        static degradation_manager instance;
        return instance;
    }

    void degradation_manager::set_callback(degradation_callback cb) {
        callback_ = std::move(cb);
    }
    void degradation_manager::reset() {
        callback_ = nullptr;
        threshold_ = quality_threshold {};
        precision_by_op_.clear();
        quality_by_op_.clear();
    }

    void degradation_manager::set_threshold(quality_threshold threshold) {
        threshold_ = threshold;
    }

    const quality_threshold& degradation_manager::get_threshold() const {
        return threshold_;
    }

    void degradation_manager::record_quality(std::string_view operation, double quality_score) {
        // Record the score and NOTHING else. This used to discard a passing score and, for a
        // failing one, only register the operation at high precision -- so the score itself
        // was never kept and should_degrade(), which the header's own example calls straight
        // after this, could not see it. Measured: a score of 0.0 and a score of 0.99 produced
        // identical answers from should_degrade().
        //
        // Registering at high precision on a failing score is also wrong on its own terms: it
        // claims the operation is at full precision when the caller has just measured that it
        // is not performing, and it would silently undo a level the caller had set through
        // trigger_degradation().
        quality_by_op_[std::string(operation)] = quality_score;
    }

    void degradation_manager::trigger_degradation(std::string_view operation, precision_level new_level, std::string_view reason) {
        auto key = std::string(operation);
        auto old_level = precision_by_op_[key];
        precision_by_op_[key] = new_level;

        if (callback_) {
            degradation_event event;
            event.operation = operation;
            event.from = old_level;
            event.to = new_level;
            event.timestamp = std::chrono::steady_clock::now();
            event.reason = reason;
            callback_(event);
        }
    }

    precision_level degradation_manager::get_precision(std::string_view operation) const {
        auto it = precision_by_op_.find(std::string(operation));
        if (it != precision_by_op_.end()) {
            return it->second;
        }
        return precision_level::high;
    }

    bool degradation_manager::should_degrade(std::string_view operation) const {
        const precision_level precision = get_precision(operation);

        // Nothing is left to give up at the floor, whatever the scores say. This was the one
        // case the old `precision > min_acceptable` comparison got visibly wrong: precision
        // could not go any lower and the caller was told to degrade again.
        if (precision >= precision_level::low) {
            return false;
        }

        // A recorded score below the threshold is the degradation signal. The header documents
        // should_degrade() as "true if quality scores indicate degradation needed" and its
        // example calls record_quality() first, but the implementation only ever looked at the
        // precision level, so record_quality() had no observable effect at all.
        const auto score = quality_by_op_.find(std::string(operation));
        if (score != quality_by_op_.end() && score->second < threshold_.min_quality_score) {
            return true;
        }

        // Otherwise fall back to the precision comparison, which is what a caller that only
        // uses trigger_degradation() reports. The enum counts DOWN in precision -- high = 0,
        // medium = 1, low = 2 -- so this reads "the current level is weaker than the minimum
        // acceptable one".
        return precision > threshold_.min_acceptable_precision;
    }

}  // namespace nova::error
