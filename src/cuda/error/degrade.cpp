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

void degradation_manager::set_threshold(quality_threshold threshold) {
    threshold_ = threshold;
}

const quality_threshold& degradation_manager::get_threshold() const {
    return threshold_;
}

void degradation_manager::record_quality(std::string_view operation, double quality_score) {
    if (quality_score >= threshold_.min_quality_score) {
        return;
    }
    auto it = precision_by_op_.find(std::string(operation));
    if (it == precision_by_op_.end()) {
        precision_by_op_[std::string(operation)] = precision_level::high;
    }
}

void degradation_manager::trigger_degradation(std::string_view operation,
                                               precision_level new_level,
                                               std::string_view reason) {
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
    auto precision = get_precision(operation);
    return precision > threshold_.min_acceptable_precision;
}

} // namespace nova::error
