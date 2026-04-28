#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace nova::error {

enum class precision_level : int {
    high = 0,
    medium = 1,
    low = 2,
    count = 3
};

inline precision_level degrade(precision_level current) {
    auto next = static_cast<int>(current) + 1;
    if (next >= static_cast<int>(precision_level::count)) {
        return current;
    }
    return static_cast<precision_level>(next);
}

inline const char* precision_level_name(precision_level level) {
    switch (level) {
        case precision_level::high: return "FP64";
        case precision_level::medium: return "FP32";
        case precision_level::low: return "FP16";
        default: return "unknown";
    }
}

struct degradation_event {
    std::string_view operation;
    precision_level from;
    precision_level to;
    std::chrono::steady_clock::time_point timestamp;
    std::string_view reason;
};

using degradation_callback = std::function<void(const degradation_event&)>;

class algorithm_registry {
public:
    using factory_func = std::function<std::unique_ptr<void>()>;

    template<typename Algorithm>
    void register_algorithm(std::string_view name, precision_level level, factory_func factory);

    template<typename Algorithm>
    [[nodiscard]] std::unique_ptr<Algorithm> create(std::string_view name, precision_level level) const;

    [[nodiscard]] bool has_fallback(std::string_view name, precision_level level) const;
    [[nodiscard]] precision_level get_best_available(std::string_view name, precision_level min_level) const;

private:
    struct entry {
        std::string name;
        precision_level level;
        factory_func factory;
    };
    std::vector<entry> entries_;
};

struct quality_threshold {
    double min_quality_score{0.8};
    int max_retry_before_degrade{3};
    precision_level min_acceptable_precision{precision_level::medium};
};

class degradation_manager {
public:
    static degradation_manager& instance();

    void set_callback(degradation_callback cb);
    void set_threshold(quality_threshold threshold);
    [[nodiscard]] const quality_threshold& get_threshold() const;

    void record_quality(std::string_view operation, double quality_score);
    void trigger_degradation(std::string_view operation, precision_level new_level, std::string_view reason);

    [[nodiscard]] precision_level get_precision(std::string_view operation) const;
    [[nodiscard]] bool should_degrade(std::string_view operation) const;

private:
    degradation_manager() = default;

    degradation_callback callback_;
    quality_threshold threshold_;
    std::unordered_map<std::string, precision_level> precision_by_op_;
};

} // namespace nova::error
