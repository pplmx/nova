/**
 * @file degrade.hpp
 * @brief Algorithm degradation and precision management
 * @defgroup degrade Algorithm Degradation
 * @ingroup error
 *
 * This module provides graceful degradation of numerical precision
 * when algorithms fail quality thresholds, allowing operations to
 * complete with lower precision rather than failing entirely.
 *
 * Example usage:
 * @code
 * auto manager = degradation_manager::instance();
 * manager.set_threshold({.min_quality_score = 0.85, .max_retry_before_degrade = 2});
 *
 * manager.record_quality("matrix_inverse", quality_score);
 * if (manager.should_degrade("matrix_inverse")) {
 *     auto new_level = degrade(manager.get_precision("matrix_inverse"));
 *     manager.trigger_degradation("matrix_inverse", new_level, "Quality below threshold");
 * }
 * @endcode
 *
 * @see retry.hpp For retry-based error recovery
 */

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace nova::error {

/**
 * @brief Numerical precision levels for algorithm degradation
 * @enum precision_level
 * @ingroup degrade
 *
 * Ordered from highest to lowest precision.
 */
enum class precision_level : int {
    /** @brief Full precision (FP64) */
    high = 0,

    /** @brief Standard precision (FP32) */
    medium = 1,

    /** @brief Reduced precision (FP16) */
    low = 2,

    /** @brief Count of precision levels */
    count = 3
};

/**
 * @brief Get the next lower precision level
 * @param current Current precision level
 * @return Degraded precision level (unchanged if already at lowest)
 * @ingroup degrade
 */
inline precision_level degrade(precision_level current) {
    auto next = static_cast<int>(current) + 1;
    if (next >= static_cast<int>(precision_level::count)) {
        return current;
    }
    return static_cast<precision_level>(next);
}

/**
 * @brief Get the name string for a precision level
 * @param level Precision level
 * @return Human-readable name ("FP64", "FP32", "FP16", or "unknown")
 * @ingroup degrade
 */
inline const char* precision_level_name(precision_level level) {
    switch (level) {
        case precision_level::high: return "FP64";
        case precision_level::medium: return "FP32";
        case precision_level::low: return "FP16";
        default: return "unknown";
    }
}

/**
 * @brief Record of a precision degradation event
 * @struct degradation_event
 * @ingroup degrade
 */
struct degradation_event {
    /** @brief Name of the degraded operation */
    std::string_view operation;

    /** @brief Previous precision level */
    precision_level from;

    /** @brief New (degraded) precision level */
    precision_level to;

    /** @brief When degradation occurred */
    std::chrono::steady_clock::time_point timestamp;

    /** @brief Reason for degradation */
    std::string_view reason;
};

/** @brief Callback invoked when degradation occurs */
using degradation_callback = std::function<void(const degradation_event&)>;

/**
 * @brief Registry for algorithms with multiple precision implementations
 * @class algorithm_registry
 * @ingroup degrade
 *
 * Manages registration and lookup of algorithms that support
 * multiple precision levels for graceful degradation.
 */
class algorithm_registry {
public:
    /** @brief Factory function type for algorithm creation */
    using factory_func = std::function<std::unique_ptr<void>()>;

    /**
     * @brief Register an algorithm implementation
     * @tparam Algorithm Algorithm class type
     * @param name Algorithm name
     * @param level Precision level
     * @param factory Factory function to create instances
     */
    template<typename Algorithm>
    void register_algorithm(std::string_view name, precision_level level, factory_func factory);

    /**
     * @brief Create an algorithm instance
     * @tparam Algorithm Algorithm class type
     * @param name Algorithm name
     * @param level Desired precision level
     * @return Unique pointer to algorithm instance, or nullptr if not available
     */
    template<typename Algorithm>
    [[nodiscard]] std::unique_ptr<Algorithm> create(std::string_view name, precision_level level) const;

    /**
     * @brief Check if fallback exists at specified level
     * @param name Algorithm name
     * @param level Precision level
     * @return true if algorithm available at level
     */
    [[nodiscard]] bool has_fallback(std::string_view name, precision_level level) const;

    /**
     * @brief Get best available precision level
     * @param name Algorithm name
     * @param min_level Minimum acceptable level
     * @return Best available level >= min_level, or min_level if none available
     */
    [[nodiscard]] precision_level get_best_available(std::string_view name, precision_level min_level) const;

private:
    struct entry {
        std::string name;
        precision_level level;
        factory_func factory;
    };
    std::vector<entry> entries_;
};

/**
 * @brief Quality thresholds for degradation decisions
 * @struct quality_threshold
 * @ingroup degrade
 */
struct quality_threshold {
    /** @brief Minimum acceptable quality score (0.0-1.0) */
    double min_quality_score{0.8};

    /** @brief Retries before triggering degradation */
    int max_retry_before_degrade{3};

    /** @brief Lowest acceptable precision level */
    precision_level min_acceptable_precision{precision_level::medium};
};

/**
 * @brief Singleton manager for tracking quality and triggering degradation
 * @class degradation_manager
 * @ingroup degrade
 *
 * Monitors operation quality and triggers degradation when thresholds are breached.
 *
 * @note Use instance() to get the singleton
 * @see degrade.hpp For precision level definitions
 */
class degradation_manager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the degradation_manager singleton
     */
    static degradation_manager& instance();

    /** @brief Set callback for degradation events */
    void set_callback(degradation_callback cb);

    /** @brief Set quality thresholds */
    void set_threshold(quality_threshold threshold);

    /** @brief Get current quality thresholds */
    [[nodiscard]] const quality_threshold& get_threshold() const;

    /**
     * @brief Record quality score for an operation
     * @param operation Operation name
     * @param quality_score Quality score (0.0-1.0)
     */
    void record_quality(std::string_view operation, double quality_score);

    /**
     * @brief Trigger degradation for an operation
     * @param operation Operation name
     * @param new_level New (degraded) precision level
     * @param reason Reason for degradation
     */
    void trigger_degradation(std::string_view operation, precision_level new_level, std::string_view reason);

    /**
     * @brief Get current precision level for an operation
     * @param operation Operation name
     * @return Current precision level
     */
    [[nodiscard]] precision_level get_precision(std::string_view operation) const;

    /**
     * @brief Check if operation should be degraded
     * @param operation Operation name
     * @return true if quality scores indicate degradation needed
     */
    [[nodiscard]] bool should_degrade(std::string_view operation) const;
    /**
     * @brief Reset all tracked state back to defaults.
     *
     * Clears the registered callback, restores default quality thresholds,
     * and forgets every operation's current precision. This lets a
     * long-running process (or a test runner) isolate one usage of the
     * manager from another, instead of leaking a previously-installed
     * callback (which may capture references to already-destroyed objects)
     * and stale precision/threshold state across scopes.
     */
    void reset();

private:
    degradation_manager() = default;

    degradation_callback callback_;
    quality_threshold threshold_;
    std::unordered_map<std::string, precision_level> precision_by_op_;
};

} // namespace nova::error
