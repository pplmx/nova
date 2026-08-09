/**
 * @file timeout.hpp
 * @brief Operation timeout management with watchdog monitoring
 * @defgroup timeout Timeout Management
 * @ingroup error
 *
 * This module provides structured timeout management for GPU operations with:
 * - Per-operation timeout tracking and cancellation
 * - Watchdog monitoring for stalled operations
 * - Callback-based timeout notification
 * - RAII guards for automatic cleanup
 *
 * Example usage:
 * @code
 * {
 *     timeout_guard guard("matrix_multiply", std::chrono::seconds(30));
 *     // Perform operation
 *     guard.extend(std::chrono::seconds(10));  // If needed
 * } // Auto-cleans up on scope exit
 * @endcode
 *
 * @note Include this header to enable timeout monitoring
 * @see cuda_error.hpp For error handling
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nova::error {

/**
 * @brief Timeout-specific error codes
 * @enum timeout_error_code
 * @ingroup timeout
 */
enum class timeout_error_code : int {
    /** @brief Operation exceeded its timeout duration */
    operation_timed_out = 1,

    /** @brief Operation exceeded a parent deadline */
    deadline_exceeded = 2,

    /** @brief Watchdog detected a stalled operation */
    watchdog_timeout = 3,

    /** @brief Timeout was manually cancelled */
    timeout_cancelled = 4,
};

/**
 * @brief Timeout error category for std::error_code integration
 * @class timeout_error_category
 * @ingroup timeout
 *
 * Provides error messages and recovery hints for timeout-related errors.
 *
 * @note Use timeout_category() to get the singleton instance
 */
class timeout_error_category : public std::error_category {
public:
    /**
     * @brief Returns the name of the category
     * @return "timeout"
     */
    [[nodiscard]] const char* name() const noexcept override { return "timeout"; }

    /**
     * @brief Returns the error message for a given error code
     * @param ev Numeric error code (timeout_error_code value)
     * @return Human-readable error string
     * @ingroup timeout
     */
    [[nodiscard]] std::string message(int ev) const override {
        switch (static_cast<timeout_error_code>(ev)) {
            case timeout_error_code::operation_timed_out:
                return "Operation timed out";
            case timeout_error_code::deadline_exceeded:
                return "Deadline exceeded";
            case timeout_error_code::watchdog_timeout:
                return "Watchdog detected stalled operation";
            case timeout_error_code::timeout_cancelled:
                return "Timeout was cancelled";
            default:
                return "Unknown timeout error";
        }
    }

    /**
     * @brief Returns a recovery hint for a given error
     * @param ev Numeric error code (timeout_error_code value)
     * @return Actionable suggestion for resolving the error
     * @ingroup timeout
     */
    [[nodiscard]] std::string_view recovery_hint(int ev) const noexcept {
        switch (static_cast<timeout_error_code>(ev)) {
            case timeout_error_code::operation_timed_out:
                return "Increase timeout duration or check for deadlocks";
            case timeout_error_code::deadline_exceeded:
                return "Parent operation deadline propagated; check upstream";
            case timeout_error_code::watchdog_timeout:
                return "Operation stalled; investigate GPU kernel or memory allocation";
            case timeout_error_code::timeout_cancelled:
                return "No action needed - timeout was manually cancelled";
            default:
                return "Review operation and adjust timeout configuration";
        }
    }
};

/**
 * @brief Get the timeout error category instance
 * @return Reference to the singleton timeout_error_category
 * @ingroup timeout
 */
inline const std::error_category& timeout_category() noexcept {
    static timeout_error_category instance;
    return instance;
}

/**
 * @brief Create a timeout error code
 * @param code Timeout error code
 * @param operation Operation name (unused, for future extension)
 * @param duration Timeout duration (unused, for future extension)
 * @param device_id Device ID (unused, for future extension)
 * @return std::error_code for the timeout error
 * @ingroup timeout
 */
inline std::error_code make_timeout_error(timeout_error_code code,
                                          std::string_view operation = {},
                                          std::chrono::milliseconds duration = {},
                                          int device_id = -1) {
    (void)operation;
    (void)duration;
    (void)device_id;
    return std::error_code(static_cast<int>(code), timeout_category());
}

/** @brief Unique identifier for a tracked operation */
using operation_id = uint64_t;

/** @brief Callback invoked when an operation times out */
using timeout_callback = std::function<void(operation_id, std::error_code)>;

/**
 * @brief Configuration for timeout manager
 * @struct timeout_config
 * @ingroup timeout
 */
struct timeout_config {
    /** @brief Default timeout for operations (30 seconds) */
    std::chrono::milliseconds default_timeout{30000};

    /** @brief Interval between watchdog checks (100ms) */
    std::chrono::milliseconds watchdog_interval{100};

    /** @brief Enable watchdog monitoring */
    bool watchdog_enabled{true};

    /** @brief Maximum number of concurrent tracked operations */
    size_t max_concurrent_operations{10000};
};

/**
 * @brief Context for a tracked operation
 * @struct operation_context
 * @ingroup timeout
 */
struct operation_context {
    /** @brief Unique operation identifier */
    operation_id id;

    /** @brief Human-readable operation name */
    std::string_view name;

    /** @brief Time when operation started */
    std::chrono::steady_clock::time_point start_time;

    /** @brief Deadline for the operation */
    std::chrono::steady_clock::time_point deadline;

    /** @brief Configured timeout duration */
    std::chrono::milliseconds timeout_duration;

    /** @brief Whether operation was cancelled */
    bool cancelled{false};

    /**
     * @brief Check if deadline has passed
     * @return true if current time exceeds deadline
     */
    [[nodiscard]] bool is_expired() const noexcept {
        return std::chrono::steady_clock::now() > deadline;
    }

    /**
     * @brief Get elapsed time since operation started
     * @return Elapsed duration
     */
    [[nodiscard]] std::chrono::milliseconds elapsed() const noexcept {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
    }
};

/**
 * @brief Singleton manager for tracking operation timeouts
 * @class timeout_manager
 * @ingroup timeout
 *
 * Provides centralized timeout tracking with watchdog monitoring.
 *
 * @note Use instance() to get the singleton
 * @note Thread-safe for concurrent operations
 */
class timeout_manager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the timeout_manager singleton
     */
    static timeout_manager& instance();

    timeout_manager(const timeout_manager&) = delete;
    timeout_manager& operator=(const timeout_manager&) = delete;

    /**
     * @brief Start tracking a new operation
     * @param name Operation name for logging
     * @param timeout Timeout duration
     * @return Unique operation ID
     */
    [[nodiscard]] operation_id start_operation(std::string_view name,
                                                std::chrono::milliseconds timeout);

    /**
     * @brief Update timeout for an existing operation
     * @param id Operation ID
     * @param new_timeout New timeout duration
     */
    void update_timeout(operation_id id, std::chrono::milliseconds new_timeout);

    /** @brief Cancel a tracked operation */
    void cancel_operation(operation_id id);

    /** @brief End tracking an operation (success or permanent failure) */
    void end_operation(operation_id id);

    /** @brief Get remaining time for an operation */
    [[nodiscard]] std::chrono::milliseconds get_remaining(operation_id id) const;

    /** @brief Check if an operation has expired */
    [[nodiscard]] bool is_expired(operation_id id) const;

    /** @brief Check if an operation was cancelled */
    [[nodiscard]] bool is_cancelled(operation_id id) const;

    /** @brief Set callback for timeout notifications */
    void set_callback(timeout_callback cb);

    /** @brief Update manager configuration */
    void set_config(const timeout_config& config);

    /** @brief Get current configuration */
    [[nodiscard]] const timeout_config& get_config() const;

    /** @brief Get count of currently active tracked operations */
    [[nodiscard]] size_t active_count() const;

    /**
     * @brief Reset all tracked state back to defaults.
     *
     * Removes every active operation, clears the timeout callback, and stops
     * any running watchdog thread. Intended for test isolation and graceful
     * teardown between logical units, so state from one usage (including
     * callbacks capturing now-destroyed objects) never leaks into another.
     */
    void reset();

private:
    timeout_manager();
    ~timeout_manager();

    void watchdog_loop();
    void check_timeouts();

    mutable std::mutex mutex_;
    std::unordered_map<operation_id, operation_context> operations_;
    operation_id next_id_{1};
    std::chrono::milliseconds default_timeout_{30000};
    timeout_callback callback_;
    bool watchdog_running_{false};
    std::vector<std::jthread> watchdog_threads_;
    std::atomic<int64_t> watchdog_interval_ms_{100};
    size_t max_concurrent_{10000};
};

/**
 * @brief RAII guard for automatic timeout management
 * @class timeout_guard
 * @ingroup timeout
 *
 * Automatically registers operation with timeout_manager and cleans up on destruction.
 *
 * Example:
 * @code
 * void longRunningTask() {
 *     timeout_guard guard("task", std::chrono::minutes(5));
 *     // Do work...
 *     if (guard.is_expired()) {
 *         // Handle timeout
 *     }
 * }
 * @endcode
 *
 * @note Automatically cancels operation if destroyed without calling cancel()
 */
class [[nodiscard]] timeout_guard {
public:
    /**
     * @brief Construct a timeout_guard
     * @param name Operation name
     * @param timeout Timeout duration
     */
    timeout_guard(std::string_view name, std::chrono::milliseconds timeout);
    ~timeout_guard();

    timeout_guard(const timeout_guard&) = delete;
    timeout_guard& operator=(const timeout_guard&) = delete;
    timeout_guard(timeout_guard&& other) noexcept;
    timeout_guard& operator=(timeout_guard&& other) noexcept;

    /** @brief Get the operation ID */
    [[nodiscard]] operation_id id() const noexcept { return id_; }

    /** @brief Check if operation has expired */
    [[nodiscard]] bool is_expired() const;

    /** @brief Get remaining time until expiration */
    [[nodiscard]] std::chrono::milliseconds remaining() const;

    /** @brief Cancel this operation */
    void cancel();

    /** @brief Extend timeout by additional duration */
    void extend(std::chrono::milliseconds additional);

private:
    operation_id id_;
    bool active_{true};
};

/**
 * @brief Scope object for timeout configuration
 * @struct timeout_scope
 * @ingroup timeout
 *
 * Used with with_timeout() helper to specify timeout for operations.
 */
struct timeout_scope {
    /** @brief Timeout duration */
    std::chrono::milliseconds timeout;
};

/**
 * @brief Create a timeout scope
 * @param t Timeout duration
 * @return timeout_scope with specified duration
 * @ingroup timeout
 *
 * Example:
 * @code
 * void task(timeout_scope scope) {
 *     // Task with configured timeout
 * }
 *
 * task(with_timeout(std::chrono::seconds(30)));
 * @endcode
 */
inline timeout_scope with_timeout(std::chrono::milliseconds t) { return {t}; }

} // namespace nova::error
