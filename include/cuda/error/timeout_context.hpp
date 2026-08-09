/**
 * @file timeout_context.hpp
 * @brief Hierarchical timeout context management
 * @defgroup timeout_context Hierarchical Timeout Context
 * @ingroup timeout
 *
 * Provides hierarchical timeout tracking where child contexts inherit
 * parent deadlines and can specify additional time allowances.
 *
 * Example usage:
 * @code
 * {
 *     scoped_timeout outer("batch_process", std::chrono::minutes(10));
 *     for (auto& item : batch) {
 *         if (outer.context().is_expired()) break;
 *         process_item(item);  // With inherited deadline context
 *     }
 * }
 * @endcode
 *
 * @see timeout.hpp For basic timeout management
 */

#pragma once

#include "cuda/error/timeout.hpp"

namespace nova::error {

/**
 * @brief Hierarchical timeout context with parent deadline propagation
 * @class timeout_context
 * @ingroup timeout_context
 *
 * Child contexts inherit parent deadlines while adding their own timeout.
 * The effective deadline is min(parent_deadline, start_time + timeout).
 */
class timeout_context {
public:
    /**
     * @brief Construct a timeout context
     * @param parent Parent context (nullptr for root)
     * @param timeout Additional timeout allowance
     * @param name Operation name used when registering with the manager
     */
    timeout_context(timeout_context* parent, std::chrono::milliseconds timeout,
                    std::string_view name = "child");
    ~timeout_context();

    timeout_context(const timeout_context&) = delete;
    timeout_context& operator=(const timeout_context&) = delete;

    /**
     * @brief Get the operation ID
     * @return Unique operation identifier
     */
    [[nodiscard]] operation_id id() const noexcept { return id_; }

    /**
     * @brief Check if deadline has passed
     * @return true if current time exceeds effective deadline
     */
    [[nodiscard]] bool is_expired() const;

    /**
     * @brief Get remaining time until deadline
     * @return Remaining duration (negative if expired)
     */
    [[nodiscard]] std::chrono::milliseconds remaining() const;

    /**
     * @brief Set callback for deadline expiration
     * @param cb Callback invoked when deadline is reached
     */
    void set_deadline_callback(timeout_callback cb);

private:
    operation_id id_;
    timeout_context* parent_{nullptr};
};

/**
 * @brief RAII wrapper for scoped timeout context
 * @class scoped_timeout
 * @ingroup timeout_context
 *
 * Automatically manages a timeout_context with automatic cleanup.
 * Provides access to the underlying context for deadline checking.
 *
 * Example:
 * @code
 * void processWithDeadline(std::string_view name, auto duration) {
 *     scoped_timeout timer(name, duration);
 *     // Check deadline during processing
 *     if (timer.context().is_expired()) return;
 *     // Continue processing...
 * }
 * @endcode
 */
class [[nodiscard]] scoped_timeout {
public:
    /**
     * @brief Construct a scoped_timeout
     * @param name Operation name
     * @param timeout Timeout duration
     */
    scoped_timeout(std::string_view name, std::chrono::milliseconds timeout);
    ~scoped_timeout();

    scoped_timeout(const scoped_timeout&) = delete;
    scoped_timeout& operator=(const scoped_timeout&) = delete;

    /**
     * @brief Get the underlying timeout context
     * @return Reference to the timeout_context
     */
    [[nodiscard]] timeout_context& context() noexcept { return ctx_; }

private:
    timeout_context ctx_;
};

} // namespace nova::error
