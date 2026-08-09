/**
 * @file retry.hpp
 * @brief Retry logic with exponential backoff and circuit breaker
 * @defgroup retry Retry and Circuit Breaker
 * @ingroup error
 *
 * This module provides robust retry mechanisms with:
 * - Configurable exponential backoff with jitter
 * - Circuit breaker pattern for preventing cascade failures
 * - Integration with error handling for transient failures
 *
 * Example usage:
 * @code
 * retry_executor executor({
 *     .base_delay = std::chrono::milliseconds(100),
 *     .multiplier = 2.0,
 *     .max_delay = std::chrono::seconds(30),
 *     .max_attempts = 5
 * });
 *
 * auto result = executor.execute([]() {
 *     return my_unreliable_operation();
 * });
 * @endcode
 *
 * @see timeout.hpp For timeout management
 * @see cuda_error.hpp For CUDA error handling
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <random>
#include <thread>

namespace nova::error {

/** @brief Result of a retryable operation */
enum class retry_result { success, failure, timeout };

/**
 * @brief Configuration for retry behavior
 * @struct retry_config
 * @ingroup retry
 */
struct retry_config {
    /** @brief Initial delay between retries (100ms) */
    std::chrono::milliseconds base_delay{100};

    /** @brief Backoff multiplier (2.0 = double each attempt) */
    double multiplier{2.0};

    /** @brief Maximum delay cap (30 seconds) */
    std::chrono::milliseconds max_delay{30000};

    /** @brief Maximum retry attempts (5) */
    int max_attempts{5};

    /** @brief Enable random jitter to prevent thundering herd */
    bool jitter_enabled{true};
};

/**
 * @brief Configuration for circuit breaker
 * @struct circuit_breaker_config
 * @ingroup retry
 */
struct circuit_breaker_config {
    /** @brief Failures before opening circuit (5) */
    int failure_threshold{5};

    /** @brief Time before attempting half-open (30 seconds) */
    std::chrono::milliseconds reset_timeout{30000};

    /** @brief Successes needed in half-open to close (3) */
    int half_open_success_threshold{3};
};

/** @brief Circuit breaker state machine states */
enum class circuit_state { closed, open, half_open };

/**
 * @brief Circuit breaker pattern implementation
 * @class circuit_breaker
 * @ingroup retry
 *
 * Prevents cascade failures by stopping requests when failure rate is high.
 *
 * State transitions:
 * - closed -> open: After failure_threshold consecutive failures
 * - open -> half_open: After reset_timeout elapses
 * - half_open -> closed: After half_open_success_threshold successes
 * - half_open -> open: On any failure
 */
class circuit_breaker {
public:
    /**
     * @brief Construct a circuit breaker
     * @param config Circuit breaker configuration
     */
    explicit circuit_breaker(circuit_breaker_config config);
    circuit_breaker(circuit_breaker&&) noexcept = default;
    circuit_breaker& operator=(circuit_breaker&&) noexcept = default;

    /**
     * @brief Check if request should be allowed
     * @return true if circuit is closed or half-open
     */
    [[nodiscard]] bool allow_request() const;

    /** @brief Record a successful operation */
    void record_success();

    /** @brief Record a failed operation */
    void record_failure();

    /** @brief Get current circuit state */
    [[nodiscard]] circuit_state state() const noexcept { return state_; }

private:
    void transition_to_open();
    void transition_to_half_open();
    void transition_to_closed();

    circuit_breaker_config config_;
    circuit_state state_{circuit_state::closed};
    int failure_count_{0};
    int success_count_{0};
    std::chrono::steady_clock::time_point last_failure_time_;
};

/**
 * @brief Executes operations with automatic retry
 * @class retry_executor
 * @ingroup retry
 *
 * Automatically retries failed operations with exponential backoff
 * and optional circuit breaker integration.
 *
 * Example:
 * @code
 * retry_executor executor(retry_config{.max_attempts = 3});
 * auto result = executor.execute([]() { return fetch_data(); });
 * std::cout << "Attempts: " << executor.attempt_count() << "\n";
 * @endcode
 */
class retry_executor {
public:
    /**
     * @brief Construct a retry executor
     * @param config Retry configuration
     */
    explicit retry_executor(retry_config config);

    /**
     * @brief Execute function with retry logic
     * @tparam Func Callable type
     * @param func Function to execute
     * @return Result of func() on success
     * @throws std::runtime_error if max attempts exceeded or circuit open
     */
    template<typename Func>
    std::invoke_result_t<Func> execute(Func&& func);

    /** @brief Set circuit breaker for the executor */
    void set_circuit_breaker(circuit_breaker cb);

    /** @brief Get number of attempts made in last execute */
    [[nodiscard]] int attempt_count() const noexcept { return attempts_; }

    /** @brief Check if last execute succeeded */
    [[nodiscard]] bool was_successful() const noexcept { return success_; }

private:
    std::chrono::milliseconds calculate_delay(int attempt);
    std::chrono::milliseconds apply_jitter(std::chrono::milliseconds delay);

    retry_config config_;
    circuit_breaker circuit_breaker_;
    int attempts_{0};
    bool success_{false};
    mutable std::uniform_int_distribution<int> dist_;
};

/**
 * @brief Calculate exponential backoff delay
 * @param attempt Attempt number (1-indexed)
 * @param base Base delay
 * @param multiplier Exponential multiplier
 * @param max_delay Maximum delay cap
 * @return Calculated delay
 * @ingroup retry
 */
inline std::chrono::milliseconds calculate_backoff(int attempt,
                                                   std::chrono::milliseconds base,
                                                   double multiplier,
                                                   std::chrono::milliseconds max_delay) {
    auto delay = static_cast<double>(base.count()) * std::pow(multiplier, attempt - 1);
    auto capped = std::min(delay, static_cast<double>(max_delay.count()));
    return std::chrono::milliseconds(static_cast<int>(capped));
}

/**
 * @brief Apply full jitter to delay
 * @param delay Base delay
 * @return Random value between 0 and delay
 * @ingroup retry
 *
 * Full jitter: pick a random value between 0 and the calculated delay.
 * This prevents thundering herd when multiple clients recover simultaneously.
 */
inline std::chrono::milliseconds full_jitter(std::chrono::milliseconds delay) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(delay.count()));
    return std::chrono::milliseconds(dist(gen));
}

template<typename Func>
std::invoke_result_t<Func> retry_executor::execute(Func&& func) {
    success_ = false;
    attempts_ = 0;

    while (attempts_ < config_.max_attempts) {
        if (!circuit_breaker_.allow_request()) {
            throw std::runtime_error("Circuit breaker is open");
        }

        ++attempts_;
        try {
            auto result = func();
            circuit_breaker_.record_success();
            success_ = true;
            return result;
        } catch (...) {
            circuit_breaker_.record_failure();
            if (attempts_ < config_.max_attempts) {
                auto delay = calculate_delay(attempts_);
                if (config_.jitter_enabled) {
                    delay = apply_jitter(delay);
                }
                std::this_thread::sleep_for(delay);
            }
        }
    }

    throw std::runtime_error("Max retry attempts exceeded");
}

} // namespace nova::error
