#pragma once

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

enum class timeout_error_code : int {
    operation_timed_out = 1,
    deadline_exceeded = 2,
    watchdog_timeout = 3,
    timeout_cancelled = 4,
};

class timeout_error_category : public std::error_category {
public:
    [[nodiscard]] const char* name() const noexcept override { return "timeout"; }

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

inline const std::error_category& timeout_category() noexcept {
    static timeout_error_category instance;
    return instance;
}

inline std::error_code make_timeout_error(timeout_error_code code,
                                          std::string_view operation = {},
                                          std::chrono::milliseconds duration = {},
                                          int device_id = -1) {
    return std::error_code(static_cast<int>(code), timeout_category());
}

using operation_id = uint64_t;

using timeout_callback = std::function<void(operation_id, std::error_code)>;

struct timeout_config {
    std::chrono::milliseconds default_timeout{30000};
    std::chrono::milliseconds watchdog_interval{100};
    bool watchdog_enabled{true};
    size_t max_concurrent_operations{10000};
};

struct operation_context {
    operation_id id;
    std::string_view name;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point deadline;
    std::chrono::milliseconds timeout_duration;
    bool cancelled{false};

    [[nodiscard]] bool is_expired() const noexcept {
        return std::chrono::steady_clock::now() > deadline;
    }

    [[nodiscard]] std::chrono::milliseconds elapsed() const noexcept {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
    }
};

class timeout_manager {
public:
    static timeout_manager& instance();

    timeout_manager(const timeout_manager&) = delete;
    timeout_manager& operator=(const timeout_manager&) = delete;

    [[nodiscard]] operation_id start_operation(std::string_view name,
                                                std::chrono::milliseconds timeout);

    void update_timeout(operation_id id, std::chrono::milliseconds new_timeout);
    void cancel_operation(operation_id id);
    void end_operation(operation_id id);

    [[nodiscard]] std::chrono::milliseconds get_remaining(operation_id id) const;
    [[nodiscard]] bool is_expired(operation_id id) const;
    [[nodiscard]] bool is_cancelled(operation_id id) const;

    void set_callback(timeout_callback cb);
    void set_config(const timeout_config& config);
    [[nodiscard]] const timeout_config& get_config() const;

    [[nodiscard]] size_t active_count() const;

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
    size_t max_concurrent_{10000};
};

class [[nodiscard]] timeout_guard {
public:
    timeout_guard(std::string_view name, std::chrono::milliseconds timeout);
    ~timeout_guard();

    timeout_guard(const timeout_guard&) = delete;
    timeout_guard& operator=(const timeout_guard&) = delete;
    timeout_guard(timeout_guard&& other) noexcept;
    timeout_guard& operator=(timeout_guard&& other) noexcept;

    [[nodiscard]] operation_id id() const noexcept { return id_; }
    [[nodiscard]] bool is_expired() const;
    [[nodiscard]] std::chrono::milliseconds remaining() const;

    void cancel();
    void extend(std::chrono::milliseconds additional);

private:
    operation_id id_;
    bool active_{true};
};

struct timeout_scope {
    std::chrono::milliseconds timeout;
};

inline timeout_scope with_timeout(std::chrono::milliseconds t) { return {t}; }

} // namespace nova::error
