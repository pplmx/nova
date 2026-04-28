#include "cuda/error/timeout.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

namespace nova::error {

timeout_manager::timeout_manager() = default;

timeout_manager::~timeout_manager() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        watchdog_running_ = false;
    }
    watchdog_threads_.clear();
}

timeout_manager& timeout_manager::instance() {
    static timeout_manager instance;
    return instance;
}

operation_id timeout_manager::start_operation(std::string_view name,
                                               std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (operations_.size() >= max_concurrent_) {
        return 0;
    }

    operation_id id = next_id_++;
    const auto now = std::chrono::steady_clock::now();

    operation_context ctx;
    ctx.id = id;
    ctx.name = name;
    ctx.start_time = now;
    ctx.deadline = now + timeout;
    ctx.timeout_duration = timeout;
    ctx.cancelled = false;

    operations_.emplace(id, std::move(ctx));

    return id;
}

void timeout_manager::update_timeout(operation_id id, std::chrono::milliseconds new_timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operations_.find(id);
    if (it != operations_.end()) {
        it->second.deadline = std::chrono::steady_clock::now() + new_timeout;
        it->second.timeout_duration = new_timeout;
    }
}

void timeout_manager::cancel_operation(operation_id id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operations_.find(id);
    if (it != operations_.end()) {
        it->second.cancelled = true;
        if (callback_) {
            callback_(id, make_timeout_error(timeout_error_code::timeout_cancelled));
        }
    }
}

void timeout_manager::end_operation(operation_id id) {
    std::lock_guard<std::mutex> lock(mutex_);
    operations_.erase(id);
}

std::chrono::milliseconds timeout_manager::get_remaining(operation_id id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operations_.find(id);
    if (it == operations_.end()) {
        return std::chrono::milliseconds{0};
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= it->second.deadline) {
        return std::chrono::milliseconds{0};
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        it->second.deadline - now);
}

bool timeout_manager::is_expired(operation_id id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operations_.find(id);
    if (it == operations_.end()) {
        return false;
    }
    return std::chrono::steady_clock::now() > it->second.deadline;
}

bool timeout_manager::is_cancelled(operation_id id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = operations_.find(id);
    if (it == operations_.end()) {
        return true;
    }
    return it->second.cancelled;
}

void timeout_manager::set_callback(timeout_callback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(cb);
}

void timeout_manager::set_config(const timeout_config& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_timeout_ = config.default_timeout;
    max_concurrent_ = config.max_concurrent_operations;

    if (config.watchdog_enabled && watchdog_threads_.empty()) {
        watchdog_running_ = true;
        watchdog_threads_.emplace_back([this]() { watchdog_loop(); });
    } else if (!config.watchdog_enabled) {
        watchdog_running_ = false;
        watchdog_threads_.clear();
    }
}

const timeout_config& timeout_manager::get_config() const {
    static timeout_config config;
    std::lock_guard<std::mutex> lock(mutex_);
    config.default_timeout = default_timeout_;
    config.max_concurrent_operations = max_concurrent_;
    return config;
}

size_t timeout_manager::active_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return operations_.size();
}

void timeout_manager::watchdog_loop() {
    while (watchdog_running_) {
        check_timeouts();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void timeout_manager::check_timeouts() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::steady_clock::now();

    std::vector<operation_id> expired;
    for (const auto& [id, ctx] : operations_) {
        if (!ctx.cancelled && now > ctx.deadline) {
            expired.push_back(id);
        }
    }

    for (const auto& id : expired) {
        if (callback_) {
            callback_(id, make_timeout_error(timeout_error_code::operation_timed_out));
        }
    }
}

timeout_guard::timeout_guard(std::string_view name, std::chrono::milliseconds timeout)
    : id_(timeout_manager::instance().start_operation(name, timeout)) {}

timeout_guard::~timeout_guard() {
    if (active_) {
        timeout_manager::instance().end_operation(id_);
    }
}

timeout_guard::timeout_guard(timeout_guard&& other) noexcept
    : id_(other.id_), active_(other.active_) {
    other.active_ = false;
}

timeout_guard& timeout_guard::operator=(timeout_guard&& other) noexcept {
    if (this != &other) {
        if (active_) {
            timeout_manager::instance().end_operation(id_);
        }
        id_ = other.id_;
        active_ = other.active_;
        other.active_ = false;
    }
    return *this;
}

bool timeout_guard::is_expired() const {
    return timeout_manager::instance().is_expired(id_);
}

std::chrono::milliseconds timeout_guard::remaining() const {
    return timeout_manager::instance().get_remaining(id_);
}

void timeout_guard::cancel() {
    if (active_) {
        timeout_manager::instance().cancel_operation(id_);
        active_ = false;
    }
}

void timeout_guard::extend(std::chrono::milliseconds additional) {
    const auto current = timeout_manager::instance().get_remaining(id_);
    timeout_manager::instance().update_timeout(id_, current + additional);
}

} // namespace nova::error
