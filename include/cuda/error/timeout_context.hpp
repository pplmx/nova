#pragma once

#include "cuda/error/timeout.hpp"

namespace nova::error {

class timeout_context {
public:
    timeout_context(timeout_context* parent, std::chrono::milliseconds timeout);
    ~timeout_context();

    timeout_context(const timeout_context&) = delete;
    timeout_context& operator=(const timeout_context&) = delete;

    [[nodiscard]] operation_id id() const noexcept { return id_; }
    [[nodiscard]] bool is_expired() const;
    [[nodiscard]] std::chrono::milliseconds remaining() const;

    void set_deadline_callback(timeout_callback cb);

private:
    operation_id id_;
    timeout_context* parent_{nullptr};
};

class [[nodiscard]] scoped_timeout {
public:
    scoped_timeout(std::string_view name, std::chrono::milliseconds timeout);
    ~scoped_timeout();

    scoped_timeout(const scoped_timeout&) = delete;
    scoped_timeout& operator=(const scoped_timeout&) = delete;

    [[nodiscard]] timeout_context& context() noexcept { return ctx_; }

private:
    timeout_context ctx_;
};

} // namespace nova::error
