#include "cuda/error/timeout_context.hpp"

namespace nova::error {

timeout_context::timeout_context(timeout_context* parent, std::chrono::milliseconds timeout)
    : parent_(parent) {
    if (parent && timeout == std::chrono::milliseconds::zero()) {
        auto remaining = timeout_manager::instance().get_remaining(parent->id_);
        id_ = timeout_manager::instance().start_operation("child", remaining);
    } else {
        id_ = timeout_manager::instance().start_operation("child", timeout);
    }
}

timeout_context::~timeout_context() {
    timeout_manager::instance().end_operation(id_);
}

bool timeout_context::is_expired() const {
    return timeout_manager::instance().is_expired(id_);
}

std::chrono::milliseconds timeout_context::remaining() const {
    return timeout_manager::instance().get_remaining(id_);
}

void timeout_context::set_deadline_callback(timeout_callback cb) {
    auto& manager = timeout_manager::instance();
    manager.set_callback([cb = std::move(cb), id = id_](operation_id op_id, std::error_code ec) {
        if (op_id == id) {
            cb(op_id, ec);
        }
    });
}

scoped_timeout::scoped_timeout(std::string_view name, std::chrono::milliseconds timeout)
    : ctx_(nullptr, timeout) {
    auto id = timeout_manager::instance().start_operation(name, timeout);
    static_cast<void>(id);
}

scoped_timeout::~scoped_timeout() = default;

} // namespace nova::error
