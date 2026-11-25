#include "cuda/error/timeout_context.hpp"

namespace nova::error {

    timeout_context::timeout_context(timeout_context* parent, std::chrono::milliseconds timeout, std::string_view name)
        : parent_(parent) {
        // The header documents the effective deadline as min(parent_deadline, start + timeout), and
        // parent_ exists to carry that. The parent used to be consulted only when timeout was zero,
        // so a child asking for longer than its parent had left outlived the operation it belonged
        // to -- measured: parent(50ms) with child(&parent, 5000ms) reported 4999ms remaining, and
        // still reported 4879ms after the parent had expired.
        //
        // Taking the smaller of the two makes both halves of that sentence true: a zero timeout
        // still means "all the parent has left", and an explicit timeout still wins when it is the
        // shorter one.
        auto allowed = timeout;
        if (parent) {
            const auto parent_remaining = timeout_manager::instance().get_remaining(parent->id_);
            if (timeout == std::chrono::milliseconds::zero() || parent_remaining < allowed) {
                allowed = parent_remaining;
            }
        }
        id_ = timeout_manager::instance().start_operation(name, allowed);
    }

    timeout_context::~timeout_context() {
        timeout_manager::instance().end_operation(id_);
    }

    bool timeout_context::is_tracked() const {
        return timeout_manager::instance().is_tracked(id_);
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
        : ctx_(nullptr, timeout, name) {}

    scoped_timeout::~scoped_timeout() = default;

}  // namespace nova::error
