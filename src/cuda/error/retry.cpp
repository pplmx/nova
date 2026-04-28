#include "cuda/error/retry.hpp"

namespace nova::error {

circuit_breaker::circuit_breaker(circuit_breaker_config config)
    : config_(config), gen_(rd_()) {}

bool circuit_breaker::allow_request() const {
    switch (state_) {
        case circuit_state::closed:
            return true;
        case circuit_state::open: {
            auto elapsed = std::chrono::steady_clock::now() - last_failure_time_;
            if (elapsed >= config_.reset_timeout) {
                const_cast<circuit_breaker*>(this)->transition_to_half_open();
                return true;
            }
            return false;
        }
        case circuit_state::half_open:
            return true;
    }
    return false;
}

void circuit_breaker::record_success() {
    switch (state_) {
        case circuit_state::closed:
            failure_count_ = 0;
            break;
        case circuit_state::half_open:
            ++success_count_;
            if (success_count_ >= config_.half_open_success_threshold) {
                transition_to_closed();
            }
            break;
        case circuit_state::open:
            break;
    }
}

void circuit_breaker::record_failure() {
    last_failure_time_ = std::chrono::steady_clock::now();

    switch (state_) {
        case circuit_state::closed:
            ++failure_count_;
            if (failure_count_ >= config_.failure_threshold) {
                transition_to_open();
            }
            break;
        case circuit_state::half_open:
            transition_to_open();
            break;
        case circuit_state::open:
            break;
    }
}

void circuit_breaker::transition_to_open() {
    state_ = circuit_state::open;
    success_count_ = 0;
}

void circuit_breaker::transition_to_half_open() {
    state_ = circuit_state::half_open;
    success_count_ = 0;
}

void circuit_breaker::transition_to_closed() {
    state_ = circuit_state::closed;
    failure_count_ = 0;
    success_count_ = 0;
}

retry_executor::retry_executor(retry_config config)
    : config_(config), circuit_breaker_({}), dist_(0, 1000) {}

template<typename Func>
auto retry_executor::execute(Func&& func) -> decltype(func()) {
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

void retry_executor::set_circuit_breaker(circuit_breaker cb) {
    circuit_breaker_ = std::move(cb);
}

std::chrono::milliseconds retry_executor::calculate_delay(int attempt) {
    return calculate_backoff(attempt, config_.base_delay,
                             config_.multiplier, config_.max_delay);
}

std::chrono::milliseconds retry_executor::apply_jitter(std::chrono::milliseconds delay) {
    return full_jitter(delay);
}

} // namespace nova::error
