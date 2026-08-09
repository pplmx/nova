#include "cuda/comm/comm_error_recovery.h"

#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace nova::comm {

struct HealthMonitor::Impl {
    std::atomic<bool> running{false};
    std::thread monitor_thread;
    std::chrono::seconds timeout_threshold{60};
    std::chrono::milliseconds check_interval{1000};
    ErrorCallback error_callback;

    struct CommState {
        cudaStream_t stream;
        std::chrono::steady_clock::time_point last_check;
        bool is_healthy = true;
        bool is_stalled = false;
        int consecutive_not_ready = 0;
    };

    std::unordered_map<void*, CommState> comm_states;
    mutable std::shared_mutex state_mutex;

    int stall_confirmations_required = 3;
};

HealthMonitor& HealthMonitor::instance() {
    static HealthMonitor monitor;
    return monitor;
}

HealthMonitor::~HealthMonitor() {
    stop();
}

void HealthMonitor::start() {
    if (impl_->running.load()) {
        return;
    }

    impl_->running.store(true);
    impl_->monitor_thread = std::thread([this]() {
        while (impl_->running.load()) {
            // Collect callback invocations under the lock, then invoke them
            // after releasing it. Calling a user-supplied error_callback while
            // holding state_mutex could self-deadlock if that callback calls
            // register_comm()/unregister_comm() (which take a unique lock on
            // the same mutex).
            ErrorCallback callback;
            std::vector<CommError> to_report;
            {
                std::shared_lock lock(impl_->state_mutex);
                callback = impl_->error_callback;
                for (auto& [comm_id, state] : impl_->comm_states) {
                    cudaError_t result = cudaStreamQuery(state.stream);
                    if (result == cudaErrorNotReady) {
                        state.consecutive_not_ready++;
                        auto elapsed = std::chrono::steady_clock::now() - state.last_check;
                        if (elapsed > impl_->timeout_threshold &&
                            state.consecutive_not_ready >= impl_->stall_confirmations_required) {
                            if (!state.is_stalled && callback) {
                                state.is_healthy = false;
                                state.is_stalled = true;

                                CommError error;
                                error.category = ErrorCategory::Timeout;
                                error.severity = ErrorSeverity::Recoverable;
                                error.message = "Communicator stalled for " +
                                    std::to_string(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()) +
                                    "s (" + std::to_string(state.consecutive_not_ready) + " confirmations)";
                                error.device_id = 0;
                                error.stream = state.stream;
                                error.timestamp = std::chrono::steady_clock::now();
                                to_report.push_back(std::move(error));
                            }
                        }
                    } else if (result == cudaSuccess) {
                        state.last_check = std::chrono::steady_clock::now();
                        state.is_healthy = true;
                        state.is_stalled = false;
                        state.consecutive_not_ready = 0;
                    }
                }
            }
            for (const auto& error : to_report) {
                if (callback) {
                    callback(error);
                }
            }
            std::this_thread::sleep_for(impl_->check_interval);
        }
    });
}

void HealthMonitor::stop() {
    impl_->running.store(false);
    if (impl_->monitor_thread.joinable()) {
        impl_->monitor_thread.join();
    }
}

bool HealthMonitor::is_running() const {
    return impl_->running.load();
}

void HealthMonitor::register_comm(void* comm_id, cudaStream_t stream) {
    std::unique_lock lock(impl_->state_mutex);
    impl_->comm_states[comm_id] = {stream, std::chrono::steady_clock::now(), true, false};
}

void HealthMonitor::unregister_comm(void* comm_id) {
    std::unique_lock lock(impl_->state_mutex);
    impl_->comm_states.erase(comm_id);
}

void HealthMonitor::set_timeout_threshold(std::chrono::seconds timeout) {
    impl_->timeout_threshold = timeout;
}

void HealthMonitor::set_check_interval(std::chrono::milliseconds interval) {
    impl_->check_interval = interval;
}

void HealthMonitor::set_stall_confirmations(int count) {
    impl_->stall_confirmations_required = std::max(1, count);
}

void HealthMonitor::set_error_callback(ErrorCallback callback) {
    impl_->error_callback = std::move(callback);
}

std::optional<HealthMonitor::HealthStatus> HealthMonitor::get_status(void* comm_id) const {
    std::shared_lock lock(impl_->state_mutex);
    auto it = impl_->comm_states.find(comm_id);
    if (it == impl_->comm_states.end()) {
        return std::nullopt;
    }

    HealthStatus status;
    status.comm_id = comm_id;
    status.is_healthy = it->second.is_healthy;
    status.is_stalled = it->second.is_stalled;
    status.last_progress = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - it->second.last_check);
    return status;
}

struct RetryHandler::Impl {
    RetryConfig config;
    std::atomic<int> consecutive_failures{0};
    std::atomic<bool> circuit_open{false};
    std::chrono::steady_clock::time_point circuit_opened_at;
    std::chrono::seconds circuit_breaker_window{60};
    std::default_random_engine rng{std::random_device{}()};

    int get_retry_delay_ms(int attempt) {
        auto base_delay = static_cast<int>(config.initial_delay.count());
        auto delay = static_cast<int>(base_delay * std::pow(config.backoff_multiplier, attempt - 1));
        delay = std::min(delay, static_cast<int>(config.max_delay.count()));

        if (config.use_jitter) {
            std::uniform_int_distribution<int> dist(0, delay / 2);
            delay += dist(rng);
        }

        return delay;
    }
};

RetryHandler& RetryHandler::instance() {
    static RetryHandler handler;
    return handler;
}

void RetryHandler::set_config(const RetryConfig& config) {
    impl_->config = config;
}

RetryHandler::RetryConfig RetryHandler::get_config() const {
    return impl_->config;
}

void RetryHandler::reset_circuit_breaker() {
    impl_->consecutive_failures.store(0);
    impl_->circuit_open.store(false);
}

bool RetryHandler::is_circuit_open() const {
    if (!impl_->circuit_open.load()) {
        return false;
    }

    auto elapsed = std::chrono::steady_clock::now() - impl_->circuit_opened_at;
    if (elapsed > impl_->circuit_breaker_window) {
        impl_->circuit_open.store(false);
        impl_->consecutive_failures.store(0);
        return false;
    }

    return true;
}

template<typename Func>
auto RetryHandler::execute_with_retry(Func&& func, const std::string& operation_name)
    -> std::invoke_result_t<Func> {

    if (is_circuit_open()) {
        throw std::runtime_error("Circuit breaker is open for: " + operation_name);
    }

    int attempt = 0;
    while (true) {
        try {
            auto result = func();
            impl_->consecutive_failures.store(0);
            return result;
        } catch (const std::exception& e) {
            attempt++;

            if (attempt >= impl_->config.max_attempts) {
                impl_->consecutive_failures.fetch_add(1);

                if (impl_->consecutive_failures.load() >= 5) {
                    impl_->circuit_open.store(true);
                    impl_->circuit_opened_at = std::chrono::steady_clock::now();
                }

                throw;
            }

            auto delay_ms = impl_->get_retry_delay_ms(attempt);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }
}

ErrorClassifier& ErrorClassifier::instance() {
    static ErrorClassifier classifier;
    return classifier;
}

ErrorCategory ErrorClassifier::classify_from_nccl(int nccl_result) {
#if NOVA_NCCL_ENABLED
    if (nccl_result == ncclSuccess) {
        return ErrorCategory::Unknown;
    }

    if (nccl_result == ncclUnhandledCudaError) {
        return ErrorCategory::NetworkError;
    }
    if (nccl_result == ncclRemoteError) {
        return ErrorCategory::NetworkError;
    }
    if (nccl_result == ncclTransportFailure) {
        return ErrorCategory::NetworkError;
    }
    if (nccl_result == ncclTimeoutError) {
        return ErrorCategory::Timeout;
    }
    if (nccl_result == ncclSystemError) {
        return ErrorCategory::HardwareError;
    }
    if (nccl_result == ncclInternalError) {
        return ErrorCategory::InternalError;
    }
    if (nccl_result == ncclInvalidArgument) {
        return ErrorCategory::InvalidArgument;
    }
    if (nccl_result == ncclInvalidUsage) {
        return ErrorCategory::InvalidArgument;
    }
#endif

    return ErrorCategory::Unknown;
}

ErrorSeverity ErrorClassifier::classify_severity(ErrorCategory category,
                                                 const std::string& details) {
    if (is_permanent_error(category)) {
        return ErrorSeverity::Permanent;
    }

    if (category == ErrorCategory::Timeout) {
        if (details.find("intermittent") != std::string::npos) {
            return ErrorSeverity::Transient;
        }
        return ErrorSeverity::Recoverable;
    }

    if (category == ErrorCategory::NetworkError) {
        return ErrorSeverity::Transient;
    }

    return ErrorSeverity::Recoverable;
}

bool ErrorClassifier::is_transient_error(ErrorCategory category) const {
    return is_timeout_indicative(category) || is_network_indicative(category);
}

bool ErrorClassifier::is_permanent_error(ErrorCategory category) const {
    return category == ErrorCategory::InvalidArgument ||
           category == ErrorCategory::InternalError;
}

bool ErrorClassifier::is_timeout_indicative(ErrorCategory category) const {
    return category == ErrorCategory::Timeout;
}

bool ErrorClassifier::is_network_indicative(ErrorCategory category) const {
    return category == ErrorCategory::NetworkError;
}

std::string ErrorClassifier::get_error_message(ErrorCategory category) const {
    switch (category) {
        case ErrorCategory::Timeout:
            return "Operation timed out";
        case ErrorCategory::NetworkError:
            return "Network communication error";
        case ErrorCategory::HardwareError:
            return "Hardware error detected";
        case ErrorCategory::InvalidArgument:
            return "Invalid argument passed to NCCL";
        case ErrorCategory::InternalError:
            return "Internal NCCL error";
        default:
            return "Unknown error";
    }
}

struct CommErrorRecovery::Impl {
    HealthMonitor* health_monitor = nullptr;
    RetryHandler* retry_handler = nullptr;
    ErrorClassifier* error_classifier = nullptr;
    std::chrono::seconds timeout_threshold{60};
    std::atomic<bool> initialized{false};
};

CommErrorRecovery& CommErrorRecovery::instance() {
    static CommErrorRecovery recovery;
    return recovery;
}

void CommErrorRecovery::initialize(HealthMonitor::ErrorCallback error_callback) {
    impl_->health_monitor = &HealthMonitor::instance();
    impl_->retry_handler = &RetryHandler::instance();
    impl_->error_classifier = &ErrorClassifier::instance();

    impl_->health_monitor->set_error_callback([this](const CommError& error) {
        on_comm_error(error);
    });

    impl_->health_monitor->start();
    impl_->initialized.store(true);
}

void CommErrorRecovery::shutdown() {
    if (impl_->health_monitor) {
        impl_->health_monitor->stop();
    }
    impl_->initialized.store(false);
}

void CommErrorRecovery::on_comm_error(const CommError& error) {
    if (error.is_recoverable()) {
        impl_->retry_handler->reset_circuit_breaker();
    }
}

bool CommErrorRecovery::should_retry(const CommError& error) {
    if (!error.is_recoverable()) {
        return false;
    }

    return impl_->error_classifier->is_transient_error(error.category) ||
           error.severity == ErrorSeverity::Recoverable;
}

void CommErrorRecovery::recreate_communicator(int device_id) {
    impl_->retry_handler->reset_circuit_breaker();
}

void CommErrorRecovery::set_timeout_threshold(std::chrono::seconds timeout) {
    impl_->timeout_threshold = timeout;
    if (impl_->health_monitor) {
        impl_->health_monitor->set_timeout_threshold(timeout);
    }
}

template<typename CollectiveFunc>
auto CommErrorRecovery::wrap_collective(CollectiveFunc&& func,
                                        int device_id,
                                        cudaStream_t stream,
                                        const std::string& operation_name)
    -> std::invoke_result_t<CollectiveFunc> {

    return impl_->retry_handler->execute_with_retry(
        [this, &func, device_id, stream]() {
            return func();
        },
        operation_name
    );
}

} // namespace nova::comm
