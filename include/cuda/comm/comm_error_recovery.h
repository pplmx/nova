#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#include <atomic>
#include <chrono>
#include <thread>
#include <optional>
#include <cuda_runtime.h>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace nova::comm {

    enum class ErrorSeverity { Recoverable, Transient, Permanent };

    enum class ErrorCategory { Timeout, NetworkError, HardwareError, InvalidArgument, InternalError, Unknown };

    struct CommError {
        ErrorCategory category;
        ErrorSeverity severity;
        std::string message;
        int device_id;
        cudaStream_t stream;
        std::chrono::steady_clock::time_point timestamp;

        bool is_recoverable() const { return severity == ErrorSeverity::Recoverable || severity == ErrorSeverity::Transient; }
    };

    class HealthMonitor {
    public:
        static HealthMonitor& instance();

        void start();
        void stop();
        bool is_running() const;

        void register_comm(void* comm_id, cudaStream_t stream);
        void unregister_comm(void* comm_id);

        void set_timeout_threshold(std::chrono::seconds timeout);
        void set_check_interval(std::chrono::milliseconds interval);
        void set_stall_confirmations(int count);

        using ErrorCallback = std::function<void(const CommError&)>;
        void set_error_callback(ErrorCallback callback);

        struct HealthStatus {
            void* comm_id;
            bool is_healthy;
            bool is_stalled;
            std::chrono::milliseconds last_progress;
        };

        std::optional<HealthStatus> get_status(void* comm_id) const;

    private:
        HealthMonitor() = default;
        ~HealthMonitor();
        HealthMonitor(const HealthMonitor&) = delete;
        HealthMonitor& operator=(const HealthMonitor&) = delete;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class RetryHandler {
    public:
        static RetryHandler& instance();

        struct RetryConfig {
            int max_attempts = 3;
            std::chrono::milliseconds initial_delay {100};
            std::chrono::milliseconds max_delay {10000};
            double backoff_multiplier = 2.0;
            bool use_jitter = true;
        };

        struct State {
            RetryConfig config;
            std::atomic<int> consecutive_failures {0};
            std::atomic<bool> circuit_open {false};
            std::chrono::steady_clock::time_point circuit_opened_at;
            std::chrono::seconds circuit_breaker_window {60};
            std::default_random_engine rng {std::random_device {}()};

            void record_success() { consecutive_failures.store(0); }

            void record_failure() {
                consecutive_failures.fetch_add(1);
                if (consecutive_failures.load() >= 5) {
                    circuit_open.store(true);
                    circuit_opened_at = std::chrono::steady_clock::now();
                }
            }

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

        void set_config(const RetryConfig& config);
        RetryConfig get_config() const;

        template <typename Func>
        auto execute_with_retry(Func&& func, const std::string& operation_name) -> std::invoke_result_t<Func> {
            // A template's definition must be reachable from every call site, so it
            // lives here rather than in the .cpp (where it could never be
            // instantiated: "declared using local type ... used but never defined").
            if (is_circuit_open()) {
                throw std::runtime_error("Circuit breaker is open for: " + operation_name);
            }

            int attempt = 0;
            while (true) {
                try {
                    auto result = func();
                    impl_->record_success();
                    return result;
                } catch (const std::exception&) {
                    ++attempt;
                    if (attempt >= impl_->config.max_attempts) {
                        impl_->record_failure();
                        throw;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(impl_->get_retry_delay_ms(attempt)));
                }
            }
        }

        void reset_circuit_breaker();
        bool is_circuit_open() const;

    private:
        // State is complete before the template above, so the template can use it.
        RetryHandler() = default;

        std::unique_ptr<State> impl_;
    };

    class ErrorClassifier {
    public:
        static ErrorClassifier& instance();

        ErrorCategory classify_from_nccl(int nccl_result);
        ErrorSeverity classify_severity(ErrorCategory category, const std::string& details);

        bool is_transient_error(ErrorCategory category) const;
        bool is_permanent_error(ErrorCategory category) const;

        std::string get_error_message(ErrorCategory category) const;

    private:
        ErrorClassifier() = default;

        bool is_timeout_indicative(ErrorCategory category) const;
        bool is_network_indicative(ErrorCategory category) const;
    };

    class CommErrorRecovery {
    public:
        static CommErrorRecovery& instance();

        void initialize(HealthMonitor::ErrorCallback error_callback);
        void shutdown();

        void on_comm_error(const CommError& error);
        bool should_retry(const CommError& error);

        void recreate_communicator(int device_id);

        template <typename CollectiveFunc>
        auto wrap_collective(CollectiveFunc&& func, int device_id, cudaStream_t stream, const std::string& operation_name) -> std::invoke_result_t<CollectiveFunc> {
            // Same reason as execute_with_retry: the definition has to be visible.
            (void)device_id;
            (void)stream;
            return RetryHandler::instance().execute_with_retry([&func]() { return func(); }, operation_name);
        }

        void set_timeout_threshold(std::chrono::seconds timeout);

    private:
        CommErrorRecovery() = default;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace nova::comm
