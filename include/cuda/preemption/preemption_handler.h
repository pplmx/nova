#pragma once

#include <csignal>
#include <functional>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace nova::preemption {

enum class ShutdownPhase {
    Idle,
    Signaling,
    Checkpointing,
    Finalizing,
    Complete
};

struct ShutdownConfig {
    std::chrono::seconds shutdown_timeout{30};
    bool checkpoint_on_shutdown{true};
    bool validate_checkpoint_before_save{true};
    int max_checkpoint_retries{3};
    bool coordinated_checkpoint{true};
};

/**
 * Installs handlers for SIGTERM and SIGUSR1.
 *
 * @warning The installed handler is NOT async-signal-safe, and this is not a small caveat.
 * POSIX allows only a short list of functions to be called from a signal handler, and the
 * handler here does several things outside it:
 *
 *   - takes a std::mutex (pthread_mutex_lock is not on the list; a signal arriving while
 *     the same thread holds g_signal_mutex deadlocks the process permanently)
 *   - calls SignalHandler::instance(), whose function-local static initialisation is not
 *     async-signal-safe on first use
 *   - reads a std::chrono::steady_clock, which is not on the list
 *   - invokes a std::function<void(int)>, which may allocate or lock internally
 *   - invokes whatever the user registered, which is by definition not restricted
 *
 * A correct implementation defers all of this out of the handler: write the signal number
 * to a self-pipe or signalfd in the handler, and run the callback from a thread that reads
 * it. That is a redesign of this class, not a local edit, so it has not been attempted.
 * Until then, treat the callback as best-effort: it is reliable when the signal arrives
 * while the main thread is idle, and it can deadlock or misbehave when the signal
 * interrupts code that already holds g_signal_mutex or is inside the allocator.
 */
class SignalHandler {
public:
    static SignalHandler& instance();

    void install_handlers();
    void uninstall_handlers();

    bool is_shutdown_requested() const;
    int received_signal() const;

    using ShutdownCallback = std::function<void(int signal)>;
    void set_shutdown_callback(ShutdownCallback callback);

    struct HandlerState {
        bool handler_installed;
        bool shutdown_requested;
        int received_signal_number;
        std::chrono::steady_clock::time_point signal_received_at;
    };

    HandlerState get_state() const;

private:
    SignalHandler();
    ~SignalHandler() = default;

    static void signal_handler(int signal);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ShutdownCoordinator {
public:
    static ShutdownCoordinator& instance();

    void initialize(const ShutdownConfig& config);
    void shutdown();

    void request_shutdown(int signal);

    ShutdownPhase get_phase() const;
    bool is_shutdown_in_progress() const;
    bool is_shutdown_complete() const;

    void begin_graceful_shutdown();
    // Runs the user-registered checkpoint callback when checkpoint_on_shutdown
    // is set. Returns true only when the callback ran and reported success;
    // false when checkpointing was requested but no callback is registered or
    // the callback itself failed (never a silent "saving…" lie).
    bool checkpoint_coordinated();
    void finalize_shutdown();

    using ShutdownStageCallback = std::function<void(ShutdownPhase)>;
    void set_stage_callback(ShutdownStageCallback callback);

    // The user-supplied durable save invoked by checkpoint_coordinated().
    // Registered via PreemptionManager::set_checkpoint_callback.
    using CheckpointCallback = std::function<bool()>;
    void set_checkpoint_callback(CheckpointCallback callback);

    std::chrono::milliseconds get_elapsed_time() const;
    std::chrono::seconds get_remaining_timeout() const;

    bool extend_timeout(std::chrono::seconds additional_time);

private:
    ShutdownCoordinator();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ResumeValidator {
public:
    static ResumeValidator& instance();

    struct ValidationResult {
        bool is_valid;
        bool has_model_state;
        bool has_optimizer_state;
        bool has_rng_state;
        int checkpoint_step;
        std::string error_message;
        std::vector<std::string> warnings;
    };

    ValidationResult validate_checkpoint(const std::string& checkpoint_path);

    bool recover_state(const std::string& checkpoint_path);

    struct RecoveryResult {
        bool success;
        int recovered_step;
        std::string error_message;
    };

    RecoveryResult attempt_recovery(const std::string& checkpoint_path);

    std::string get_latest_checkpoint_path() const;

    void set_checkpoint_dir(const std::string& dir);

private:
    ResumeValidator();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class PreemptionManager {
public:
    static PreemptionManager& instance();

    void initialize(const ShutdownConfig& config);
    void shutdown();

    void on_preemption_signal(int signal);

    bool is_shutdown_requested() const;
    void wait_for_shutdown();

    bool request_timeout_extension(std::chrono::seconds additional_time);

    using PreemptionCallback = std::function<void(int signal)>;
    void set_preemption_callback(PreemptionCallback callback);

    // The durable save to run when ShutdownConfig::checkpoint_on_shutdown is
    // set. Without one, checkpoint_coordinated() reports the misconfiguration
    // instead of pretending to save. Return true only if the state was durably
    // persisted.
    using CheckpointCallback = ShutdownCoordinator::CheckpointCallback;
    void set_checkpoint_callback(CheckpointCallback callback);

    struct Status {
        bool preemption_handlers_installed;
        bool shutdown_in_progress;
        bool shutdown_complete;
        int received_signal;
        std::chrono::milliseconds shutdown_elapsed;
        std::chrono::seconds remaining_timeout;
    };

    Status get_status() const;

private:
    PreemptionManager();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nova::preemption
