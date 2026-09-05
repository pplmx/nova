#include "cuda/preemption/preemption_handler.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <fstream>
#include <sstream>

namespace nova::preemption {

static std::atomic<bool> g_shutdown_requested{false};
static std::atomic<int> g_received_signal{0};
static std::mutex g_signal_mutex;
static std::chrono::steady_clock::time_point g_signal_time{};

struct SignalHandler::Impl {
    bool installed = false;
    ShutdownCallback callback;
};

SignalHandler& SignalHandler::instance() {
    static SignalHandler handler;
    return handler;
}

SignalHandler::SignalHandler()
    : impl_(std::make_unique<Impl>()) {}

void SignalHandler::signal_handler(int signal) {
    {
        std::lock_guard<std::mutex> lock(g_signal_mutex);
        g_shutdown_requested.store(true, std::memory_order_relaxed);
        g_received_signal.store(signal, std::memory_order_relaxed);
        g_signal_time = std::chrono::steady_clock::now();
    }

    auto& instance = SignalHandler::instance();
    auto& impl = *instance.impl_;

    ShutdownCallback callback;
    {
        std::lock_guard<std::mutex> lock(g_signal_mutex);
        callback = impl.callback;
    }

    if (callback) {
        callback(signal);
    }
}

void SignalHandler::install_handlers() {
    if (impl_->installed) {
        return;
    }

    struct sigaction sa {};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGUSR1, &sa, nullptr);

    impl_->installed = true;
}

void SignalHandler::uninstall_handlers() {
    if (!impl_->installed) {
        return;
    }

    struct sigaction sa {};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGUSR1, &sa, nullptr);

    impl_->installed = false;
}

bool SignalHandler::is_shutdown_requested() const {
    std::lock_guard<std::mutex> lock(g_signal_mutex);
    return g_shutdown_requested.load(std::memory_order_relaxed);
}

int SignalHandler::received_signal() const {
    std::lock_guard<std::mutex> lock(g_signal_mutex);
    return g_received_signal.load(std::memory_order_relaxed);
}

void SignalHandler::set_shutdown_callback(ShutdownCallback callback) {
    impl_->callback = std::move(callback);
}

SignalHandler::HandlerState SignalHandler::get_state() const {
    std::lock_guard<std::mutex> lock(g_signal_mutex);
    HandlerState state;
    state.handler_installed = impl_->installed;
    state.shutdown_requested = g_shutdown_requested.load(std::memory_order_relaxed);
    state.received_signal_number = g_received_signal.load(std::memory_order_relaxed);
    state.signal_received_at = g_signal_time;
    return state;
}

struct ShutdownCoordinator::Impl {
    ShutdownConfig config;
    std::atomic<ShutdownPhase> phase{ShutdownPhase::Idle};
    // Atomic so a status poller racing a REUSE round (reset via shutdown() then
    // a new request_shutdown) never formally data-races on a plain time_point —
    // the reader may pair with the previous round's store, but that is a benign
    // one-poll-stale value, not UB (review MEDIUM).
    std::atomic<std::chrono::steady_clock::time_point> shutdown_start{};
    std::atomic<int64_t> timeout_remaining{30};
    std::atomic<bool> shutdown_in_progress{false};
    std::atomic<bool> shutdown_complete{false};

    ShutdownStageCallback stage_callback;
    ShutdownCoordinator::CheckpointCallback checkpoint_callback;
    std::mutex callback_mutex;

    std::condition_variable shutdown_cv;
    std::mutex shutdown_mutex;
};

ShutdownCoordinator& ShutdownCoordinator::instance() {
    static ShutdownCoordinator coordinator;
    return coordinator;
}

ShutdownCoordinator::ShutdownCoordinator()
    : impl_(std::make_unique<Impl>()) {}

void ShutdownCoordinator::initialize(const ShutdownConfig& config) {
    impl_->config = config;
    impl_->timeout_remaining.store(config.shutdown_timeout.count());
}

void ShutdownCoordinator::shutdown() {
    // Guard (RIL TASK-079, ISS-018): the old reset ran unconditionally while a
    // detached shutdown thread (begin → checkpoint → finalize) was still
    // touching phase/shutdown_complete — a data race, and PreemptionManager::
    // shutdown() could reset state under the thread that "Completes" it. Refuse
    // the reset while a shutdown is MID-FLIGHT (in progress, not yet complete);
    // a completed shutdown still resets so the singleton is reusable.
    if (impl_->shutdown_in_progress.load() && !impl_->shutdown_complete.load()) {
        return;
    }
    impl_->phase.store(ShutdownPhase::Idle);
    impl_->shutdown_in_progress.store(false);
    impl_->shutdown_complete.store(false);
}

void ShutdownCoordinator::request_shutdown(int signal) {
    if (impl_->shutdown_in_progress.load()) {
        return;
    }

    // Publish the deadline state BEFORE in_progress, so any reader that sees
    // in_progress == true also sees a fully initialized start time / timeout.
    impl_->shutdown_start.store(std::chrono::steady_clock::now());
    impl_->timeout_remaining.store(impl_->config.shutdown_timeout.count());
    impl_->shutdown_in_progress.store(true);

    std::thread([this, signal]() {
        begin_graceful_shutdown();
        checkpoint_coordinated();
        finalize_shutdown();
    }).detach();
}

void ShutdownCoordinator::begin_graceful_shutdown() {
    impl_->phase.store(ShutdownPhase::Signaling);

    if (impl_->stage_callback) {
        impl_->stage_callback(ShutdownPhase::Signaling);
    }

    std::cout << "[Preemption] Beginning graceful shutdown" << std::endl;
}

bool ShutdownCoordinator::checkpoint_coordinated() {
    impl_->phase.store(ShutdownPhase::Checkpointing);

    if (impl_->stage_callback) {
        impl_->stage_callback(ShutdownPhase::Checkpointing);
    }

    if (!impl_->config.checkpoint_on_shutdown) {
        return true;  // checkpointing not requested — nothing to do
    }

    CheckpointCallback cb;
    {
        std::lock_guard lock(impl_->callback_mutex);
        cb = impl_->checkpoint_callback;
    }

    if (!cb) {
        // RIL TASK-079, ISS-018: the old code printed "Saving checkpoint before
        // shutdown" and saved nothing. Never pretend — report the
        // misconfiguration so an operator wires PreemptionManager::
        // set_checkpoint_callback (or disables checkpoint_on_shutdown).
        std::cerr << "[Preemption] checkpoint_on_shutdown is enabled but no "
                     "checkpoint callback is registered — nothing was saved\n";
        return false;
    }

    const bool saved = cb();
    if (!saved) {
        std::cerr << "[Preemption] Checkpoint during shutdown FAILED\n";
    }
    return saved;
}

void ShutdownCoordinator::set_checkpoint_callback(CheckpointCallback callback) {
    std::lock_guard lock(impl_->callback_mutex);
    impl_->checkpoint_callback = std::move(callback);
}

void ShutdownCoordinator::finalize_shutdown() {
    impl_->phase.store(ShutdownPhase::Finalizing);

    if (impl_->stage_callback) {
        impl_->stage_callback(ShutdownPhase::Finalizing);
    }

    std::cout << "[Preemption] Finalizing shutdown" << std::endl;

    impl_->phase.store(ShutdownPhase::Complete);
    impl_->shutdown_complete.store(true);

    impl_->shutdown_cv.notify_all();
}

ShutdownPhase ShutdownCoordinator::get_phase() const {
    return impl_->phase.load();
}

bool ShutdownCoordinator::is_shutdown_in_progress() const {
    return impl_->shutdown_in_progress.load();
}

bool ShutdownCoordinator::is_shutdown_complete() const {
    return impl_->shutdown_complete.load();
}

void ShutdownCoordinator::set_stage_callback(ShutdownStageCallback callback) {
    std::lock_guard lock(impl_->callback_mutex);
    impl_->stage_callback = std::move(callback);
}

std::chrono::milliseconds ShutdownCoordinator::get_elapsed_time() const {
    if (!impl_->shutdown_in_progress.load()) {
        return std::chrono::milliseconds{0};
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - impl_->shutdown_start.load());
}

std::chrono::seconds ShutdownCoordinator::get_remaining_timeout() const {
    // RIL TASK-079, ISS-018: previously returned the static stored value and
    // never counted down — a "timeout" that never elapsed. Compute from the
    // deadline (start + configured budget, extended by extend_timeout) against
    // the real clock, clamped at zero. Idle reports the full configured budget.
    if (!impl_->shutdown_in_progress.load()) {
        return std::chrono::seconds(impl_->config.shutdown_timeout.count());
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - impl_->shutdown_start.load());
    const int64_t remaining =
        impl_->timeout_remaining.load() - elapsed.count();
    return std::chrono::seconds(remaining > 0 ? remaining : 0);
}

bool ShutdownCoordinator::extend_timeout(std::chrono::seconds additional_time) {
    impl_->timeout_remaining.store(
        impl_->timeout_remaining.load() + additional_time.count());
    return true;
}

struct ResumeValidator::Impl {
    std::string checkpoint_dir;
};

ResumeValidator& ResumeValidator::instance() {
    static ResumeValidator validator;
    return validator;
}

ResumeValidator::ResumeValidator()
    : impl_(std::make_unique<Impl>()) {}

ResumeValidator::ValidationResult ResumeValidator::validate_checkpoint(
    const std::string& checkpoint_path) {

    ValidationResult result;
    result.is_valid = false;

    std::ifstream manifest_file(checkpoint_path + "/manifest.txt");
    if (!manifest_file) {
        result.error_message = "Checkpoint manifest not found";
        return result;
    }

    std::string line;
    int step = 0;

    // Manifest layout: line 1 is checkpoint version (metadata only), line 2 is
    // the training step used below. Remaining fields are informational
    // model/optimizer metadata and are not part of validation.
    std::getline(manifest_file, line);  // skip version
    std::getline(manifest_file, line);
    step = std::stoi(line);

    manifest_file.close();

    std::ifstream model_file(checkpoint_path + "/model.bin", std::ios::binary | std::ios::ate);
    std::ifstream optimizer_file(checkpoint_path + "/optimizer.bin", std::ios::binary | std::ios::ate);

    result.has_model_state = model_file.good();
    result.has_optimizer_state = optimizer_file.good();
    result.has_rng_state = false;
    result.checkpoint_step = step;
    result.is_valid = result.has_model_state && result.has_optimizer_state;

    return result;
}

bool ResumeValidator::recover_state(const std::string& checkpoint_path) {
    auto result = validate_checkpoint(checkpoint_path);
    return result.is_valid;
}

ResumeValidator::RecoveryResult ResumeValidator::attempt_recovery(
    const std::string& checkpoint_path) {

    RecoveryResult result;
    result.success = false;

    auto validation = validate_checkpoint(checkpoint_path);

    if (!validation.is_valid) {
        result.error_message = validation.error_message;
        return result;
    }

    result.recovered_step = validation.checkpoint_step;
    result.success = true;

    return result;
}

std::string ResumeValidator::get_latest_checkpoint_path() const {
    return impl_->checkpoint_dir;
}

void ResumeValidator::set_checkpoint_dir(const std::string& dir) {
    impl_->checkpoint_dir = dir;
}

struct PreemptionManager::Impl {
    SignalHandler* signal_handler = nullptr;
    ShutdownCoordinator* shutdown_coordinator = nullptr;
    ResumeValidator* resume_validator = nullptr;

    ShutdownConfig config;
    PreemptionCallback preemption_callback;
    CheckpointCallback checkpoint_callback;
    std::atomic<bool> initialized{false};

    std::thread shutdown_waiter;
};

PreemptionManager& PreemptionManager::instance() {
    static PreemptionManager manager;
    return manager;
}

PreemptionManager::PreemptionManager()
    : impl_(std::make_unique<Impl>()) {}

void PreemptionManager::initialize(const ShutdownConfig& config) {
    impl_->config = config;
    impl_->signal_handler = &SignalHandler::instance();
    impl_->shutdown_coordinator = &ShutdownCoordinator::instance();
    impl_->resume_validator = &ResumeValidator::instance();

    impl_->shutdown_coordinator->initialize(config);

    // Re-forward any checkpoint callback registered before initialize() so the
    // coordinator's checkpoint_on_shutdown path sees it. Forwarded BY VALUE: the
    // coordinator's callback_mutex-protected slot is then the single shared
    // state, and the detached shutdown worker never reads manager state
    // unsynchronized (review MEDIUM).
    if (impl_->checkpoint_callback) {
        impl_->shutdown_coordinator->set_checkpoint_callback(
            impl_->checkpoint_callback);
    }

    impl_->signal_handler->set_shutdown_callback([this](int signal) {
        this->on_preemption_signal(signal);
    });

    impl_->signal_handler->install_handlers();
    impl_->initialized.store(true);
}

void PreemptionManager::shutdown() {
    if (!impl_->initialized.load()) {
        return;
    }

    impl_->signal_handler->uninstall_handlers();
    impl_->shutdown_coordinator->shutdown();
    impl_->initialized.store(false);
}

void PreemptionManager::on_preemption_signal(int signal) {
    std::cout << "[Preemption] Received signal " << signal << std::endl;

    if (impl_->preemption_callback) {
        impl_->preemption_callback(signal);
    }

    impl_->shutdown_coordinator->request_shutdown(signal);
}

bool PreemptionManager::is_shutdown_requested() const {
    return impl_->signal_handler->is_shutdown_requested();
}

void PreemptionManager::wait_for_shutdown() {
    while (!impl_->shutdown_coordinator->is_shutdown_complete()) {
        if (impl_->shutdown_coordinator->get_phase() == ShutdownPhase::Idle &&
            impl_->signal_handler->is_shutdown_requested()) {
            impl_->shutdown_coordinator->request_shutdown(
                impl_->signal_handler->received_signal());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool PreemptionManager::request_timeout_extension(std::chrono::seconds additional_time) {
    return impl_->shutdown_coordinator->extend_timeout(additional_time);
}

void PreemptionManager::set_preemption_callback(PreemptionCallback callback) {
    impl_->preemption_callback = std::move(callback);
}

void PreemptionManager::set_checkpoint_callback(CheckpointCallback callback) {
    impl_->checkpoint_callback = std::move(callback);
    // Forward by value (see initialize): the coordinator's mutex-protected slot
    // becomes the single shared state, so a re-registration mid-shutdown can't
    // race the detached worker's read (review MEDIUM). Forwarding nullptr lets
    // checkpoint_coordinated() report the misconfiguration instead of
    // pretending to save.
    if (impl_->shutdown_coordinator) {
        impl_->shutdown_coordinator->set_checkpoint_callback(
            impl_->checkpoint_callback);
    }
}

PreemptionManager::Status PreemptionManager::get_status() const {
    Status status;
    auto handler_state = impl_->signal_handler->get_state();

    status.preemption_handlers_installed = handler_state.handler_installed;
    status.shutdown_in_progress = impl_->shutdown_coordinator->is_shutdown_in_progress();
    status.shutdown_complete = impl_->shutdown_coordinator->is_shutdown_complete();
    status.received_signal = handler_state.received_signal_number;
    status.shutdown_elapsed = impl_->shutdown_coordinator->get_elapsed_time();
    status.remaining_timeout = impl_->shutdown_coordinator->get_remaining_timeout();

    return status;
}

} // namespace nova::preemption
