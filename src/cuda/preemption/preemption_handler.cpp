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

void SignalHandler::signal_handler(int signal) {
    g_shutdown_requested.store(true);
    g_received_signal.store(signal);
    {
        std::lock_guard<std::mutex> lock(g_signal_mutex);
        g_signal_time = std::chrono::steady_clock::now();
    }

    auto& instance = SignalHandler::instance();
    auto& impl = *instance.impl_;

    if (impl.callback) {
        impl.callback(signal);
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
    return g_shutdown_requested.load();
}

int SignalHandler::received_signal() const {
    return g_received_signal.load();
}

void SignalHandler::set_shutdown_callback(ShutdownCallback callback) {
    impl_->callback = std::move(callback);
}

SignalHandler::HandlerState SignalHandler::get_state() const {
    HandlerState state;
    state.handler_installed = impl_->installed;
    state.shutdown_requested = g_shutdown_requested.load();
    state.received_signal_number = g_received_signal.load();
    {
        std::lock_guard<std::mutex> lock(g_signal_mutex);
        state.signal_received_at = g_signal_time;
    }
    return state;
}

struct ShutdownCoordinator::Impl {
    ShutdownConfig config;
    std::atomic<ShutdownPhase> phase{ShutdownPhase::Idle};
    std::chrono::steady_clock::time_point shutdown_start;
    std::atomic<int64_t> timeout_remaining{30};
    std::atomic<bool> shutdown_in_progress{false};
    std::atomic<bool> shutdown_complete{false};

    ShutdownStageCallback stage_callback;
    std::mutex callback_mutex;

    std::condition_variable shutdown_cv;
    std::mutex shutdown_mutex;
};

ShutdownCoordinator& ShutdownCoordinator::instance() {
    static ShutdownCoordinator coordinator;
    return coordinator;
}

void ShutdownCoordinator::initialize(const ShutdownConfig& config) {
    impl_->config = config;
    impl_->timeout_remaining.store(config.shutdown_timeout.count());
}

void ShutdownCoordinator::shutdown() {
    impl_->phase.store(ShutdownPhase::Idle);
    impl_->shutdown_in_progress.store(false);
    impl_->shutdown_complete.store(false);
}

void ShutdownCoordinator::request_shutdown(int signal) {
    if (impl_->shutdown_in_progress.load()) {
        return;
    }

    impl_->shutdown_in_progress.store(true);
    impl_->shutdown_start = std::chrono::steady_clock::now();
    impl_->timeout_remaining.store(impl_->config.shutdown_timeout.count());

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

void ShutdownCoordinator::checkpoint_coordinated() {
    impl_->phase.store(ShutdownPhase::Checkpointing);

    if (impl_->stage_callback) {
        impl_->stage_callback(ShutdownPhase::Checkpointing);
    }

    if (impl_->config.checkpoint_on_shutdown) {
        std::cout << "[Preemption] Saving checkpoint before shutdown" << std::endl;
    }
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
        std::chrono::steady_clock::now() - impl_->shutdown_start);
}

std::chrono::seconds ShutdownCoordinator::get_remaining_timeout() const {
    return std::chrono::seconds(impl_->timeout_remaining.load());
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
    int version = 0, step = 0;
    int64_t timestamp = 0;
    size_t model_count = 0, optimizer_count = 0;

    std::getline(manifest_file, line);
    version = std::stoi(line);
    std::getline(manifest_file, line);
    step = std::stoi(line);
    std::getline(manifest_file, line);
    std::getline(manifest_file, line);
    std::getline(manifest_file, line);

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
    std::atomic<bool> initialized{false};

    std::thread shutdown_waiter;
};

PreemptionManager& PreemptionManager::instance() {
    static PreemptionManager manager;
    return manager;
}

void PreemptionManager::initialize(const ShutdownConfig& config) {
    impl_->config = config;
    impl_->signal_handler = &SignalHandler::instance();
    impl_->shutdown_coordinator = &ShutdownCoordinator::instance();
    impl_->resume_validator = &ResumeValidator::instance();

    impl_->shutdown_coordinator->initialize(config);

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
