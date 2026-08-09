#include "cuda/memory_error/memory_error_handler.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <iostream>

namespace nova::memory {

struct DeviceHealthMonitor::Impl {
    std::atomic<bool> running{false};
    std::thread monitor_thread;
    std::chrono::milliseconds check_interval{5000};
    float warning_threshold = 0.9f;
    float critical_threshold = 0.95f;
    HealthCallback health_callback;
    std::unordered_map<int, DeviceHealth> health_cache;
    mutable std::shared_mutex cache_mutex;
    std::chrono::steady_clock::time_point start_time;
};

DeviceHealthMonitor& DeviceHealthMonitor::instance() {
    static DeviceHealthMonitor monitor;
    return monitor;
}

DeviceHealthMonitor::~DeviceHealthMonitor() {
    stop_monitoring();
}

void DeviceHealthMonitor::start_monitoring() {
    if (impl_->running.load()) {
        return;
    }

    impl_->start_time = std::chrono::steady_clock::now();
    impl_->running.store(true);

    impl_->monitor_thread = std::thread([this]() {
        while (impl_->running.load()) {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);

            if (err != cudaSuccess) {
                std::this_thread::sleep_for(impl_->check_interval);
                continue;
            }

            for (int i = 0; i < device_count; ++i) {
                DeviceHealth health;
                health.device_id = i;

                cudaError_t set_err = cudaSetDevice(i);
                if (set_err != cudaSuccess) {
                    std::this_thread::sleep_for(impl_->check_interval);
                    continue;
                }

                size_t free_mem = 0, total_mem = 0;
                if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
                    std::this_thread::sleep_for(impl_->check_interval);
                    continue;
                }

                health.memory_used = total_mem - free_mem;
                health.memory_total = total_mem;

                float utilization = static_cast<float>(health.memory_used) / total_mem;
                health.utilization = utilization;

                health.is_healthy = (utilization < impl_->critical_threshold);
                health.ecc_errors = 0;

                if (utilization >= impl_->critical_threshold) {
                    health.status_message = "Critical: Memory usage above " +
                        std::to_string(static_cast<int>(utilization * 100)) + "%";
                } else if (utilization >= impl_->warning_threshold) {
                    health.status_message = "Warning: Memory usage above " +
                        std::to_string(static_cast<int>(utilization * 100)) + "%";
                } else {
                    health.status_message = "Healthy";
                }

                {
                    std::unique_lock lock(impl_->cache_mutex);
                    impl_->health_cache[i] = health;
                }

                if (impl_->health_callback) {
                    impl_->health_callback(health);
                }
            }

            std::this_thread::sleep_for(impl_->check_interval);
        }
    });
}

void DeviceHealthMonitor::stop_monitoring() {
    impl_->running.store(false);
    if (impl_->monitor_thread.joinable()) {
        impl_->monitor_thread.join();
    }
}

DeviceHealthMonitor::DeviceHealth DeviceHealthMonitor::get_health(int device_id) {
    std::shared_lock lock(impl_->cache_mutex);
    if (impl_->health_cache.count(device_id)) {
        return impl_->health_cache.at(device_id);
    }

    DeviceHealth health;
    health.device_id = device_id;
    health.is_healthy = true;
    health.memory_used = 0;
    health.memory_total = 0;
    health.utilization = 0.0f;
    health.ecc_errors = 0;
    health.status_message = "Unknown";
    return health;
}

bool DeviceHealthMonitor::is_device_healthy(int device_id) const {
    std::shared_lock lock(impl_->cache_mutex);
    if (impl_->health_cache.count(device_id)) {
        return impl_->health_cache.at(device_id).is_healthy;
    }
    return true;
}

void DeviceHealthMonitor::set_check_interval(std::chrono::milliseconds interval) {
    impl_->check_interval = interval;
}

void DeviceHealthMonitor::set_memory_threshold(float warning_threshold,
                                                float critical_threshold) {
    impl_->warning_threshold = warning_threshold;
    impl_->critical_threshold = critical_threshold;
}

void DeviceHealthMonitor::set_health_callback(HealthCallback callback) {
    impl_->health_callback = std::move(callback);
}

struct MemoryErrorHandler::Impl {
    std::atomic<int> current_tp_degree{1};
    std::atomic<int> total_errors{0};
    std::atomic<int> critical_errors{0};
    std::atomic<int> recoverable_errors{0};
    std::atomic<int> reductions_triggered{0};
    std::atomic<size_t> peak_memory_usage{0};
    std::chrono::steady_clock::time_point start_time;

    std::vector<std::function<void(const MemoryError&)>> error_callbacks;
    std::mutex callback_mutex;

    DegradationManager::DegradationLevel degradation_level =
        DegradationManager::DegradationLevel::Nominal;
};

MemoryErrorHandler& MemoryErrorHandler::instance() {
    static MemoryErrorHandler handler;
    return handler;
}

void MemoryErrorHandler::initialize() {
    impl_->start_time = std::chrono::steady_clock::now();
    DeviceHealthMonitor::instance().start_monitoring();
}

void MemoryErrorHandler::shutdown() {
    DeviceHealthMonitor::instance().stop_monitoring();
}

void MemoryErrorHandler::handle_error(const MemoryError& error) {
    impl_->total_errors.fetch_add(1);

    if (error.severity == MemoryErrorSeverity::Critical ||
        error.severity == MemoryErrorSeverity::Fatal) {
        impl_->critical_errors.fetch_add(1);
    } else {
        impl_->recoverable_errors.fetch_add(1);
    }

    if (should_reduce_parallelism(error)) {
        impl_->reductions_triggered.fetch_add(1);
    }

    {
        std::lock_guard lock(impl_->callback_mutex);
        for (auto& callback : impl_->error_callbacks) {
            callback(error);
        }
    }

    std::cerr << "[MemoryError] " << error.message
              << " (severity: " << static_cast<int>(error.severity) << ")"
              << " device: " << error.device_id
              << " CUDA error: " << error.get_cuda_error_string()
              << std::endl;
}

void MemoryErrorHandler::register_error_callback(
    std::function<void(const MemoryError&)> callback) {
    std::lock_guard lock(impl_->callback_mutex);
    impl_->error_callbacks.push_back(std::move(callback));
}

bool MemoryErrorHandler::should_reduce_parallelism(const MemoryError& error) {
    if (error.severity == MemoryErrorSeverity::Fatal) {
        return true;
    }

    if (error.category == MemoryErrorCategory::AllocationFailure) {
        return true;
    }

    if (error.category == MemoryErrorCategory::ECCError) {
        return true;
    }

    return false;
}

bool MemoryErrorHandler::should_fallback_to_cpu(const MemoryError& error) {
    if (error.severity == MemoryErrorSeverity::Fatal) {
        return true;
    }

    if (error.category == MemoryErrorCategory::MemoryCorruption) {
        return true;
    }

    if (impl_->reductions_triggered.load() >= 3) {
        return true;
    }

    return false;
}

int MemoryErrorHandler::get_recommended_tp_degree() const {
    int reductions = impl_->reductions_triggered.load();
    int current = impl_->current_tp_degree.load();

    if (reductions == 0) {
        return current;
    }

    return std::max(1, current - reductions);
}

void MemoryErrorHandler::set_current_tp_degree(int degree) {
    impl_->current_tp_degree.store(degree);
}

MemoryErrorHandler::Telemetry MemoryErrorHandler::get_telemetry() const {
    Telemetry tel;
    tel.total_errors = impl_->total_errors.load();
    tel.critical_errors = impl_->critical_errors.load();
    tel.recoverable_errors = impl_->recoverable_errors.load();
    tel.reductions_triggered = impl_->reductions_triggered.load();
    tel.peak_memory_usage = impl_->peak_memory_usage.load();
    tel.uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - impl_->start_time);
    return tel;
}

void MemoryErrorHandler::reset_telemetry() {
    impl_->total_errors.store(0);
    impl_->critical_errors.store(0);
    impl_->recoverable_errors.store(0);
    impl_->reductions_triggered.store(0);
    impl_->peak_memory_usage.store(0);
}

CudaErrorDetector& CudaErrorDetector::instance() {
    static CudaErrorDetector detector;
    return detector;
}

MemoryErrorCategory CudaErrorDetector::classify_error(cudaError_t error) {
    if (is_allocation_failure(error)) {
        return MemoryErrorCategory::AllocationFailure;
    }
    if (is_peer_access_error(error)) {
        return MemoryErrorCategory::PeerAccessError;
    }
    if (error == cudaErrorMemoryAllocation ||
        error == cudaErrorInitializationError) {
        return MemoryErrorCategory::CudaError;
    }
    return MemoryErrorCategory::CudaError;
}

MemoryErrorSeverity CudaErrorDetector::classify_severity(
    MemoryErrorCategory category, cudaError_t error) {

    if (category == MemoryErrorCategory::MemoryCorruption) {
        return MemoryErrorSeverity::Fatal;
    }

    if (category == MemoryErrorCategory::AllocationFailure) {
        if (is_out_of_memory(error)) {
            return MemoryErrorSeverity::Critical;
        }
        return MemoryErrorSeverity::Recoverable;
    }

    if (is_invalid_value(error)) {
        return MemoryErrorSeverity::Warning;
    }

    return MemoryErrorSeverity::Recoverable;
}

std::string CudaErrorDetector::get_error_description(cudaError_t error) {
    const char* err_str = cudaGetErrorString(error);
    if (err_str) {
        return std::string(err_str);
    }
    return "Unknown CUDA error";
}

bool CudaErrorDetector::is_memory_related(cudaError_t error) {
    return is_out_of_memory(error) ||
           error == cudaErrorMemoryAllocation ||
           error == cudaErrorInitializationError;
}

bool CudaErrorDetector::is_allocation_failure(cudaError_t error) {
    return is_out_of_memory(error) ||
           error == cudaErrorMemoryAllocation;
}

bool CudaErrorDetector::is_out_of_memory(cudaError_t error) {
    return error == cudaErrorLaunchOutOfResources ||
           error == cudaErrorMemoryAllocation;
}

bool CudaErrorDetector::is_peer_access_error(cudaError_t error) {
    return error == cudaErrorPeerAccessAlreadyEnabled ||
           error == cudaErrorPeerAccessNotEnabled;
}

bool CudaErrorDetector::is_invalid_value(cudaError_t error) {
    return error == cudaErrorInvalidValue ||
           error == cudaErrorInvalidDevice ||
           error == cudaErrorInvalidKernelImage;
}

struct DegradationManager::Impl {
    std::atomic<DegradationLevel> level{DegradationLevel::Nominal};
    int base_tp_degree = 1;
    float batch_reduction_factor = 1.0f;
    int recovery_attempts = 0;
};

DegradationManager& DegradationManager::instance() {
    static DegradationManager manager;
    return manager;
}

void DegradationManager::apply_degradation(DegradationLevel level) {
    impl_->level.store(level);
}

DegradationManager::DegradationLevel DegradationManager::get_current_level() const {
    return impl_->level.load();
}

void DegradationManager::reduce_tensor_parallelism(int device_count) {
    impl_->base_tp_degree = std::max(1, device_count);
    impl_->level.store(DegradationLevel::ReducedTP);
}

void DegradationManager::increase_batch_size_reduction(float factor) {
    impl_->batch_reduction_factor *= factor;
    impl_->level.store(DegradationLevel::ReducedBatch);
}

void DegradationManager::trigger_cpu_fallback() {
    impl_->level.store(DegradationLevel::CPUFallback);
}

bool DegradationManager::can_recover(DegradationLevel level) const {
    return level != DegradationLevel::Abort &&
           level != DegradationLevel::CPUFallback;
}

void DegradationManager::attempt_recovery() {
    impl_->recovery_attempts++;
    impl_->level.store(DegradationLevel::Nominal);
}

MemoryError make_memory_error(cudaError_t error, int device_id,
                              size_t attempted_size,
                              MemoryErrorCategory category) {
    MemoryError mem_error;
    mem_error.cuda_error = error;
    mem_error.device_id = device_id;
    mem_error.attempted_size = attempted_size;
    mem_error.category = category;
    mem_error.timestamp = std::chrono::steady_clock::now();

    auto& detector = CudaErrorDetector::instance();
    mem_error.severity = detector.classify_severity(category, error);
    mem_error.message = detector.get_error_description(error);

    return mem_error;
}

} // namespace nova::memory
