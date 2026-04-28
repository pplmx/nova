#include "cuda/production/async_error_tracker.h"

#include <chrono>

namespace cuda::production {

void AsyncErrorTracker::record(cudaError_t error, const std::string& context) {
    if (!enabled_ || error == cudaSuccess) {
        return;
    }

    int device_id = 0;
    cudaGetDevice(&device_id);

    auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();

    std::lock_guard<std::mutex> lock(mutex_);

    AsyncError err;
    err.error = error;
    err.message = cudaGetErrorString(error);
    err.device_id = device_id;
    err.timestamp_ns = timestamp;
    err.context = context;
    errors_.push_back(err);

    last_error_.store(error, std::memory_order_relaxed);
    error_count_.fetch_add(1, std::memory_order_relaxed);
}

void AsyncErrorTracker::record(const std::string& message, const std::string& context) {
    if (!enabled_) {
        return;
    }

    int device_id = 0;
    cudaGetDevice(&device_id);

    auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();

    std::lock_guard<std::mutex> lock(mutex_);

    AsyncError err;
    err.error = cudaErrorUnknown;
    err.message = message;
    err.device_id = device_id;
    err.timestamp_ns = timestamp;
    err.context = context;
    errors_.push_back(err);

    error_count_.fetch_add(1, std::memory_order_relaxed);
}

bool AsyncErrorTracker::check() {
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        record(error, "cudaPeekAtLastError");
        return false;
    }
    return true;
}

std::vector<AsyncError> AsyncErrorTracker::get_errors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return errors_;
}

size_t AsyncErrorTracker::error_count() const {
    return error_count_.load(std::memory_order_relaxed);
}

std::optional<AsyncError> AsyncErrorTracker::get_last_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (errors_.empty()) {
        return std::nullopt;
    }
    return errors_.back();
}

void AsyncErrorTracker::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    errors_.clear();
    last_error_.store(cudaSuccess, std::memory_order_relaxed);
    error_count_.store(0, std::memory_order_relaxed);
}

void AsyncErrorTracker::set_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
}

bool AsyncErrorTracker::is_enabled() const {
    return enabled_.load(std::memory_order_relaxed);
}

ScopedErrorTracking::ScopedErrorTracking(AsyncErrorTracker& tracker) : tracker_(tracker) {}

ScopedErrorTracking::~ScopedErrorTracking() {
    if (needs_check_) {
        tracker_.check();
    }
}

void ScopedErrorTracking::check_and_throw() {
    needs_check_ = false;
    if (!tracker_.check()) {
        auto last = tracker_.get_last_error();
        if (last.has_value()) {
            throw device::CudaException(last->error, __FILE__, __LINE__);
        }
    }
}

}  // namespace cuda::production
