#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::production {

struct AsyncError {
    cudaError_t error;
    std::string message;
    int device_id;
    uint64_t timestamp_ns;
    std::string context;
};

class AsyncErrorTracker {
public:
    AsyncErrorTracker() = default;

    void record(cudaError_t error, const std::string& context = "");
    void record(const std::string& message, const std::string& context = "");

    bool check();

    [[nodiscard]] std::vector<AsyncError> get_errors() const;
    [[nodiscard]] size_t error_count() const;
    [[nodiscard]] std::optional<AsyncError> get_last_error() const;

    void clear();
    void set_enabled(bool enabled);
    [[nodiscard]] bool is_enabled() const;

private:
    mutable std::mutex mutex_;
    std::vector<AsyncError> errors_;
    std::atomic<bool> enabled_{true};
    std::atomic<cudaError_t> last_error_{cudaSuccess};
    std::atomic<uint64_t> error_count_{0};
};

class ScopedErrorTracking {
public:
    explicit ScopedErrorTracking(AsyncErrorTracker& tracker);
    ~ScopedErrorTracking();

    ScopedErrorTracking(const ScopedErrorTracking&) = delete;
    ScopedErrorTracking& operator=(const ScopedErrorTracking&) = delete;

    void check_and_throw();

private:
    AsyncErrorTracker& tracker_;
    bool needs_check_{true};
};

}  // namespace cuda::production
