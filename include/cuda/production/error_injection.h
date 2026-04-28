#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::production {

enum class ErrorTarget {
    Allocation,
    Launch,
    Collective,
    MemoryCopy,
    Custom,
};

class ErrorInjector {
public:
    ErrorInjector() = default;

    void inject_once(ErrorTarget target, cudaError_t error);
    void inject_always(ErrorTarget target, cudaError_t error);
    void inject_random(ErrorTarget target, cudaError_t error, double probability);

    void disable();
    void enable();

    [[nodiscard]] bool should_inject(ErrorTarget target) const;
    [[nodiscard]] cudaError_t get_error(ErrorTarget target) const;

    [[nodiscard]] size_t injection_count(ErrorTarget target) const;
    [[nodiscard]] size_t total_injection_count() const;

    void increment_count(ErrorTarget target);
    void reset();

private:
    struct InjectionConfig {
        cudaError_t error = cudaSuccess;
        bool always = false;
        double probability = 0.0;
        std::atomic<size_t> count{0};
    };

    InjectionConfig configs_[5];
    std::atomic<bool> enabled_{true};
    mutable std::mt19937 rng_{std::random_device{}()};
};

class ScopedErrorInjection {
public:
    ScopedErrorInjection(ErrorInjector& injector, ErrorTarget target, cudaError_t error);
    ~ScopedErrorInjection();

    ScopedErrorInjection(const ScopedErrorInjection&) = delete;
    ScopedErrorInjection& operator=(const ScopedErrorInjection&) = delete;

private:
    ErrorInjector& injector_;
    ErrorTarget target_;
    bool was_injected_{false};
};

class MemoryPressureTest {
public:
    explicit MemoryPressureTest(size_t limit_bytes);

    [[nodiscard]] bool allocate(size_t bytes) const;
    [[nodiscard]] size_t remaining() const;
    [[nodiscard]] bool is_under_pressure() const;

    void set_limit(size_t bytes);
    void release();

private:
    size_t limit_bytes_;
    mutable std::atomic<size_t> used_bytes_{0};
};

class StressTestConfig {
public:
    size_t max_concurrent_streams = 8;
    size_t max_allocations = 1024;
    size_t allocation_size = 1024 * 1024;
    double failure_probability = 0.01;
    unsigned int timeout_seconds = 60;
};

bool run_memory_pressure_test(const StressTestConfig& config);
bool run_concurrent_stream_test(const StressTestConfig& config);

}  // namespace cuda::production
