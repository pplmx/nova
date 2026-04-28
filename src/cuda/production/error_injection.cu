#include "cuda/production/error_injection.h"

#include "cuda/stream/stream.h"
#include <thread>

namespace cuda::production {

void ErrorInjector::inject_once(ErrorTarget target, cudaError_t error) {
    if (static_cast<size_t>(target) < 5) {
        configs_[static_cast<size_t>(target)].error = error;
        configs_[static_cast<size_t>(target)].always = false;
        configs_[static_cast<size_t>(target)].probability = 1.0;
    }
}

void ErrorInjector::inject_always(ErrorTarget target, cudaError_t error) {
    if (static_cast<size_t>(target) < 5) {
        configs_[static_cast<size_t>(target)].error = error;
        configs_[static_cast<size_t>(target)].always = true;
        configs_[static_cast<size_t>(target)].probability = 1.0;
    }
}

void ErrorInjector::inject_random(ErrorTarget target, cudaError_t error, double probability) {
    if (static_cast<size_t>(target) < 5) {
        configs_[static_cast<size_t>(target)].error = error;
        configs_[static_cast<size_t>(target)].always = false;
        configs_[static_cast<size_t>(target)].probability = probability;
    }
}

void ErrorInjector::disable() {
    enabled_.store(false, std::memory_order_relaxed);
}

void ErrorInjector::enable() {
    enabled_.store(true, std::memory_order_relaxed);
}

bool ErrorInjector::should_inject(ErrorTarget target) const {
    if (!enabled_.load(std::memory_order_relaxed)) {
        return false;
    }

    if (static_cast<size_t>(target) >= 5) {
        return false;
    }

    const auto& config = configs_[static_cast<size_t>(target)];

    if (config.always) {
        return true;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < config.probability;
}

cudaError_t ErrorInjector::get_error(ErrorTarget target) const {
    if (static_cast<size_t>(target) >= 5) {
        return cudaSuccess;
    }
    return configs_[static_cast<size_t>(target)].error;
}

size_t ErrorInjector::injection_count(ErrorTarget target) const {
    if (static_cast<size_t>(target) >= 5) {
        return 0;
    }
    return configs_[static_cast<size_t>(target)].count.load(std::memory_order_relaxed);
}

size_t ErrorInjector::total_injection_count() const {
    size_t total = 0;
    for (int i = 0; i < 5; ++i) {
        total += configs_[i].count.load(std::memory_order_relaxed);
    }
    return total;
}

void ErrorInjector::increment_count(ErrorTarget target) {
    if (static_cast<size_t>(target) < 5) {
        configs_[static_cast<size_t>(target)].count.fetch_add(1);
    }
}

void ErrorInjector::reset() {
    for (auto& config : configs_) {
        config.error = cudaSuccess;
        config.always = false;
        config.probability = 0.0;
        config.count.store(0, std::memory_order_relaxed);
    }
    enabled_.store(true, std::memory_order_relaxed);
}

ScopedErrorInjection::ScopedErrorInjection(ErrorInjector& injector,
                                           ErrorTarget target,
                                           cudaError_t error)
    : injector_(injector), target_(target) {
    was_injected_ = injector.should_inject(target);
    if (was_injected_) {
        injector_.increment_count(target_);
    }
}

ScopedErrorInjection::~ScopedErrorInjection() {
    if (was_injected_) {
        throw device::CudaException(injector_.get_error(target_), __FILE__, __LINE__);
    }
}

MemoryPressureTest::MemoryPressureTest(size_t limit_bytes) : limit_bytes_(limit_bytes) {}

bool MemoryPressureTest::allocate(size_t bytes) const {
    size_t current = used_bytes_.load(std::memory_order_relaxed);
    if (current + bytes > limit_bytes_) {
        return false;
    }
    used_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

size_t MemoryPressureTest::remaining() const {
    size_t current = used_bytes_.load(std::memory_order_relaxed);
    return (current >= limit_bytes_) ? 0 : (limit_bytes_ - current);
}

bool MemoryPressureTest::is_under_pressure() const {
    size_t current = used_bytes_.load(std::memory_order_relaxed);
    return current >= (limit_bytes_ * 9) / 10;
}

void MemoryPressureTest::set_limit(size_t bytes) {
    limit_bytes_ = bytes;
}

void MemoryPressureTest::release() {
    used_bytes_.store(0, std::memory_order_relaxed);
}

bool run_memory_pressure_test(const StressTestConfig& config) {
    MemoryPressureTest pressure(config.max_allocations * config.allocation_size);

    std::vector<void*> allocations;

    for (size_t i = 0; i < config.max_allocations; ++i) {
        if (pressure.is_under_pressure()) {
            break;
        }

        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, config.allocation_size);

        if (err == cudaSuccess && pressure.allocate(config.allocation_size)) {
            allocations.push_back(ptr);
        } else {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    }

    bool success = !allocations.empty();

    for (auto ptr : allocations) {
        cudaFree(ptr);
    }

    return success;
}

bool run_concurrent_stream_test(const StressTestConfig& config) {
    std::vector<cuda::stream::Stream> streams;
    streams.reserve(config.max_concurrent_streams);

    for (size_t i = 0; i < config.max_concurrent_streams; ++i) {
        streams.emplace_back();
    }

    std::vector<int*> d_data(streams.size(), nullptr);

    for (size_t i = 0; i < streams.size(); ++i) {
        cudaMalloc(&d_data[i], config.allocation_size);
        cudaMemset(d_data[i], 0, config.allocation_size);
    }

    for (auto& stream : streams) {
        stream.synchronize();
    }

    for (size_t i = 0; i < streams.size(); ++i) {
        if (d_data[i]) {
            cudaFree(d_data[i]);
        }
    }

    return true;
}

}  // namespace cuda::production
