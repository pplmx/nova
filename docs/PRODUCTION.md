# Production Hardening Guide

## Overview

The Nova CUDA library provides production-ready error handling, timeout management, retry mechanisms, and graceful degradation capabilities.

## Timeout Management

### Per-Operation Timeouts

```cpp
#include "cuda/error/timeout.hpp"

using namespace nova::error;

// RAII guard for automatic timeout tracking
{
    timeout_guard guard("matrix_multiply", std::chrono::seconds{30});
    
    // Your CUDA operation here
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    
    if (guard.is_expired()) {
        // Handle timeout
    }
} // Guard automatically ends operation
```

### Watchdog Timer

The timeout manager includes a background watchdog thread that automatically detects expired operations:

```cpp
timeout_config config;
config.default_timeout = std::chrono::seconds{30};
config.watchdog_interval = std::chrono::milliseconds{100};
config.watchdog_enabled = true;

timeout_manager::instance().set_config(config);
```

### Timeout Callbacks

Register callbacks to be notified when operations time out:

```cpp
timeout_manager::instance().set_callback([](operation_id id, std::error_code ec) {
    if (ec.category() == timeout_category()) {
        std::cerr << "Operation " << id << " timed out: " << ec.message() << "\n";
    }
});
```

## Retry Mechanisms

### Exponential Backoff

```cpp
#include "cuda/error/retry.hpp"

retry_config config;
config.base_delay = std::chrono::milliseconds{100};
config.multiplier = 2.0;
config.max_delay = std::chrono::seconds{30};
config.max_attempts = 5;
config.jitter_enabled = true;

retry_executor executor(config);

auto result = executor.execute([&]() {
    auto status = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return result;
});
```

### Circuit Breaker

```cpp
circuit_breaker_config cb_cfg;
cb_cfg.failure_threshold = 5;
cb_cfg.reset_timeout = std::chrono::seconds{30};
cb_cfg.half_open_success_threshold = 3;

circuit_breaker cb(cb_cfg);

if (!cb.allow_request()) {
    // Circuit is open - skip the operation
    return fallback_result();
}

try {
    auto result = perform_operation();
    cb.record_success();
    return result;
} catch (...) {
    cb.record_failure();
    throw;
}
```

## Graceful Degradation

### Precision Levels

```cpp
#include "cuda/error/degrade.hpp"

using namespace nova::error;

// Available precision levels
enum class precision_level { high, medium, low };

// Automatically degrade precision
auto degraded = degrade(current_precision);
degradation_manager::instance().trigger_degradation(
    "matrix_multiply", degraded, "Memory pressure detected"
);
```

### Quality Thresholds

```cpp
quality_threshold threshold;
threshold.min_quality_score = 0.8;
threshold.max_retry_before_degrade = 3;
threshold.min_acceptable_precision = precision_level::medium;

degradation_manager::instance().set_threshold(threshold);
```

## Error Categories

### Timeout Errors

```cpp
std::error_code make_timeout_error(timeout_error_code code);
```

### Error Code Integration

All Nova error types integrate with `std::error_code`:

```cpp
try {
    timeout_guard guard("operation", std::chrono::seconds{10});
    // Operation
} catch (const std::system_error& e) {
    if (e.code().category() == timeout_category()) {
        // Handle timeout specifically
    }
}
```

## Best Practices

1. **Always use RAII guards** for timeout tracking
2. **Set appropriate timeouts** based on operation type and hardware
3. **Enable jitter** to prevent thundering herd on retries
4. **Configure circuit breakers** per operation type
5. **Monitor degradation events** for system health

## Configuration Defaults

| Parameter | Default Value |
|-----------|--------------|
| Default Timeout | 30 seconds |
| Watchdog Interval | 100 ms |
| Retry Base Delay | 100 ms |
| Retry Multiplier | 2.0 |
| Max Retry Attempts | 5 |
| Circuit Breaker Threshold | 5 failures |
| Reset Timeout | 30 seconds |
