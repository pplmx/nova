# Architecture Research: Production Hardening Integration

**Domain:** Nova CUDA Library v2.4 - Production Hardening
**Researched:** 2026-04-28
**Confidence:** HIGH (based on existing codebase analysis and CUDA best practices)
**For:** v2.4 Production Hardening

## Executive Summary

Production hardening for Nova CUDA Library requires integration at four key integration points: error handling enrichment, performance optimization hooks, stress testing expansion, and reliability monitoring. The existing five-layer architecture provides solid foundations, but v2.4 must add cross-cutting concerns that span all layers. This document recommends placing production hardening features in a new `production/` layer that enriches existing components rather than modifying them, preserving backward compatibility while adding fault tolerance, performance capture, and observability.

## Current State Analysis

### Existing Production Infrastructure

| Component | Location | Status | Gap for v2.4 |
|-----------|----------|--------|--------------|
| Error Handling | `cuda/error/cuda_error.hpp` | RAII guards, std::error_code | No per-algorithm recovery strategies |
| Stream Management | `cuda/stream/`, `cuda/async/` | Priority streams, StreamManager | No CUDA Graph capture/replay |
| Memory Pool | `cuda/memory/memory_pool.hpp` | Metrics, fragmentation tracking | No adaptive allocation strategies |
| Profiler | `cuda/performance/profiler.hpp` | Kernel timing, memory bandwidth | No reliability event correlation |
| Memory Error Handler | `cuda/memory_error/` | Health monitoring, degradation | No integration with algorithm layer |
| Fuzz Testing | `tests/fuzz/` | libFuzzer integration | Limited to memory and matmul |

### Five-Layer Architecture with Production Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Production Layer (NEW)                         │
│     (reliability monitoring, production hooks, health reporting)     │
│                    Depends on: all other layers                     │
├─────────────────────────────────────────────────────────────────────┤
│                          API Layer                                   │
│              (include/cuda/api/) - Public interface                 │
│                     Depends on: algo layer                          │
├─────────────────────────────────────────────────────────────────────┤
│                       Algorithm Layer                                │
│         (include/cuda/algo/) - Parallel algorithm wrappers          │
│                     Uses: device, memory, production                │
├─────────────────────────────────────────────────────────────────────┤
│                        Device Layer                                  │
│           (include/cuda/device/) - Device management                │
│              Shared: reduce kernels, warp/block primitives          │
├─────────────────────────────────────────────────────────────────────┤
│                       Memory Layer                                   │
│           (include/cuda/memory/) - Buffer, MemoryPool               │
│         Reusable by: all algorithm domains                          │
├─────────────────────────────────────────────────────────────────────┤
│                        Stream Layer                                  │
│            (include/cuda/stream/) - Async operations                │
│         Reusable by: all algorithm domains                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Error Handling Integration

### 1.1 Current Error Handling Architecture

**Existing patterns in `cuda/error/cuda_error.hpp`:**
- `cuda_error_guard` - RAII guard for error capture
- `NOVA_CHECK` macro - Throws `cuda_exception` on error
- `std::error_code` integration via `cuda_error_category`
- Recovery hints per error type

**Integration Points for v2.4:**
- Algorithm layer (enhance with recovery strategies)
- Memory layer (integrate with MemoryPool)
- Stream layer (capture stream-specific errors)

### 1.2 Recommended Patterns

#### Pattern 1: Algorithm-Specific Error Categories

```cpp
// include/cuda/production/error_categories.hpp
#pragma once

#include "cuda/error/cuda_error.hpp"
#include <system_error>

namespace nova::production {

// Error category for algorithm-specific errors
class algorithm_error_category : public std::error_category {
public:
    const char* name() const noexcept override { return "nova-algorithm"; }
    std::string message(int ev) const override;
    bool equivalent(const std::error_code& code, int condition) const noexcept override;
};

const std::error_category& algorithm_category();

// Algorithm error codes
enum class algorithm_errc {
    memory_allocation_failed = 1,
    invalid_input_size = 2,
    algorithm_timeout = 3,
    numerical_overflow = 4,
    stream_dependency_cycle = 5,
};

}  // namespace nova::production

// Extend std::error_code for algorithm errors
namespace std {
    template<>
    struct is_error_code_enum<nova::production::algorithm_errc> : true_type {};
}
```

#### Pattern 2: Recovery-Aware Result Type

```cpp
// include/cuda/production/result.hpp
#pragma once

#include <expected>
#include <variant>
#include <functional>

namespace nova::production {

enum class RecoveryStrategy {
    None,           // No recovery possible
    Retry,          // Retry with same parameters
    RetryWithBackoff,  // Retry with exponential backoff
    ReduceScale,    // Reduce batch size or parallelism
    Fallback,       // Fall back to simpler algorithm
    Degrade,        // Degrade to reduced functionality
    Abort           // Must abort, cannot continue
};

struct ErrorContext {
    cudaError_t cuda_error;
    const char* operation;
    const char* file;
    int line;
    int device_id;
    void* stream;
    std::chrono::steady_clock::time_point timestamp;
    RecoveryStrategy suggested_strategy;
    std::function<void()> recovery_action;
};

template<typename T>
class Result {
public:
    Result(T value) : variant_(std::move(value)) {}
    Result(ErrorContext error) : variant_(std::move(error)) {}
    
    bool has_value() const { return std::holds_alternative<T>(variant_); }
    T& value() { return std::get<T>(variant_); }
    const T& value() const { return std::get<T>(variant_); }
    
    bool has_error() const { return std::holds_alternative<ErrorContext>(variant_); }
    const ErrorContext& error() const { return std::get<ErrorContext>(variant_); }
    
    // Execute recovery if available
    bool attempt_recovery() {
        if (!has_error()) return true;
        const auto& ctx = error();
        if (ctx.recovery_action && ctx.suggested_strategy != RecoveryStrategy::None) {
            ctx.recovery_action();
            return true;
        }
        return false;
    }

private:
    std::variant<T, ErrorContext> variant_;
};

// Convenience macro for error-aware algorithms
#define NOVA_TRY_RESULT(expr) \
    ({ \
        auto __result = (expr); \
        if (!__result.has_value()) { \
            return __result.error(); \
        } \
        __result.value(); \
    })
```

#### Pattern 3: Algorithm-Level Error Hook

```cpp
// include/cuda/production/error_hook.hpp
#pragma once

#include <functional>
#include <vector>
#include <mutex>
#include "result.hpp"

namespace nova::production {

class ErrorHookRegistry {
public:
    using ErrorHook = std::function<void(const ErrorContext&)>;
    
    static ErrorHookRegistry& instance();
    
    void register_hook(const char* algorithm_name, ErrorHook hook);
    void unregister_hook(const char* algorithm_name);
    
    void invoke_hooks(const ErrorContext& ctx) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = hooks_.find(ctx.operation);
        if (it != hooks_.end()) {
            for (const auto& hook : it->second) {
                hook(ctx);
            }
        }
    }

private:
    ErrorHookRegistry() = default;
    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<ErrorHook>> hooks_;
};

}  // namespace nova::production
```

### 1.3 Integration with Existing Layers

**Memory Layer Integration:**
```cpp
// Extend cuda/memory/memory_pool.h

class MemoryPool {
public:
    // Existing methods...
    
    // New: Error-aware allocation with recovery
    Result<void*> allocate_with_retry(size_t bytes, int stream_id = -1,
                                      int max_retries = 3);
    
    // New: Get allocation failure statistics
    struct AllocationStats {
        size_t total_allocations;
        size_t failed_allocations;
        size_t retry_successes;
        size_t fallback_allocations;
    };
    AllocationStats get_allocation_stats() const;
};
```

**Stream Layer Integration:**
```cpp
// Extend cuda/stream/stream.h

class Stream {
public:
    // Existing constructors...
    
    // New: Error-tracking stream wrapper
    class TrackedStream {
    public:
        TrackedStream(cudaStream_t stream);
        
        cudaStream_t get() const { return stream_; }
        
        // Track last error with context
        Result<void> last_error() const;
        void record_error(cudaError_t err, const char* operation);
        
        // Stream-specific metrics
        struct StreamMetrics {
            size_t pending_ops;
            float queue_time_ms;
            size_t error_count;
        };
        StreamMetrics get_metrics() const;
        
    private:
        cudaStream_t stream_;
        ErrorContext last_error_;
        std::atomic<size_t> pending_ops_{0};
    };
};
```

## 2. Performance Optimization Patterns

### 2.1 CUDA Graphs Integration

**Recommendation:** Add optional CUDA Graph capture for repeated kernel sequences.

#### CUDA Graph Capture Pattern

```cpp
// include/cuda/production/graph_capture.hpp
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <functional>

namespace nova::production {

class GraphCapture {
public:
    struct Config {
        bool enabled = false;
        size_t warmup_iterations = 10;
        size_t capture_iterations = 5;
        size_t max_graphs = 4;  // Cache multiple graph sizes
    };
    
    GraphCapture() = default;
    explicit GraphCapture(const Config& config);
    ~GraphCapture();
    
    // Disable copying (GPU resources)
    GraphCapture(const GraphCapture&) = delete;
    GraphCapture& operator=(const GraphCapture&) = delete;
    
    // Move semantics
    GraphCapture(GraphCapture&& other) noexcept;
    GraphCapture& operator=(GraphCapture&& other) noexcept;
    
    // Begin capture - all CUDA calls until end_capture() are recorded
    void begin_capture(cudaStream_t stream);
    
    // End capture and instantiate graph
    bool end_capture();
    
    // Execute captured graph
    void execute(cudaStream_t stream);
    
    // Execute with automatic warmup and capture
    void execute_with_warmup(cudaStream_t stream);
    
    // Check if graph is available for current size
    bool has_cached_graph(size_t working_size) const;
    
    // Clear cached graphs
    void clear_cache();
    
    // Performance metrics
    struct GraphMetrics {
        bool is_valid;
        size_t node_count;
        float capture_time_ms;
        float execution_time_ms;
        float speedup_vs_direct;
    };
    GraphMetrics get_metrics() const;

private:
    Config config_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaStream_t capture_stream_ = nullptr;
    bool capturing_ = false;
    bool warmed_up_ = false;
    size_t working_size_ = 0;
    
    GraphMetrics metrics_;
};

}  // namespace nova::production
```

#### Algorithm-Level Graph Integration

```cpp
// include/cuda/production/optimized_algorithm.hpp
#pragma once

#include "graph_capture.hpp"
#include <functional>

namespace nova::production {

// Mixin for algorithms that benefit from CUDA Graph
template<typename BaseAlgorithm>
class GraphOptimizedAlgorithm : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    
    void set_graph_config(typename GraphCapture::Config config) {
        graph_config_ = config;
        graph_capture_ = std::make_unique<GraphCapture>(config);
    }
    
protected:
    // Hook for subclasses to wrap their operations
    void execute_with_graph_capture(std::function<void()> operations,
                                    cudaStream_t stream) {
        if (graph_capture_ && graph_config_.enabled) {
            if (!graph_capture_->has_cached_graph(working_size_)) {
                graph_capture_->begin_capture(stream);
                operations();
                if (graph_capture_->end_capture()) {
                    graph_capture_->execute_with_warmup(stream);
                }
            } else {
                graph_capture_->execute(stream);
            }
        } else {
            operations();
        }
    }
    
    std::unique_ptr<GraphCapture> graph_capture_;
    typename GraphCapture::Config graph_config_;
    size_t working_size_ = 0;
};

}  // namespace nova::production
```

### 2.2 Stream Priority Integration

**Existing:** `StreamManager` already supports priority streams with `get_high_priority_stream()` and `get_low_priority_stream()`.

**Enhancement:** Add priority-aware scheduling for algorithm operations.

```cpp
// include/cuda/production/stream_priority.hpp
#pragma once

#include "cuda/async/stream_manager.h"
#include <unordered_map>

namespace nova::production {

enum class OperationPriority {
    Critical = 0,    // Highest priority (lowest number)
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4   // Lowest priority
};

class PriorityScheduler {
public:
    struct Operation {
        std::function<void()> func;
        OperationPriority priority;
        const char* name;
    };
    
    explicit PriorityScheduler(int num_priorities = 5);
    
    // Submit operation to priority queue
    void submit(Operation op);
    
    // Execute all pending operations, higher priority first
    void execute_all();
    
    // Get priority range from hardware
    static cuda::async::PriorityRange get_priority_range();
    
    // Map logical priority to hardware priority
    int to_hardware_priority(OperationPriority priority);

private:
    std::vector<std::vector<Operation>> priority_queues_;
    int num_priorities_;
};

// RAII scoped priority assignment
class ScopedOperationPriority {
public:
    ScopedOperationPriority(OperationPriority priority);
    ~ScopedOperationPriority();
    
    static OperationPriority current();
    
private:
    static thread_local OperationPriority current_priority_;
};

}  // namespace nova::production
```

### 2.3 Performance Hook Integration

```cpp
// include/cuda/production/performance_hooks.hpp
#pragma once

#include "cuda/performance/profiler.h"
#include <functional>

namespace nova::production {

class PerformanceHookRegistry {
public:
    using PreOperationHook = std::function<void(const char* op_name)>;
    using PostOperationHook = std::function<void(const char* op_name, float elapsed_ms)>;
    
    static PerformanceHookRegistry& instance();
    
    void register_pre_hook(PreOperationHook hook);
    void register_post_hook(PostOperationHook hook);
    
    void invoke_pre(const char* op_name) {
        for (const auto& hook : pre_hooks_) hook(op_name);
    }
    
    void invoke_post(const char* op_name, float elapsed_ms) {
        for (const auto& hook : post_hooks_) hook(op_name, elapsed_ms);
    }

private:
    std::vector<PreOperationHook> pre_hooks_;
    std::vector<PostOperationHook> post_hooks_;
};

// Macro for automatic performance tracking
#define NOVA_PROFILE_OPERATION(op_name, operation) \
    do { \
        nova::production::PerformanceHookRegistry::instance().invoke_pre(op_name); \
        auto __start = std::chrono::steady_clock::now(); \
        operation; \
        auto __end = std::chrono::steady_clock::now(); \
        float __elapsed = std::chrono::duration<float, std::milli>(__end - __start).count(); \
        nova::production::PerformanceHookRegistry::instance().invoke_post(op_name, __elapsed); \
    } while (0)

}  // namespace nova::production
```

## 3. Stress Testing Infrastructure

### 3.1 Current Fuzz Testing Architecture

**Existing in `tests/fuzz/`:**
- `memory_pool_fuzz.cpp` - Memory allocation fuzzing
- `algorithm_fuzz.cpp` - Algorithm input fuzzing
- `matmul_fuzz.cpp` - Matrix multiplication fuzzing

**Gap for v2.4:**
- No stream-specific fuzzing
- No multi-GPU stress testing
- No error injection testing

### 3.2 Stress Testing Infrastructure Pattern

```cpp
// include/cuda/production/stress_test.hpp
#pragma once

#include <random>
#include <functional>
#include <atomic>

namespace nova::production::stress {

// Stress test configuration
struct StressConfig {
    size_t max_iterations = 10000;
    size_t timeout_seconds = 300;
    float failure_threshold = 0.01f;  // 1% failure rate threshold
    size_t memory_stress_size = 1 << 30;  // 1GB max allocation
    bool enable_error_injection = false;
    float error_injection_rate = 0.001f;  // 0.1% error injection
    size_t concurrent_operations = 4;
};

// Error injection types
enum class InjectErrorType {
    CudaError,
    MemoryFailure,
    Timeout,
    Corruption,
    StreamError
};

class ErrorInjector {
public:
    explicit ErrorInjector(float injection_rate);
    
    bool should_inject() {
        return distribution_(rng_) < injection_rate_;
    }
    
    void set_error_type(InjectErrorType type);
    cudaError_t get_injected_error();

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> distribution_;
    float injection_rate_;
    InjectErrorType error_type_ = InjectErrorType::CudaError;
};

// Stress test runner
class StressRunner {
public:
    explicit StressRunner(const StressConfig& config);
    
    // Register a stress test
    void register_test(const char* name, 
                       std::function<bool()> test_func,
                       std::function<void()> setup = nullptr,
                       std::function<void()> teardown = nullptr);
    
    // Run all registered tests
    struct StressResults {
        size_t total_runs;
        size_t total_failures;
        float failure_rate;
        std::chrono::milliseconds total_time;
        std::unordered_map<std::string, size_t> failures_by_test;
    };
    
    StressResults run();
    
    // Run with parallel workers
    StressResults run_parallel(size_t num_workers);
    
private:
    struct TestCase {
        const char* name;
        std::function<bool()> test_func;
        std::function<void()> setup;
        std::function<void()> teardown;
    };
    
    StressConfig config_;
    std::vector<TestCase> tests_;
    std::atomic<bool> should_stop_{false};
};

}  // namespace nova::production::stress
```

### 3.3 Integration with Existing Test Framework

```cpp
// tests/production/stress_test.cpp
#include "cuda/production/stress_test.hpp"
#include <gtest/gtest.h>

namespace nova {
namespace production {
namespace stress {
namespace test {

// Example: Stream stress test
class StreamStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
        config_ = StressConfig{
            .max_iterations = 1000,
            .timeout_seconds = 60,
            .concurrent_operations = 4
        };
    }
    
    StressConfig config_;
};

TEST_F(StreamStressTest, ConcurrentStreamOperations) {
    StressRunner runner(config_);
    
    runner.register_test("stream_create_destroy", []() {
        cuda::stream::Stream stream;
        return stream.get() != nullptr;
    });
    
    runner.register_test("stream_priority", []() {
        auto range = cuda::async::get_stream_priority_range();
        if (range.min_priority >= range.max_priority) return true;  // Skip if no priority support
        
        cuda::stream::Stream high_priority(range.min_priority, cudaStreamNonBlocking);
        cuda::stream::Stream low_priority(range.max_priority, cudaStreamNonBlocking);
        return high_priority.get() && low_priority.get();
    });
    
    auto results = runner.run_parallel(4);
    EXPECT_LT(results.failure_rate, config_.failure_threshold);
}

// Example: Memory pool stress test
class MemoryPoolStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
        config_ = StressConfig{
            .max_iterations = 500,
            .timeout_seconds = 120,
            .memory_stress_size = 1 << 28  // 256MB
        };
    }
    
    StressConfig config_;
};

TEST_F(MemoryPoolStressTest, AllocationDeallocationCycle) {
    StressRunner runner(config_);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1024, 1 << 20);
    
    runner.register_test("pool_alloc_dealloc", [&]() {
        cuda::memory::MemoryPool pool;
        std::vector<void*> ptrs;
        
        for (size_t i = 0; i < 100; ++i) {
            void* ptr = pool.allocate(size_dist(rng));
            if (!ptr) return false;
            ptrs.push_back(ptr);
        }
        
        for (void* ptr : ptrs) {
            pool.deallocate(ptr, 0);
        }
        return true;
    });
    
    auto results = runner.run();
    EXPECT_EQ(results.total_failures, 0);
}

}  // namespace test
}  // namespace stress
}  // namespace production
}  // namespace nova
```

### 3.4 Edge Case Coverage Map

| Edge Case Category | Current Coverage | v2.4 Enhancement |
|-------------------|------------------|------------------|
| Large allocations | Basic fuzzing | Multi-GB allocation stress |
| Concurrent streams | Missing | Stream race condition tests |
| Error propagation | Missing | Error injection framework |
| Memory fragmentation | Partial | Long-running fragmentation tests |
| Device failover | Missing | Multi-device fallback tests |
| Timeout handling | Missing | Operation timeout tests |

## 4. Reliability Monitoring Integration

### 4.1 Current Monitoring Infrastructure

**Existing:**
- `DeviceHealthMonitor` - Device health tracking
- `MemoryErrorHandler` - Error classification and recovery
- `Profiler` - Kernel timing and bandwidth
- `MemoryPool::PoolMetrics` - Pool statistics

**Gap for v2.4:**
- No unified observability layer
- No correlation between errors and performance
- No alerting infrastructure

### 4.2 Unified Monitoring Architecture

```cpp
// include/cuda/production/monitor.hpp
#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>

namespace nova::production::monitor {

// Health levels for system components
enum class HealthLevel {
    Healthy,
    Degraded,
    Critical,
    Failed
};

// Metric types
enum class MetricType {
    Counter,
    Gauge,
    Histogram,
    Duration
};

struct MetricValue {
    double value;
    std::chrono::steady_clock::time_point timestamp;
};

class MetricsCollector {
public:
    static MetricsCollector& instance();
    
    // Register a metric
    void register_metric(const std::string& name, MetricType type);
    
    // Record metric values
    void record_counter(const std::string& name, double delta = 1.0);
    void record_gauge(const std::string& name, double value);
    void record_histogram(const std::string& name, double value);
    void record_duration(const std::string& name, std::chrono::microseconds duration);
    
    // Query metrics
    std::vector<MetricValue> get_metric(const std::string& name,
                                         std::chrono::steady_clock::time_point since);
    double get_rate(const std::string& counter_name,
                    std::chrono::seconds window);

private:
    MetricsCollector() = default;
    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<MetricValue>> metrics_;
    std::unordered_map<std::string, MetricType> metric_types_;
};

// Health monitoring
class HealthMonitor {
public:
    struct ComponentHealth {
        std::string component_name;
        HealthLevel level;
        std::string message;
        std::chrono::steady_clock::time_point last_check;
    };
    
    static HealthMonitor& instance();
    
    // Register component for monitoring
    void register_component(const std::string& name,
                            std::function<ComponentHealth()> check_func);
    
    // Get health of all components
    std::vector<ComponentHealth> get_all_health();
    
    // Get overall system health
    HealthLevel get_overall_health();
    
    // Set alert callback
    using AlertCallback = std::function<void(HealthLevel, const std::string&, const std::string&)>;
    void set_alert_callback(AlertCallback callback);

private:
    struct Component {
        std::string name;
        std::function<ComponentHealth()> check_func;
        HealthLevel last_level = HealthLevel::Healthy;
    };
    
    std::mutex mutex_;
    std::vector<Component> components_;
    AlertCallback alert_callback_;
};

// Event correlation for debugging
class EventCorrelator {
public:
    struct Event {
        std::string type;
        std::string source;
        std::string message;
        std::chrono::steady_clock::time_point timestamp;
        std::unordered_map<std::string, std::string> metadata;
    };
    
    static EventCorrelator& instance();
    
    void record_event(Event event);
    
    // Find correlated events within time window
    std::vector<Event> find_correlated(const std::string& event_type,
                                        std::chrono::milliseconds window);
    
    // Get event timeline for diagnosis
    struct TimelineEntry {
        std::chrono::steady_clock::time_point time;
        std::string event_type;
        std::string summary;
        bool is_error;
    };
    std::vector<TimelineEntry> get_timeline(std::chrono::milliseconds window);

private:
    EventCorrelator() = default;
    std::mutex mutex_;
    std::vector<Event> events_;
    static constexpr size_t MAX_EVENTS = 10000;
};

}  // namespace nova::production::monitor
```

### 4.3 Integration Points for Existing Components

```cpp
// Extension: cuda/memory/memory_pool.h additions

namespace cuda::memory {

class MemoryPool {
public:
    // Existing methods...
    
    // New: Register with health monitor
    void register_with_monitor() {
        monitor::HealthMonitor::instance().register_component(
            "memory_pool",
            [this]() -> monitor::HealthMonitor::ComponentHealth {
                auto metrics = get_metrics();
                double utilization = (double)metrics.peak_allocated_bytes / 
                                    (config_.block_size * config_.max_blocks);
                
                HealthLevel level = HealthLevel::Healthy;
                std::string msg;
                
                if (utilization > 0.95) {
                    level = HealthLevel::Critical;
                    msg = "Memory pool near capacity";
                } else if (utilization > 0.8) {
                    level = HealthLevel::Degraded;
                    msg = "Memory pool under pressure";
                }
                
                return { "memory_pool", level, msg, std::chrono::steady_clock::now() };
            }
        );
    }
    
    // New: Record metrics
    void record_metrics() {
        auto metrics = get_metrics();
        auto& collector = monitor::MetricsCollector::instance();
        collector.record_gauge("memory_pool_allocated", metrics.peak_allocated_bytes);
        collector.record_counter("memory_pool_hits", metrics.hits);
        collector.record_counter("memory_pool_misses", metrics.misses);
        collector.record_gauge("memory_pool_fragmentation", metrics.fragmentation_percent);
    }
};

}  // namespace cuda::memory
```

```cpp
// Extension: cuda/stream/stream.h additions

namespace cuda::stream {

class Stream {
public:
    // Existing methods...
    
    // New: Register stream metrics
    void register_metrics(const std::string& name) {
        auto& collector = monitor::MetricsCollector::instance();
        collector.register_metric("stream_" + name + "_pending", MetricType::Gauge);
        collector.register_metric("stream_" + name + "_errors", MetricType::Counter);
        
        monitor::HealthMonitor::instance().register_component(
            "stream_" + name,
            [this, name]() -> monitor::HealthMonitor::ComponentHealth {
                HealthLevel level = HealthLevel::Healthy;
                return { "stream_" + name, level, "Stream healthy",
                        std::chrono::steady_clock::now() };
            }
        );
    }
};

}  // namespace cuda::stream
```

### 4.4 Production Layer Component Map

```
Production Layer (cuda/production/)
├── include/cuda/production/
│   ├── error_categories.hpp     # Algorithm-specific error codes
│   ├── result.hpp               # Recovery-aware Result<T> type
│   ├── error_hook.hpp           # Error callback registry
│   ├── graph_capture.hpp        # CUDA Graph capture utilities
│   ├── optimized_algorithm.hpp  # Graph optimization mixin
│   ├── stream_priority.hpp      # Priority scheduling
│   ├── performance_hooks.hpp    # Performance callbacks
│   ├── stress_test.hpp          # Stress testing framework
│   ├── monitor.hpp              # Unified monitoring
│   ├── health_monitor.hpp       # Health checking
│   └── reliability.hpp          # High-level reliability API
```

## 5. Recommended Implementation Order

### Phase 1: Error Handling Foundation (Week 1-2)
1. Create `cuda/production/error_categories.hpp`
2. Create `cuda/production/result.hpp` with `Result<T>`
3. Add `allocate_with_retry` to MemoryPool
4. Add error hook registration to Stream class

### Phase 2: Monitoring Infrastructure (Week 2-3)
1. Create `cuda/production/monitor.hpp`
2. Integrate health monitoring with DeviceHealthMonitor
3. Add MetricsCollector singleton
4. Register existing components (MemoryPool, StreamManager)

### Phase 3: Performance Optimization (Week 3-4)
1. Create `cuda/production/graph_capture.hpp`
2. Add CUDA Graph capture to priority operations
3. Create `cuda/production/stream_priority.hpp`
4. Integrate with StreamManager for priority-aware scheduling

### Phase 4: Stress Testing (Week 4-5)
1. Create `cuda/production/stress_test.hpp`
2. Add stream stress tests
3. Add memory pool fragmentation tests
4. Integrate error injection framework

### Phase 5: Integration and Polish (Week 5-6)
1. Integrate all production hooks with algorithm layer
2. Add comprehensive stress test coverage
3. Performance regression testing
4. Documentation and examples

## 6. Backward Compatibility Checklist

| Change | Compatibility Impact | Mitigation |
|--------|---------------------|------------|
| New `production/` layer | None (additive) | New namespace, no existing code affected |
| MemoryPool extensions | Low | New methods optional, existing APIs unchanged |
| Stream extensions | Low | New tracked stream is opt-in |
| Result<T> type | None | New type, replaces manual error handling where used |
| Stress test framework | None | Test-only code, not linked into library |

## Anti-Patterns to Avoid

### Anti-Pattern 1: Global Mutable State in Performance Path

**What:** Singleton monitors modifying state during kernel execution.

**Why bad:** Monitoring overhead affects production performance.

**Do this instead:** Use lock-free data structures, batch metric collection, async reporting.

### Anti-Pattern 2: Mandatory Production Overhead

**What:** Force all users to pay for production hardening features.

**Why bad:** Users who don't need production features suffer performance penalty.

**Do this instead:** All production features are opt-in via compile-time flags or runtime configuration.

### Anti-Pattern 3: Duplicate Error Handling

**What:** Production layer reimplements error handling that exists in `cuda_error.hpp`.

**Why bad:** Inconsistent behavior, maintenance burden.

**Do this instead:** Extend existing `cuda_error.hpp` with production-specific categories, don't duplicate.

### Anti-Pattern 4: Blocking Monitoring Calls

**What:** Health checks that block on device queries during critical path.

**Why bad:** Latency spikes in production workloads.

**Do this instead:** Async health checks, cached health state, separate monitoring thread.

## Sources

- NVIDIA CUDA Programming Guide - CUDA Graphs
- Existing Nova codebase patterns (`cuda/error/`, `cuda/async/`, `cuda/memory/`)
- CUDA Best Practices Guide - Performance Optimization
- libFuzzer documentation for stress testing

---

*Architecture research for: Nova CUDA Library v2.4 Production Hardening*
*Researched: 2026-04-28*
