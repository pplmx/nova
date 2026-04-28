# Production Hardening Guide

**Version:** v2.4
**CUDA Requirements:** 20+

## Overview

This guide covers production hardening features added in v2.4: CUDA Graphs, L2 cache persistence, priority streams, observability, and stress testing.

## CUDA Graphs

CUDA Graphs reduce kernel launch overhead by 10-50x for batch workloads by capturing compute graphs and replaying them.

### Basic Usage

```cpp
#include "cuda/production/graph_executor.h"

cuda::production::GraphExecutor executor;
cuda::stream::Stream stream;

// Capture operations
executor.begin_capture(stream);

// Your CUDA operations here
cudaMemset(d_data, 0, size);

// Finalize graph
executor.end_capture();
executor.instantiate();

// Replay graph (fast!)
executor.launch(stream);
```

### Scoped Capture

```cpp
{
    cuda::production::ScopedCapture capture(executor, stream);
    // Operations automatically captured
}
// Graph instantiated on scope exit
```

### Memory Nodes

```cpp
cuda::production::GraphMemoryManager manager;
manager.add_device_allocation(executor, graph, size);
manager.add_host_allocation(executor, graph, size);
manager.add_managed_allocation(executor, graph, size);
```

## L2 Cache Persistence

Control L2 cache behavior for iterative algorithms working within a working set.

```cpp
#include "cuda/production/l2_persistence.h"

// RAII scoped persistence
{
    cuda::production::ScopedL2Persistence persist(bytes);

    // Your iterative kernel here
    for (int i = 0; i < iterations; ++i) {
        kernel<<<blocks, threads>>>(d_data, size);
    }
}  // L2 defaults restored

// Manual control
cuda::production::L2PersistenceManager manager;
manager.set_persistence_size(max_size / 4);
```

## Priority Streams

Priority-based stream scheduling for latency-sensitive operations.

```cpp
#include "cuda/production/priority_stream.h"

cuda::production::PriorityStreamPool pool(4);

// Acquire high-priority stream
auto high = pool.acquire(cuda::production::StreamPriority::High);
auto normal = pool.acquire(cuda::production::StreamPriority::Normal);
auto low = pool.acquire(cuda::production::StreamPriority::Low);

// Use streams...
kernel<<<..., high.get()>>>(args);

// Return to pool
pool.release(std::move(high));
```

## Observability

### NVTX Extensions

```cpp
#include "cuda/observability/nvtx_extensions.h"

// Scoped range
NOVA_NVTX_SCOPED_RANGE("my_operation");

// Or explicit push/pop
NOVA_NVTX_PUSH_RANGE(cuda::observability::NVTXDomains::Memory, "alloc");
NOVA_NVTX_POP_RANGE();
```

### Async Error Tracking

```cpp
#include "cuda/production/async_error_tracker.h"

cuda::production::AsyncErrorTracker tracker;

// After kernel launches
tracker.check();

// Access errors
if (auto err = tracker.get_last_error()) {
    log_error(err->message);
}
```

### Health Metrics

```cpp
#include "cuda/production/health_metrics.h"

cuda::production::HealthMonitor monitor;

auto json = monitor.to_json();
auto csv = monitor.to_csv();

// Record errors
monitor.record_error();
```

## Error Injection (Testing)

Fault tolerance validation without real failures.

```cpp
#include "cuda/production/error_injection.h"

cuda::production::ErrorInjector injector;
injector.inject_always(cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);

// Test recovery paths
try {
    // Operation that might fail
} catch (const cuda::device::CudaException& e) {
    // Handle gracefully
}
```

## Memory Pressure Testing

```cpp
cuda::production::StressTestConfig config;
config.max_allocations = 1024;
config.allocation_size = 10 * 1024 * 1024;  // 10MB

bool success = cuda::production::run_memory_pressure_test(config);
```

## API Reference

### GraphExecutor

| Method | Description |
|--------|-------------|
| `begin_capture(stream)` | Start capturing to graph |
| `end_capture()` | Finalize graph, return `cudaGraph_t` |
| `instantiate()` | Create executable graph |
| `launch(stream)` | Replay graph |
| `update_param(i, ptr, size)` | Update parameter without rebuild |

### L2PersistenceManager

| Method | Description |
|--------|-------------|
| `set_persistence_size(bytes)` | Set L2 cache budget |
| `restore_defaults()` | Reset to system defaults |
| `max_persistence_size()` | Get device L2 cache size |

### PriorityStreamPool

| Method | Description |
|--------|-------------|
| `acquire(priority)` | Get stream from pool |
| `release(stream)` | Return stream to pool |
| `available_count(priority)` | Count of available streams |

### AsyncErrorTracker

| Method | Description |
|--------|-------------|
| `check()` | Check for deferred errors |
| `record(error, context)` | Record an error |
| `get_errors()` | Get all recorded errors |
| `clear()` | Clear error history |

### HealthMonitor

| Method | Description |
|--------|-------------|
| `get_health_snapshot()` | Get current health metrics |
| `get_memory_snapshot()` | Get memory usage |
| `to_json()` | Export as JSON |
| `to_csv()` | Export as CSV |

## Performance Notes

1. **CUDA Graphs**: Best for static workloads with many kernel launches
2. **L2 Persistence**: Benefits iterative algorithms on small working sets
3. **Priority Streams**: Use sparingly; GPU work is not preemptible
4. **Observability**: `NOVA_NVTX_ENABLED=0` disables NVTX for production

## Migration from v2.3

No breaking changes. All v2.3 APIs remain compatible.

```diff
- #include "cuda/stream/stream.h"
+ #include "cuda/production/l2_persistence.h"
```

New features are additive and optional.
