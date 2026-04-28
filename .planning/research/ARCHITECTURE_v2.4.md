# Architecture: Nova CUDA Library v2.4 Production Hardening

**Domain:** Production Hardening Features
**Researched:** 2026-04-28
**Confidence:** MEDIUM-HIGH

## Executive Summary

The v2.4 production hardening adds four new architectural components that integrate with the existing five-layer architecture. All additions follow the principle of minimal invasion: they extend existing layers rather than creating new dependencies.

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NEW: Graph Executor (API Layer)                  │
│              Wraps algorithm pipelines in CUDA graphs                │
├─────────────────────────────────────────────────────────────────────┤
│                      NEW: Profiling Extensions                       │
│              NVTX domains per layer, async error tracking            │
├─────────────────────────────────────────────────────────────────────┤
│                      EXISTING: Stream Layer                          │
│                 (Extend with priority streams)                       │
├─────────────────────────────────────────────────────────────────────┤
│                      EXISTING: Memory Layer                          │
│              (Extend with L2 persistence hints)                      │
├─────────────────────────────────────────────────────────────────────┤
│                      EXISTING: Device Layer                          │
│                (Share reduce kernels, primitives)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. GraphExecutor Architecture

### 1.1 Component Responsibilities

| Responsibility | Description |
|----------------|-------------|
| Graph Capture | Convert stream-based algorithm sequences to CUDA graphs |
| Graph Instantiation | Create reusable executable graphs |
| Graph Launch | Execute graphs with proper synchronization |
| Memory Lifecycle | Track memory nodes to prevent invalidation |

### 1.2 Class Design

```cpp
// include/cuda/api/graph_executor.h
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <functional>
#include "cuda/stream/stream.h"

namespace cuda::graph {

class GraphExecutor {
public:
    using CaptureFunc = std::function<void(cudaStream_t)>;
    
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    bool is_captured_ = false;
    bool is_instantiated_ = false;
    
    // Memory nodes to prevent invalidation
    std::vector<void*> pinned_pointers_;
    std::vector<size_t> pinned_sizes_;
    
    int device_id_ = 0;

public:
    GraphExecutor();
    ~GraphExecutor();
    
    // Disable copying (GPU resources)
    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;
    
    // Allow move
    GraphExecutor(GraphExecutor&& other) noexcept;
    GraphExecutor& operator=(GraphExecutor&& other) noexcept;
    
    // Capture operations into a graph
    void beginCapture(Stream& stream);
    void addOperation(CaptureFunc&& operation);
    void endCapture();
    
    // Instantiate for repeated execution
    void instantiate();
    
    // Execute the graph
    void launch(Stream& stream);
    
    // Synchronization
    void synchronize() const;
    
    // Memory management
    void pinMemory(void* ptr, size_t size);
    template<typename T>
    void pinBuffer(const memory::Buffer<T>& buffer);
    
    // Status
    bool isValid() const { return is_instantiated_; }
    explicit operator bool() const { return isValid(); }
};

// RAII wrapper for automatic graph lifecycle
class ScopedGraph {
    std::unique_ptr<GraphExecutor> executor_;
public:
    static ScopedGraph capture(Stream& stream) {
        ScopedGraph g;
        g.executor_ = std::make_unique<GraphExecutor>();
        g.executor_->beginCapture(stream);
        return g;
    }
    
    void addOperation(CaptureFunc&& op) {
        executor_->addOperation(std::move(op));
    }
    
    GraphExecutor::CaptureFunc finalize() {
        return [executor = std::move(executor_)](cudaStream_t stream) {
            executor->instantiate();
            executor->launch(Stream::wrap(stream));
        };
    }
};
```

### 1.3 Usage Pattern

```cpp
// Example: Wrapping existing algorithms
void useGraphExecutor() {
    // Create executor
    graph::GraphExecutor executor;
    
    // Begin capture on a stream
    auto stream = stream::make_stream();
    executor.beginCapture(stream);
    
    // Add existing algorithm operations
    executor.addOperation([&](cudaStream_t s) {
        algo::reduce_async(data, output, n, s);
    });
    
    executor.addOperation([&](cudaStream_t s) {
        algo::scan_async(output, n, s);
    });
    
    // Pin any external memory used
    executor.pinMemory(external_buffer, buffer_size);
    
    // End capture
    executor.endCapture();
    
    // Instantiate for repeated use
    executor.instantiate();
    
    // Launch multiple times
    for (int i = 0; i < iterations; i++) {
        executor.launch(stream);
    }
    
    executor.synchronize();
}
```

---

## 2. L2PersistenceManager Architecture

### 2.1 Component Responsibilities

| Responsibility | Description |
|----------------|-------------|
| Budget Management | Track total persistent memory usage |
| Persistence Hints | Apply cudaAccessPolicyWindow to buffers |
| Cleanup | Remove persistence when buffers released |

### 2.2 Class Design

```cpp
// include/cuda/memory/l2_persistence.h
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>
#include <unordered_map>

namespace cuda::memory {

class L2PersistenceManager {
public:
    struct PersistedAllocation {
        void* ptr;
        size_t bytes;
        cudaAccessPolicyWindow window;
    };

private:
    size_t max_persistent_bytes_;
    size_t current_persistent_bytes_ = 0;
    std::vector<PersistedAllocation> allocations_;
    int device_id_ = 0;

public:
    explicit L2PersistenceManager(size_t max_bytes = 0);
    ~L2PersistenceManager();
    
    // Request persistence for a buffer
    bool requestPersistedAccess(void* ptr, size_t bytes);
    
    // Release persistence
    void releasePersistedAccess(void* ptr);
    
    // Query status
    size_t maxBytes() const { return max_persistent_bytes_; }
    size_t currentBytes() const { return current_persistent_bytes_; }
    float utilization() const {
        return static_cast<float>(current_persistent_bytes_) / max_persistent_bytes_;
    }
    
    // Check if persistence is possible
    bool canPersist(size_t bytes) const {
        return current_persistent_bytes_ + bytes <= max_persistent_bytes_;
    }
    
    // For working sets: update which buffers should persist
    void updateWorkingSet(const std::vector<void*>& active_ptrs);
};

}  // namespace cuda::memory
```

### 2.3 Usage Pattern

```cpp
void useL2Persistence() {
    // Create manager with 1/4 of L2 cache
    size_t l2_size;
    cudaDeviceGetL2CacheSize(&l2_size, device);
    memory::L2PersistenceManager l2{l2_size / 4};
    
    // Allocate working buffers
    Buffer<float> buffer1(1024 * 1024);
    Buffer<float> buffer2(1024 * 1024);
    
    // Request persistence (if budget allows)
    if (l2.canPersist(buffer1.size())) {
        l2.requestPersistedAccess(buffer1.data(), buffer1.size());
    }
    
    // Iterative algorithm with working set
    for (int iter = 0; iter < 100; iter++) {
        kernel1<<<..., stream>>>(buffer1.data());  // Fast: persisted in L2
        kernel2<<<..., stream>>>(buffer2.data());
        
        l2.updateWorkingSet({buffer1.data()});  // Only buffer1 persists
    }
    
    // Cleanup
    l2.releasePersistedAccess(buffer1.data());
}
```

---

## 3. PriorityStreamPool Architecture

### 3.1 Component Responsibilities

| Responsibility | Description |
|----------------|-------------|
| Priority Range Query | Handle device variations in priority support |
| Stream Pooling | Reuse stream objects to avoid resource exhaustion |
| Traffic Separation | Critical vs. batch stream isolation |

### 3.2 Class Design

```cpp
// include/cuda/stream/priority_pool.h
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace cuda::stream {

class PriorityStreamPool {
public:
    enum class Priority { High, Normal, Low };
    
private:
    struct StreamEntry {
        cudaStream_t stream;
        Priority priority;
        bool in_use = false;
    };
    
    std::vector<StreamEntry> streams_;
    int priority_min_ = 0;
    int priority_max_ = 0;
    int device_id_ = 0;
    
    // Prevent construction
    PriorityStreamPool() = default;

public:
    // Singleton access
    static PriorityStreamPool& instance();
    
    // Initialize with device
    void initialize(int device_id);
    
    // Acquire a stream of the given priority
    struct AcquiredStream {
        cudaStream_t stream;
        Priority priority;
    };
    AcquiredStream acquire(Priority priority);
    
    // Release a stream back to the pool
    void release(cudaStream_t stream);
    
    // Query priorities
    int priorityRangeMin() const { return priority_min_; }
    int priorityRangeMax() const { return priority_max_; }
    bool supportsPriorities() const { return priority_min_ != priority_max_; }
};

}  // namespace cuda::stream
```

### 3.3 Usage Pattern

```cpp
void usePriorityPool() {
    auto& pool = stream::PriorityStreamPool::instance();
    pool.initialize(0);
    
    // High-priority for latency-sensitive work
    auto critical = pool.acquire(stream::PriorityStreamPool::Priority::High);
    kernel1<<<..., critical.stream>>>();
    
    // Normal for standard operations
    auto normal = pool.acquire(stream::PriorityStreamPool::Priority::Normal);
    kernel2<<<..., normal.stream>>>();
    
    // Low-priority for background work
    auto batch = pool.acquire(stream::PriorityStreamPool::Priority::Low);
    batchKernel<<<..., batch.stream>>>();
    
    // Release when done
    pool.release(critical.stream);
    pool.release(normal.stream);
    pool.release(batch.stream);
}
```

---

## 4. AsyncErrorTracker Architecture

### 4.1 Component Responsibilities

| Responsibility | Description |
|----------------|-------------|
| Error Recording | Capture errors with operation context |
| Error Propagation | Preserve error chain through async operations |
| Diagnostics | Provide actionable error information |

### 4.2 Class Design

```cpp
// include/cuda/error/async_tracker.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <optional>

namespace cuda::error {

struct ErrorRecord {
    const char* operation_name;
    cudaError_t error;
    uint64_t timestamp_ns;
    std::optional<int> stream_id;
    std::optional<size_t> allocation_bytes;
};

class AsyncErrorTracker {
    static constexpr size_t MAX_RECORDS = 1024;
    std::vector<ErrorRecord> records_;
    uint64_t start_time_;
    
public:
    AsyncErrorTracker();
    
    // Record an operation and its potential error
    void record(const char* operation, cudaError_t result);
    
    // After a sync point, check for errors
    void checkSyncPoint(cudaStream_t stream);
    
    // Get first error encountered
    cudaError_t getFirstError() const;
    
    // Get error with most context
    const ErrorRecord* getMostInformativeError() const;
    
    // Clear records
    void reset();
    
    // Diagnostics
    size_t errorCount() const;
    void dumpRecords() const;  // For logging/debugging
};

// RAII wrapper for error tracking
class TrackedStream {
    AsyncErrorTracker& tracker_;
    cudaStream_t stream_;
    
public:
    TrackedStream(AsyncErrorTracker& tracker, Priority priority = Priority::Normal);
    ~TrackedStream();
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
};

}  // namespace cuda::error
```

### 4.3 Usage Pattern

```cpp
void useErrorTracker() {
    error::AsyncErrorTracker tracker;
    
    {
        error::TrackedStream stream{tracker};
        
        // Operations tracked automatically
        cudaMemcpyAsync(..., stream);
        kernel1<<<..., stream>>>();
        kernel2<<<..., stream>>>();
        
        // Check at sync point
        cudaStreamSynchronize(stream);
        tracker.checkSyncPoint(stream);
        
        // Get diagnostic info
        if (auto* err = tracker.getMostInformativeError()) {
            fmt::print("Error in {}: {}\n",
                err->operation_name,
                cudaGetErrorString(err->error));
        }
    }
}
```

---

## 5. Integration with Existing Architecture

### 5.1 Layer Mapping

| New Component | Integrates With | Integration Point |
|---------------|-----------------|-------------------|
| GraphExecutor | API Layer | Wraps existing algo functions |
| L2PersistenceManager | Memory Layer | Extends Buffer with persistence hints |
| PriorityStreamPool | Stream Layer | Reuses Stream class, adds priority |
| AsyncErrorTracker | Error Layer | Extends existing error framework (v1.8) |

### 5.2 CMake Integration

```cmake
# New source sets for v2.4
set(PRODUCTION_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/graph/graph_executor.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/memory/l2_persistence.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/stream/priority_pool.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/error/async_tracker.cu
)

# Add to cuda_impl library
target_sources(cuda_impl PRIVATE ${PRODUCTION_SOURCES})

# NVBench integration
FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG main
)
FetchContent_MakeAvailable(nvbench)

target_link_libraries(cuda_impl PRIVATE nvbench::nvbench)
```

### 5.3 Backward Compatibility

All v2.4 components are additive:
- Existing code continues to work unchanged
- New features are opt-in via new classes/functions
- No changes to existing API signatures
- No changes to existing error categories

---

## 6. Scalability Considerations

| Scale | Architecture Adjustments |
|-------|-------------------------|
| Single GPU | All components work as-is |
| Multi-GPU | GraphExecutor per-device; L2 manager per-device |
| Multi-Node | Stream priorities local; no cross-node coordination needed |

### Future Extension Points

1. **Distributed Graphs**: NCCL-aware graph capture for multi-GPU
2. **Persistent Threads**: Combine CUDA Graphs with persistent kernel pattern
3. **Memory Pool Integration**: L2 persistence with existing MemoryPool v2.0

---

## Sources

- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [CUDA Runtime API: Memory Pools](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)
- [CUDA Runtime API: Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
- [NVBench GitHub](https://github.com/NVIDIA/nvbench)

---

*Architecture research for: Nova CUDA Library v2.4 Production Hardening*
*Researched: 2026-04-28*
