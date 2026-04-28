# Production Hardening Pitfalls: Nova CUDA Library v2.4

**Domain:** CUDA Production Hardening (Error Handling, Performance, Reliability)
**Researched:** 2026-04-28
**Confidence:** MEDIUM-HIGH (based on NVIDIA documentation and established patterns)

## Executive Summary

This document catalogs pitfalls specific to production hardening features for v2.4: CUDA Graphs, async error handling, L2 cache persistence, stream priorities, and observability integration. Unlike algorithm correctness pitfalls, these focus on deployment, monitoring, and fault tolerance.

---

## 1. CUDA Graphs Pitfalls

### 1.1 Graph Invalidation on Resource Change

**What goes wrong:** Instantiated graphs become invalid when memory they reference is freed or reallocated, causing undefined behavior on launch.

**Why it happens:** CUDA graphs capture the DAG structure at instantiation time. If pointers passed to graph nodes are invalidated (freed, memory pool reused), the graph launches with garbage pointers.

```cpp
// DANGEROUS: Graph references pool-allocated memory
void dangerous() {
    auto buffer = memory_pool.allocate(1024);
    auto graph = captureGraph([&]{
        kernel<<<..., stream>>>(buffer.data());  // Captured pointer
    });
    // buffer goes out of scope, pointer invalidates
    cudaGraphLaunch(graph.exec, stream);  // UNDEFINED BEHAVIOR
}
```

**Consequences:**
- Silent data corruption
- Device memory errors (cudaErrorInvalidDevicePointer)
- Random failures that appear intermittent

**Prevention:**
```cpp
// SAFE: Use graph memory nodes for allocation lifecycle
class GraphExecutor {
    cudaGraph_t graph_;
    cudaGraphExec_t exec_;
    std::vector<cudaGraphNode_t> mem_nodes_;  // Track allocations
    
public:
    void addMemoryAllocation(Buffer& buffer) {
        cudaGraphNode_t alloc_node;
        cudaMemAllocNodeParams params = {};
        params.bytesize = buffer.size();
        params.poolProps.allocType = cudaMemAllocationTypePinned;
        params.poolProps.location.type = cudaMemLocationTypeDevice;
        params.poolProps.location.id = device_id_;
        
        cudaGraphAddMemAllocNode(&alloc_node, graph_, {}, 0, &params);
        cudaGraphAddMemFreeNode(&free_node, graph_, {alloc_node}, 1, 
                                &params.dptr);
        mem_nodes_.push_back(alloc_node);
    }
    
    // Alternative: Pin memory and never free during graph lifetime
    void addPinnedMemory(void* ptr, size_t size) {
        // Memory must remain valid for graph lifetime
        pinned_ptrs_.push_back({ptr, size});
    }
};
```

**Detection:** Enable CUDA API checking via `cudaSetDebugSync(true)` or `CUDA_API_CHECK=1`.

**Phase Recommendation:** Phase 1 - Define memory lifecycle management for graph nodes.

---

### 1.2 Host Callback Incompatibility

**What goes wrong:** Graphs do not support all host callback patterns available with regular streams.

**Why it happens:** `cudaStreamAddCallback()` is not supported in stream capture mode. Some callback-based patterns (e.g., profiling callbacks, progress tracking) cannot be captured.

**Consequences:**
- Capture fails silently with `cudaErrorStreamCaptureInvalidated`
- Profiling instrumentation missing from graph execution
- Progress callbacks not triggered

**Prevention:**
```cpp
// CHECK: Before capture
bool isCapturable(const cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    cudaStreamGetCaptureInfo(stream, nullptr, &status, nullptr, nullptr);
    return status == cudaStreamCaptureStatusNone;
}

// USE: Alternative patterns for graph execution
class GraphExecutor {
    // Instead of callbacks, use events
    cudaEvent_t completion_event_;
    
public:
    // Post-launch synchronization
    void launchAndWait(cudaStream_t stream) {
        cudaGraphLaunch(exec_, stream);
        cudaEventRecord(completion_event_, stream);
        cudaEventSynchronize(completion_event_);
    }
    
    // Or: CPU-side polling
    bool isComplete() {
        return cudaEventQuery(completion_event_) == cudaSuccess;
    }
};
```

**Phase Recommendation:** Phase 1 - Audit callback usage and convert to event-based patterns.

---

### 1.3 Conditional Node Complexity

**What goes wrong:** Conditional graph nodes (IF/WHILE/SWITCH) have complex constraints that cause runtime errors.

**Why it happens:** Conditional nodes require:
- All paths to produce the same memory access pattern
- Valid node handles for both branches
- Proper stream capture synchronization

**Consequences:**
- `cudaErrorInvalidValue` on graph instantiation
- Undefined behavior when condition is true at wrong time
- Memory allocation mismatches between branches

**Prevention:**
```cpp
// SIMPLE: Prefer parameterizable graphs over conditionals
// Instead of IF(condition) { kernelA } else { kernelB }
// Use: kernelA with conditional logic inside

__global__ void conditionalKernel(bool condition, ...) {
    if (condition) {
        // Path A
    } else {
        // Path B
    }
}

// For complex branching, consider multiple graph instantiations
std::unordered_map<Config, cudaGraphExec_t> compiled_graphs_;
cudaGraphExec_t getGraph(const Config& config) {
    auto it = compiled_graphs_.find(config);
    if (it != compiled_graphs_.end()) return it->second;
    return compileForConfig(config);
}
```

**Recommendation:** Defer conditional nodes to later phases. Start with static graphs.

---

## 2. Async Error Handling Pitfalls

### 2.1 Error Masking in Async Operations

**What goes wrong:** CUDA errors from kernel launches are asynchronous and don't appear until a synchronization point.

**Why it happens:** `cudaKernelLaunch()` returns immediately. The actual error (out-of-resources, invalid configuration) is deferred until `cudaStreamSynchronize()`, `cudaEventSynchronize()`, or `cudaGetLastError()`.

```cpp
// MASKED: Error goes unnoticed
void masked() {
    cudaKernel<<<1024*1024, 1024>>>();  // Too many blocks!
    // Returns cudaSuccess immediately
    doOtherWork();  // Continues while kernel fails
    // Error not detected until much later
}

// DETECTED: Check after synchronization
void detected() {
    cudaKernel<<<1024*1024, 1024>>>();
    cudaStreamSynchronize(stream);  // Error surfaces here
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaError(err);  // Now caught!
    }
}
```

**Consequences:**
- Failures attributed to wrong operation
- Memory leaks from continued execution
- Corrupted state from partially-completed work

**Prevention:**
```cpp
// PRODUCTION: Wrapper that checks async errors
class CheckedStream {
    cudaStream_t stream_;
    
public:
    template<typename Func>
    cudaError_t execute(Func&& f) {
        cudaError_t err = f();
        if (err != cudaSuccess) return err;
        
        // Synchronize to catch async errors
        err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) return err;
        
        return cudaSuccess;
    }
    
    // Alternative: Lazy error checking with error callback
    void launch(const void* kernel, dim3 grid, dim3 block, ...) {
        cudaLaunchKernel(kernel, grid, block, ..., stream_);
        // Schedule error check for after launch
        scheduleAsyncErrorCheck();
    }
};

// Per-project policy
#if defined(PRODUCTION_BUILD)
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) [[unlikely]] \
                throw cuda::CudaException(err); \
        } while(0)
#else
    #define CUDA_CHECK(call) call
#endif
```

**Phase Recommendation:** Phase 3 - Implement async-aware error checking layer.

---

### 2.2 Lost Error Context

**What goes wrong:** Multiple operations queue errors, and only the last one is retrievable.

**Why it happens:** `cudaGetLastError()` returns the most recent error. If operation A fails and operation B succeeds, the error from A is lost.

```cpp
// LOST ERROR
void lostError() {
    badKernel<<<...>>>();    // Fails with cudaErrorInvalidConfiguration
    goodKernel<<<...>>>();   // Succeeds, clears last error
    auto err = cudaGetLastError();  // Returns cudaSuccess!
    // The real error is lost
}
```

**Prevention:**
```cpp
// CAPTURE: Save errors at each operation
struct OperationResult {
    const char* operation_name;
    cudaError_t error;
    uint64_t timestamp;
};

class ErrorTracker {
    std::vector<OperationResult> results_;
    cudaStream_t stream_;
    
public:
    void record(const char* name, cudaError_t err) {
        results_.push_back({name, err, now_ns()});
    }
    
    cudaError_t getFirstError() const {
        for (const auto& r : results_) {
            if (r.error != cudaSuccess) return r.error;
        }
        return cudaSuccess;
    }
    
    // Print diagnostic on failure
    void dumpErrors() const {
        for (const auto& r : results_) {
            if (r.error != cudaSuccess) {
                fmt::print("Error at {}: {}\n", 
                    r.operation_name, cudaGetErrorString(r.error));
            }
        }
    }
};
```

---

## 3. L2 Cache Persistence Pitfalls

### 3.1 Persistence Window Exhaustion

**What goes wrong:** L2 persistence set-aside memory is limited. Setting persistence on too many allocations causes `cudaErrorMemoryAllocation`.

**Why it happens:** L2 persistence reduces available cache for normal operations. GPU has a configurable set-aside size (typically 0-32MB). All persistent accesses compete for this limited resource.

```cpp
// EXHAUSTED: Too many persistent allocations
void exhausted() {
    std::vector<Buffer> persistents;
    for (int i = 0; i < 100; i++) {
        Buffer buf(1024 * 1024);  // 1MB each = 100MB!
        setPersistence(buf);  // Eventually fails
    }
}
```

**Consequences:**
- `cudaErrorMemoryAllocation` on late allocations
- Performance degradation as some data becomes non-persistent
- Unpredictable behavior based on allocation order

**Prevention:**
```cpp
// MANAGED: Limit total persistent memory
class L2PersistenceManager {
    size_t max_persistent_bytes_;
    size_t current_persistent_bytes_ = 0;
    
public:
    bool requestPersistence(size_t bytes) {
        if (current_persistent_bytes_ + bytes > max_persistent_bytes_) {
            return false;  // Deny request
        }
        current_persistent_bytes_ += bytes;
        return true;
    }
    
    void releasePersistence(size_t bytes) {
        current_persistent_bytes_ -= bytes;
    }
    
    // For iterative algorithms, keep only active working set
    void updateWorkingSet(const std::vector<Buffer*>& active) {
        // Release non-active, allocate for new active
    }
};

// Usage
L2PersistenceManager l2_manager(
    /* max bytes = L2 cache size * 0.25, typical default */
    queryL2CacheSize() / 4
);
```

---

### 3.2 Persistence on Wrong Data

**What goes wrong:** Setting persistence on read-once data wastes the limited L2 persistence budget.

**Why it happens:** Persistence is beneficial for data accessed repeatedly. Setting it on single-use data (e.g., input that won't be reused) reduces effective cache size.

**Prevention:**
```cpp
// PROFILE: Before enabling persistence
bool shouldPersist(const Buffer& buffer, const Workload& w) {
    // Access count estimation
    if (w.iterations <= 1) return false;  // Single use
    if (w.temporalLocality < 0.5f) return false;  // Poor reuse
    
    // Size threshold
    if (buffer.size() > l2_size_ / 4) return false;  // Too large
    
    return true;
}

// Alternative: Make persistence opt-in
struct BufferOptions {
    bool enable_l2_persistence = false;
};

void allocate(const BufferOptions& opts) {
    if (opts.enable_l2_persistence) {
        l2_manager.requestPersistence(size());
        // Set access policy
    }
}
```

---

## 4. Stream Priority Pitfalls

### 4.1 Priority Inversion

**What goes wrong:** Low-priority streams can block high-priority streams due to hardware scheduling.

**Why it happens:** GPU has limited resources. Low-priority work can occupy SMs when high-priority work is queued but waiting for resources.

```cpp
// INVERSION: Low priority blocks high priority
void inversion() {
    cudaStream_t low, high;
    cudaStreamCreateWithPriority(&low, ...);
    cudaStreamCreateWithPriority(&high, ...);  // Higher priority
    
    // Low priority kernel starts and takes all SMs
    lowPriorityKernel<<<..., low>>>();
    
    // High priority kernel queued but can't start
    highPriorityKernel<<<..., high>>>();  // Waits
    
    // Even with priority, can't preempt in-flight kernel
}
```

**Consequences:**
- Latency-sensitive work delayed
- Priority-based scheduling not guaranteeing preemptability
- Unpredictable latency for "high-priority" operations

**Prevention:**
```cpp
// DESIGN: Priority streams for scheduling, not preemption
// GPU work is not preemptible, so priority affects:
// 1. Order of launch when multiple pending
// 2. Queue position when resources contested

// Better approach: Separate time-critical from batch work
class StreamPool {
    cudaStream_t critical_stream_;  // High priority
    std::vector<cudaStream_t> batch_streams_;  // Low priority
    
public:
    // Use critical for latency-sensitive kernels
    cudaStream_t critical() { return critical_stream_; }
    
    // Batch work gets dedicated streams
    cudaStream_t getBatchStream(int idx) {
        return batch_streams_[idx % batch_streams_.size()];
    }
};

// Or: Don't use priority at all, use separate queues
// High priority kernels get their own SM reservation
```

---

### 4.2 Priority Range Query Failure

**What goes wrong:** Querying priority range returns unexpected values or errors.

**Why it happens:** Not all GPUs support stream priorities. Query can fail or return degenerate ranges.

**Prevention:**
```cpp
// ROBUST: Check priority support
std::pair<int, int> getStreamPriorityRange() {
    int low, high;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&low, &high);
    if (err != cudaSuccess) {
        // Priority not supported on this device
        return {0, 0};  // Treat as single priority level
    }
    // Note: low < high on most systems, but not guaranteed
    return {std::min(low, high), std::max(low, high)};
}

// SAFE creation
cudaStream_t createPriorityStream(bool high_priority) {
    auto [min_pri, max_pri] = getStreamPriorityRange();
    if (min_pri == max_pri) {
        // No priority support, create normal stream
        return createStream();
    }
    
    int priority = high_priority ? max_pri : min_pri;
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    return stream;
}
```

---

## 5. Observability Pitfalls

### 5.1 NVTX Overhead

**What goes wrong:** Excessive NVTX annotations cause measurable performance degradation.

**Why it happens:** NVTX calls have overhead (string hashing, timestamp capture). Fine-grained ranges every kernel can add 5-10% overhead.

**Prevention:**
```cpp
// COMPILE-TIME TOGGLE
#if defined(NVTX_ENABLED)
    #define NVTX_SCOPED_RANGE(name) nvtxScopedRange __nvtx__(name)
    #define NVTX_MARK(name) nvtxMark(name)
#else
    #define NVTX_SCOPED_RANGE(name) ((void)0)
    #define NVTX_MARK(name) ((void)0)
#endif

// Or: Sampling-based profiling
void profile_sample() {
    // Only annotate every Nth iteration
    static thread_local int counter = 0;
    if (++counter % 1000 == 0) {
        NVTX_MARK("sampled_iteration");
    }
}
```

---

### 5.2 Missing Error Context in Traces

**What goes wrong:** NVTX traces show failures but not why or which operation caused them.

**Prevention:**
```cpp
// ENRICHED: Attach error context to ranges
class TracedStream {
    nvtxDomainHandle_t domain_;
    cudaStream_t stream_;
    
public:
    void launch(const char* name, ...) {
        nvtxRangePushA(domain_, name);
        nvtxPayloadPush_int32(domain_, getCurrentSize());
        
        cudaLaunchKernel(...);
        
        // Record expected completion
        nvtxMarkEx(domain_, name, cudaEvent_t completion_event);
        
        nvtxRangePop(domain_);
    }
};
```

---

## 6. Stress Testing Pitfalls

### 6.1 Error Injection False Positives

**What goes wrong:** Error injection tests pass but miss real failure modes.

**Why it happens:** Only injecting a few error codes, not covering all failure paths.

**Prevention:**
```cpp
// COMPREHENSIVE: Test all error codes
void testAllCudaErrors() {
    for (int e = 0; e < cudaErrorApiFailureBase; e++) {
        cudaError_t err = static_cast<cudaError_t>(e);
        const char* name = cudaGetErrorName(err);
        
        if (name == nullptr) continue;  // Unknown error
        
        // Only test meaningful errors
        if (isRecoverable(err)) {
            testRecovery(err);
        }
    }
}

// Cover both expected and unexpected errors
void chaosTesting() {
    // Simulate ECC memory errors
    injectEccError(device_ptr, byte_offset);
    
    // Simulate device reset
    simulateDeviceReset();
    
    // Simulate Xid error (NVIDIA driver error)
    injectXidError(45);  // Generic error from Xid
    
    // Verify graceful degradation
}
```

---

## Summary: Phase Recommendations

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Phase 1: CUDA Graphs | Memory invalidation | Graph memory nodes or pinned memory |
| Phase 1: CUDA Graphs | Callback incompatibility | Event-based alternatives |
| Phase 2: L2 Cache | Persistence exhaustion | L2PersistenceManager with limits |
| Phase 2: Stream Priorities | Priority inversion | Separate critical from batch work |
| Phase 3: Async Errors | Error masking | CheckedStream with sync |
| Phase 3: Observability | NVTX overhead | Compile-time toggle |
| Phase 4: Stress Testing | False negatives | Comprehensive error injection |

---

## Sources

- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) - HIGH
- [CUDA Programming Guide: L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management) - HIGH
- [CUDA Programming Guide: Stream Priorities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-priorities) - HIGH
- [NVTX Documentation](https://nvidia.github.io/NVTX/) - MEDIUM
- [CUDA Runtime API: Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html) - HIGH

---

*Research for: Nova CUDA Library v2.4 Production Hardening*
*Researched: 2026-04-28*
