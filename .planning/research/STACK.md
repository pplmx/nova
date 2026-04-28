# Technology Stack: Nova CUDA Library v2.4 Production Hardening

**Project:** Nova CUDA Library
**Version:** v2.4 Production Hardening
**Researched:** 2026-04-28
**Confidence:** HIGH (based on official NVIDIA documentation and established patterns)

## Executive Summary

This v2.4 milestone focuses on NEW production hardening features that complement the existing infrastructure (error framework v1.8, profiling v1.6-1.7, fuzz testing v2.0). The primary additions are CUDA Graphs for batch workload optimization, enhanced observability hooks, L2 cache persistence controls, and CUDA-native benchmarking.

---

## 1. Recommended New Technologies

### 1.1 CUDA Graphs (Primary Addition)

**Why needed:** Reduces host-side kernel launch overhead by 10-50x for repeated workloads. Critical for production batch processing.

| Component | Version | Purpose | Why |
|-----------|---------|---------|-----|
| CUDA Graphs API | CUDA 10+ | Capture/replay compute graphs | Eliminates per-kernel launch overhead |
| Stream Capture | Built-in | Convert existing streams to graphs | Non-invasive integration with current async model |

**Key capabilities:**
- `cudaGraphCreate()`, `cudaGraphInstantiate()`, `cudaGraphLaunch()`
- Conditional nodes (IF/WHILE/SWITCH) for dynamic workloads
- Graph memory nodes for stream-ordered allocation in graphs
- Device graph launch for nested execution

**Integration point:** The existing five-layer architecture can add a `GraphExecutor` in the API layer that wraps algorithm pipelines.

```cpp
// Example: Wrapping existing algorithms in a graph
class GraphExecutor {
    cudaGraph_t graph_;
    cudaGraphExec_t executable_;
public:
    template<typename... Ops>
    void capture(Stream& stream, Ops&&... ops) {
        cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal);
        (ops.execute(stream), ...);  // Existing algorithm calls
        cudaStreamEndCapture(stream.get(), &graph_);
        cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0);
    }
    void launch(Stream& stream) {
        cudaGraphLaunch(executable_, stream.get());
    }
};
```

**Source:** [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

### 1.2 NVBench (Benchmarking Library)

**Why needed:** Google Benchmark (v1.7) handles end-to-end benchmarks. NVBench provides GPU kernel microbenchmarking with NVIDIA-specific features.

| Component | Version | Purpose | Why |
|-----------|---------|---------|-----|
| NVBench | 1.x | GPU kernel microbenchmarking | Official NVIDIA benchmarking library |
| nvbench::cli | Built-in | Runtime configuration | Parameter sweeps without recompilation |

**Why not just extend Google Benchmark:** NVBench has built-in support for L2 cache management, thermal throttling detection, memory bandwidth measurement, and GPU-specific measurement types (cold vs. batch measurements).

**Integration:** Install via CMake FetchContent or as a submodule alongside existing benchmark infrastructure.

**Source:** [NVIDIA/nvbench GitHub](https://github.com/NVIDIA/nvbench)

### 1.3 L2 Cache Persistence

**Why needed:** Control cache behavior for working sets that should persist across kernel launches.

| Component | API | Purpose | Why |
|-----------|-----|---------|-----|
| cudaDeviceSetL2CacheEnabled() | CUDA 11+ | Enable L2 persisting accesses | Keep working set in cache |
| cudaAccessPolicyWindow | CUDA 11+ | Per-pointer persistence hints | Granular control |

**Use case:** After allocating memory for an algorithm's working set, prefetch to GPU and set persisting access. The data stays in L2 across multiple kernel launches.

```cpp
// Example: Persisting L2 access for iterative algorithms
void enablePersistence(DeviceBuffer& buffer) {
    cudaAccessPolicyWindow window = {
        .base_ptr = buffer.data(),
        .num_bytes = buffer.size(),
        .hitProp = cudaAccessPropertyPersisting,
        .missProp = cudaAccessPropertyAlways,
        .hitRatio = 1.0  // 100% persisting
    };
    cudaLaunchKernelSetAccessPolicy(&window, 1);
}
```

**Source:** [CUDA Programming Guide: L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management)

### 1.4 Stream Priorities

**Why needed:** Prioritize latency-sensitive operations over batch workloads.

| Component | API | Purpose | Why |
|-----------|-----|---------|-----|
| cudaStreamCreateWithPriority() | CUDA 10+ | Create high-priority streams | Preempt lower-priority work |
| cudaStreamGetPriority() | CUDA 10+ | Query stream priority | Monitoring |

**Use case:** Create high-priority streams for time-critical algorithms while batch work uses default/low priority streams.

```cpp
// Example: Priority streams
int priority_low, priority_high;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// priority_high is the highest priority (most negative on some systems)

cudaStream_t urgent_stream, batch_stream;
cudaStreamCreateWithPriority(&urgent_stream, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&batch_stream, cudaStreamNonBlocking, priority_low);
```

**Source:** [CUDA Programming Guide: Stream Priorities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-priorities)

### 1.5 Async Error Propagation

**Why needed:** CUDA errors are asynchronous - they appear on subsequent calls. Production code needs better visibility.

| Component | API | Purpose | Why |
|-----------|-----|---------|-----|
| cudaGetLastError() | Built-in | Check for async errors | Required after kernel launches |
| cudaError_t async notification | CUDA 12+ | Programmatic async error callbacks | Real-time error detection |

**Integration:** Extend the existing error framework (v1.8) with async-aware error checking.

```cpp
// Extend existing error category
class CudaAsyncErrorCategory : public std::error_category {
    // Maps cudaError_t to production-meaningful messages
    // Includes context: which stream, which operation
};
```

**Source:** [CUDA Runtime API: Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)

---

## 2. Observability Integration Points

### 2.1 NVTX Extensions (Already exists - extend)

The v1.7 NVTX integration should be extended with:

| Extension | Purpose |
|-----------|---------|
| `nvtx_range_push/pop` wrappers | Scoped timing regions |
| Custom domain creation | Separate domains for each layer (memory, device, algo, api) |
| Payload annotations | Attach metadata (size, operation type) to ranges |

### 2.2 CUpti Integration

For deeper observability, consider adding CUPTI (CUDA Performance Tools Interface):

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| cupiActivity | Activity tracing (kernel, memcpy) | Production profiling |
| cupiEvent | Hardware counters | Performance tuning |
| cupiSourceLocator | Line-level profiling | Kernel optimization |

**Recommendation:** Start with extending NVTX. Add CUPTI only if users request hardware counter access.

### 2.3 Device Health Monitoring

Extending v1.5 memory error detection:

| Metric | API | Purpose |
|--------|-----|---------|
| Device memory used | cudaMemGetInfo | Memory pressure |
| ECC errors | cuEvent APIs | Hardware health |
| Temperature | NVML API | Thermal throttling |
| Power draw | NVML API | Performance ceiling |

---

## 3. Stress Testing Enhancements

### 3.1 CUDA Error Injection

**Why needed:** Test fault tolerance without real hardware failures.

```cpp
// Production hardening: Test error recovery paths
class ErrorInjector {
    std::random_device rd_;
    std::mt19937 gen_;
    double failure_rate_ = 0.0;

public:
    void setFailureRate(double rate) { failure_rate_ = rate; }

    // Wrap CUDA calls to inject failures in test builds
    template<typename Func>
    cudaError_t wrap(Func&& f) {
        if (std::uniform_real_distribution<>(0.0, 1.0)(gen_) < failure_rate_) {
            return cudaErrorMemoryAllocation;  // Simulate OOM
        }
        return f();
    }
};
```

### 3.2 Chaos Testing with Stream Priorities

Test behavior under resource contention by mixing high and low priority work.

---

## 4. Already Covered - Do NOT Add

| Feature | Exists In | Why Not Duplicate |
|---------|-----------|-------------------|
| Error framework (std::error_code) | v1.8 | Comprehensive |
| CUDA event profiling | v1.6 | Works well |
| NVTX annotations | v1.7 | Header-only, compile-time toggle |
| Google Benchmark | v1.7 | End-to-end benchmarks complete |
| libFuzzer | v2.0 | Fuzzing infrastructure complete |
| Property-based testing | v2.0 | QuickCheck-style tests complete |
| Memory pool statistics | v1.0 | Fragmentation reporting works |
| Stream-based async | v1.0 | Foundation solid |
| Coverage reports (lcov/genhtml) | v2.0 | CI integration complete |
| Performance regression testing | v1.7 | Statistical significance (Welch's t-test) |

---

## 5. Dependencies

### 5.1 New External Dependencies

| Library | Version | Purpose | CMake Integration |
|---------|---------|---------|-------------------|
| NVBench | 1.x | GPU microbenchmarking | FetchContent_Declare |

```cmake
# Add to cmake/fetch_dependencies.cmake
FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG main
)
FetchContent_MakeAvailable(nvbench)
```

### 5.2 CUDA Toolkit Features Required

All features are in CUDA 10+ (compatible with CUDA 20 target):
- CUDA Graphs: CUDA 10+
- Stream Priorities: CUDA 10+
- L2 Cache Persistence: CUDA 11+
- Async Error Notifications: CUDA 12+

---

## 6. Implementation Recommendations

### Phase Structure

1. **Phase 1: CUDA Graphs Foundation**
   - GraphExecutor wrapper class
   - Integration with existing Stream layer
   - Example: Wrap reduce/scan algorithms

2. **Phase 2: Performance Extensions**
   - L2 cache persistence for iterative algorithms
   - Stream priority management
   - NVBench microbenchmarks for kernel tuning

3. **Phase 3: Observability**
   - NVTX domain extensions per layer
   - Async error propagation
   - Device health metrics

4. **Phase 4: Stress Testing**
   - Error injection framework
   - Chaos testing patterns
   - Integration with existing fuzz testing

### Key Design Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GraphExecutor in API layer | Non-invasive, composable with existing algorithms | Wrapper pattern |
| NVBench over custom timing | NVIDIA-maintained, GPU-aware measurement | Better accuracy |
| Extend NVTX vs. new system | Minimal overhead, v1.7 already exists | Additive only |
| Error injection opt-in | Only for test builds, never production | Safety |

---

## 7. Sources

- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Runtime API:** https://docs.nvidia.com/cuda/cuda-runtime-api/
- **NVBench:** https://github.com/NVIDIA/nvbench
- **CCCL (Thrust/CUB/libcudacxx):** https://github.com/NVIDIA/cccl
- **CUDA Graphs Best Practices:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs

---

*Last updated: 2026-04-28*
