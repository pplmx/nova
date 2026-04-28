# Feature Landscape: Production Hardening

**Domain:** CUDA Library Production Hardening (Error Handling, Performance Optimization, Testing, Reliability)
**Researched:** 2026-04-28
**Confidence:** HIGH (based on NVIDIA official documentation and existing codebase analysis)

## Executive Summary

Nova v2.4 Production Hardening targets four critical areas for production readiness: error handling, performance optimization, stress testing, and reliability features. The project already has substantial infrastructure in place that should be extended rather than replaced.

Key findings:
- **Error handling:** Existing `cuda/device/error.h` with `CudaException` and context macros. Needs enhancement for CUDA Graphs, cuSOLVER, and async error propagation.
- **Performance optimization:** Stream infrastructure exists; CUDA Graphs and stream priorities are missing.
- **Stress testing:** Fuzzing (`tests/fuzz/`) and property-based tests (`tests/property/`) exist. Needs CUDA-specific fuzzing and benchmark integration.
- **Reliability:** `comm_error_recovery.h`, `preemption_handler.h`, `checkpoint_manager.h` exist. Health monitoring and diagnostics need enhancement.

---

## 1. Error Handling & Recovery

### 1.1 Existing Infrastructure

| Component | Location | Coverage |
|-----------|----------|----------|
| `CudaException` | `include/cuda/device/error.h` | Runtime errors |
| `CudaExceptionWithContext` | `include/cuda/device/error.h` | Contextual errors |
| `CUDA_CHECK` macro | `include/cuda/device/error.h` | Sync error checking |
| `OperationContext` | `include/cuda/device/error.h` | Error context tracking |
| `CommErrorRecovery` | `include/cuda/comm/comm_error_recovery.h` | NCCL/TCP recovery |
| `HealthMonitor` | `include/cuda/comm/comm_error_recovery.h` | Health monitoring |

### 1.2 Table Stakes (Required for Production)

| Feature | Status | Implementation | Complexity | Notes |
|---------|--------|----------------|------------|-------|
| **Sync error propagation** | Existing | `CUDA_CHECK` macro | LOW | Already implemented |
| **Async error capture** | Needs work | Event-based error collection | MEDIUM | Errors in async operations |
| **cuSOLVER error handling** | Needs work | Custom `CUSOLVER_CHECK` | LOW | Similar to `CUBLAS_CHECK` |
| **cuFFT error handling** | Needs work | Custom `CUFFT_CHECK` | LOW | cufftResult_t checking |
| **Error context enrichment** | Existing | `CudaExceptionWithContext` | LOW | Extend for graphs |

### 1.3 Differentiators (Advanced Error Handling)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **CUDA Graph error recovery** | Graceful graph node failure | MEDIUM | Partial graph execution recovery |
| **Multi-GPU error isolation** | Isolate failing GPU in mesh | MEDIUM | Already has peer access detection |
| **ECC error detection** | Memory error handling | MEDIUM | `cudaDeviceGetAttribute` for ECC |
| **Timeout detection** | Stall detection | MEDIUM | Uses `cudaStreamQuery` |
| **Error aggregation** | Collect multi-operation errors | MEDIUM | Batch error reporting |

### 1.4 Error Categories & Recovery Patterns

```cpp
// Error Category Hierarchy
enum class NovaErrorCategory {
    // Runtime errors
    MemoryAllocation,      // cudaErrorMemoryAllocation
    InvalidConfiguration,  // cudaErrorInvalidConfiguration
    LaunchFailure,         // cudaErrorLaunchFailure
    LaunchTimeout,         // cudaErrorLaunchTimeout
    
    // Library errors
    CublasError,           // cublasStatus_t
    CufftError,            // cufftResult
    CusolverError,         // cusolverStatus_t
    NcclError,             // ncclResult
    
    // Multi-GPU errors
    PeerAccessError,       // P2P access failures
    CollectiveTimeout,     // NCCL collective stalls
    TopologyError,         // Device mesh errors
    
    // Application errors
    CheckpointError,       // Serialization failures
    PreemptionError,       // Signal handling failures
};

// Recovery Strategy by Category
struct RecoveryStrategy {
    NovaErrorCategory category;
    enum class Action { Retry, Fallback, Abort, Isolate } action;
    int max_retries;
    std::chrono::milliseconds retry_delay;
};
```

### 1.5 Async Error Handling Patterns

```cpp
// Pattern: Async error collection via events
class AsyncErrorCollector {
public:
    void record_error_point(cudaStream_t stream, cudaEvent_t* event_out) {
        // Create event that captures async errors
        CUDA_CHECK(cudaEventCreateWithFlags(event_out, 
            cudaEventDisableTiming | cudaEventBlockingSync));
    }
    
    cudaError_t check_async_error(cudaEvent_t event) {
        // Synchronize and capture error
        CUDA_CHECK(cudaEventSynchronize(event));
        return cudaPeekAtLastError();  // Captures kernel launch errors
    }
    
    // For streams: poll periodically
    bool is_stream_stalled(cudaStream_t stream, 
                          std::chrono::milliseconds timeout) {
        cudaError_t err = cudaStreamQuery(stream);
        if (err == cudaSuccess) return false;
        if (err == cudaErrorNotReady) {
            // Check if exceeded timeout
            auto elapsed = std::chrono::steady_clock::now() - start_time_;
            return elapsed > timeout;
        }
        throw CudaException(err, __FILE__, __LINE__);
    }
};

// Pattern: Error propagation wrapper
template<typename Func>
auto wrap_with_error_tracking(Func&& func, 
                              const OperationContext& ctx) 
    -> std::invoke_result_t<Func> 
{
    try {
        auto result = func();
        // Check for async errors
        cudaError_t async_err = cudaPeekAtLastError();
        if (async_err != cudaSuccess) {
            throw CudaExceptionWithContext(async_err, __FILE__, 
                                           __LINE__, ctx);
        }
        return result;
    } catch (const CudaException& e) {
        // Enrich with context
        throw CudaExceptionWithContext(e.error(), __FILE__, 
                                       __LINE__, ctx);
    }
}
```

### 1.6 CUDA Graph Error Handling

```cpp
// Pattern: CUDA Graph execution with partial failure handling
class GraphExecutionMonitor {
public:
    struct NodeStatus {
        int node_id;
        bool executed;
        cudaError_t error;
    };
    
    std::optional<NodeStatus> execute_with_tracking(
        cudaGraphExec_t graph_exec,
        cudaStream_t stream) 
    {
        cudaEvent_t completion_event;
        CUDA_CHECK(cudaEventCreate(&completion_event));
        
        // Record state before
        cudaError_t pre_err = cudaPeekAtLastError();
        
        // Execute graph
        cudaError_t result = cudaGraphLaunch(graph_exec, stream);
        
        // Record completion
        CUDA_CHECK(cudaEventRecord(completion_event, stream));
        CUDA_CHECK(cudaEventSynchronize(completion_event));
        
        // Check errors
        cudaError_t post_err = cudaPeekAtLastError();
        
        if (result != cudaSuccess || post_err != cudaSuccess) {
            // Query individual node status if available
            return NodeStatus{
                .node_id = -1,
                .executed = false,
                .error = result != cudaSuccess ? result : post_err
            };
        }
        
        return std::nullopt;
    }
    
    // For conditional graphs: handle branch failures
    void execute_conditional_with_fallback(
        cudaGraphExec_t if_graph,
        cudaGraphExec_t else_graph,
        bool condition,
        cudaStream_t stream) 
    {
        cudaGraphExec_t chosen = condition ? if_graph : else_graph;
        cudaError_t result = cudaGraphLaunch(chosen, stream);
        
        if (result != cudaSuccess) {
            // Fallback to safe default path
            launch_safe_fallback(stream);
        }
    }
};
```

### 1.7 Anti-Patterns to Avoid

| Anti-Pattern | Why Avoid | Instead |
|--------------|-----------|---------|
| **Ignoring async errors** | Silent corruption | Check via events after synchronization |
| **Catching all exceptions** | Masks specific failures | Catch specific exception types |
| **Blocking on every launch** | Kills parallelism | Use event-based polling |
| **No timeout on collectives** | Indefinite hangs | Implement timeout with abort |
| **Retrying permanent errors** | Wasteful, delays failure | Classify errors first |

### Sources

- [NVIDIA CUDA Runtime API - Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html) — **HIGH confidence**
- [NVIDIA CUDA Programming Guide - Error Checking](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking) — **HIGH confidence**
- Existing `include/cuda/device/error.h` implementation — **HIGH confidence**

---

## 2. Performance Optimization

### 2.1 Existing Infrastructure

| Component | Location | Coverage |
|-----------|----------|----------|
| `Stream` class | `include/cuda/stream/stream.h` | Basic streams |
| `Event` class | `include/cuda/stream/event.h` | Basic events |
| `Profiler` | `include/cuda/performance/profiler.h` | Kernel timing |
| `MemoryMetrics` | `include/cuda/performance/memory_metrics.h` | Memory tracking |
| `Autotuner` | `include/cuda/performance/autotuner.h` | Config tuning |
| `NVTX` | `include/cuda/benchmark/nvtx.h` | GPU tracing |

### 2.2 Table Stakes (Required for Production)

| Feature | Status | Implementation | Complexity | Notes |
|---------|--------|----------------|------------|-------|
| **Stream priorities** | Missing | Priority streams API | LOW | Low-priority compute, high-priority latency-critical |
| **L2 cache persistence** | Missing | `cudaMemAccess` API | MEDIUM | For iterative algorithms |
| **Memory prefetching** | Missing | Stream-ordered allocation | MEDIUM | `cudaMallocAsync` |
| **Kernel fusion** | Existing | `fused_matmul_bias_act.h` | MEDIUM | Extend to more patterns |

### 2.3 Differentiators (Advanced Optimization)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **CUDA Graphs** | 10-100x launch overhead reduction | MEDIUM | Graph capture and instantiation |
| **Graph updates** | Efficient parameter updates | MEDIUM | Partial graph updates |
| **Tensor memory access** | Large transfers via TMA | HIGH | Hopper+ only |
| **Async memory operations** | Overlap copies with compute | MEDIUM | `cudaMemcpyAsync` |
| **Occupancy calculator** | Auto-tune launch config | MEDIUM | Use `cudaOccupancyMaxPotentialBlockSize` |

### 2.4 CUDA Graphs Integration

```cpp
// Pattern: Stream capture for repeated workloads
class GraphBuilder {
public:
    // Capture pattern: wrap algorithm in capture stream
    void capture_sort_pipeline(cudaStream_t capture_stream) {
        CUDA_CHECK(cudaStreamBeginCapture(capture_stream, 
            cudaStreamCaptureModeGlobal));
        
        // All operations on capture_stream become graph nodes
        pre_process_kernel_<<<..., capture_stream>>>(...);
        sort_kernel_<<<..., capture_stream>>>(...);
        post_process_kernel_<<<..., capture_stream>>>(...);
        
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
        
        // Instantiate once, execute many times
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, 
            nullptr, nullptr, 0));
        
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    
    void execute(size_t iteration) {
        // Each launch has minimal overhead
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
    }
    
    // Update parameters without full rebuild
    void update_sort_thresholds(float* new_thresholds) {
        cudaGraphNode_t* nodes;
        size_t num_nodes;
        CUDA_CHECK(cudaGraphGetNodes(graph_exec_, &nodes, &num_nodes));
        
        // Find parameter node and update
        for (size_t i = 0; i < num_nodes; ++i) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
            if (type == cudaGraphNodeTypeKernel) {
                // Update kernel parameters
                void* params[] = { /* updated params */ };
                CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
                    graph_exec_, nodes[i], &kernel_params_));
            }
        }
    }

private:
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaStream_t stream_;
};

// Pattern: Conditional graph nodes for adaptive algorithms
class AdaptiveGraphExecutor {
public:
    void build_conditional_sort(cudaGraph_t graph) {
        // Create conditional handle
        cudaGraphConditionalHandle handle;
        CUDA_CHECK(cudaGraphConditionalHandleCreate(
            &handle, graph, 0, cudaStreamTailGraphLaunch));
        
        // Small array: use simple sort
        cudaGraph_t small_sort_graph = build_simple_sort_graph();
        cudaGraphAddConditionalNode(graph, &small_node_, handle,
            cudaGraphConditionalNodeTypeIF, &small_sort_graph);
        
        // Large array: use radix sort
        cudaGraph_t large_sort_graph = build_radix_sort_graph();
        cudaGraphAddConditionalNode(graph, &large_node_, handle,
            cudaGraphConditionalNodeTypeIF, &large_sort_graph);
        
        // Set condition based on array size
        CUDA_CHECK(cudaGraphConditionalNodeSetParameters(
            large_node_, &param_));  // param controls branching
    }
};
```

### 2.5 Stream Priority Implementation

```cpp
// Pattern: Priority-based stream management
class PriorityStreamPool {
public:
    struct StreamConfig {
        int priority;  // Lower = higher priority
        bool coalescable;  // Enable CUDA Graph capture
    };
    
    void initialize(int device_id, int num_high, int num_normal, 
                    int num_low) {
        // Query priority range for device
        int least_priority, greatest_priority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
            &least_priority, &greatest_priority));
        
        // Create priority pools
        for (int i = 0; i < num_high; ++i) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, greatest_priority));
            high_priority_streams_.push_back(stream);
        }
        
        for (int i = 0; i < num_normal; ++i) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, 
                (least_priority + greatest_priority) / 2));
            normal_priority_streams_.push_back(stream);
        }
        
        for (int i = 0; i < num_low; ++i) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, least_priority));
            low_priority_streams_.push_back(stream);
        }
    }
    
    cudaStream_t get_stream(StreamPriority priority) {
        switch (priority) {
            case StreamPriority::High:
                return get_from_pool(high_priority_streams_);
            case StreamPriority::Normal:
                return get_from_pool(normal_priority_streams_);
            case StreamPriority::Low:
                return get_from_pool(low_priority_streams_);
        }
    }
    
    void return_stream(StreamPriority priority, cudaStream_t stream) {
        // Reset stream state before returning to pool
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return_to_pool(priority, stream);
    }

private:
    std::vector<cudaStream_t> high_priority_streams_;
    std::vector<cudaStream_t> normal_priority_streams_;
    std::vector<cudaStream_t> low_priority_streams_;
};

enum class StreamPriority { High, Normal, Low };
```

### 2.6 L2 Cache Persistence

```cpp
// Pattern: L2 cache persistence for iterative algorithms
class L2CacheOptimizer {
public:
    void configure_persistent_access(
        void* d_data, 
        size_t size,
        cudaMemoryAdvise advice) 
    {
        // Hint: data will be accessed repeatedly
        CUDA_CHECK(cudaMemAdvise(d_data, size, 
            cudaMemAdviseSetPreferredLocation, 0));
        
        // Set persisting access category
        CUDA_CHECK(cudaMemAdvise(d_data, size,
            cudaMemAdviseSetAccessedBy, 0));
        
        // For iterative algorithms: set persisting
        if (advice == cudaMemAdviseSetPersistingAccess) {
            CUDA_CHECK(cudaMemAdvise(d_data, size,
                cudaMemAdviseSetPersistingAccess, 0));
        }
    }
    
    void reset_to_normal() {
        // Reset L2 cache to normal after persistent operations
        CUDA_CHECK(cudaCtxResetPersistingL2Cache());
    }
    
    // Query L2 properties
    struct L2Properties {
        size_t total_size;
        size_t set_aside_size;
        size_t granulariy;
    };
    
    L2Properties get_properties(int device_id) {
        L2Properties props;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &props.set_aside_size,
            cudaDevAttrL2CacheSize, device_id));
        // Query other properties...
        return props;
    }
};
```

### 2.7 Anti-Patterns to Avoid

| Anti-Pattern | Why Avoid | Instead |
|--------------|-----------|---------|
| **Capturing every frame** | Capture overhead | Pre-capture once, update params |
| **Large graphs** | Update cost high | Split into subgraphs |
| **Blocking on priority streams** | Defeats priority | Use events |
| **Ignoring L2 cache** | Bandwidth waste | Use persistence for iterative |

### Sources

- [NVIDIA CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) — **HIGH confidence**
- [NVIDIA CUDA Programming Guide - Stream Priorities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-priorities) — **HIGH confidence**
- [NVIDIA CUDA Programming Guide - L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management) — **HIGH confidence**

---

## 3. Stress Testing Patterns

### 3.1 Existing Infrastructure

| Component | Location | Coverage |
|-----------|----------|----------|
| Fuzz tests | `tests/fuzz/*.cpp` | Memory pool, algorithms, matmul |
| Property tests | `tests/property/*.cpp` | Math, numerical, algorithmic |
| Regression tests | `tests/benchmark/regression_test.cpp` | Performance regression |
| Benchmark suite | `tests/benchmark/` | Throughput testing |

### 3.2 Table Stakes (Required for Production)

| Feature | Status | Implementation | Complexity | Notes |
|---------|--------|----------------|------------|-------|
| **Edge case coverage** | Partial | Existing fuzz tests | LOW | Extend to new algorithms |
| **Memory pressure tests** | Missing | OOM simulation | MEDIUM | Allocate near limits |
| **Concurrent stress** | Missing | Multi-stream contention | MEDIUM | Race condition detection |
| **CUDA graph testing** | Missing | Graph error injection | MEDIUM | Validate error recovery |

### 3.3 Differentiators (Advanced Testing)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Deterministic fuzzing** | Reproducible failures | LOW | Seed-based fuzzing |
| **Coverage-guided fuzzing** | Target uncovered paths | MEDIUM | libFuzzer integration |
| **Property-based GPU tests** | Algorithm correctness | MEDIUM | Arbitrary input generation |
| **Fault injection** | Hardware error simulation | HIGH | ECC, timeout simulation |
| **Chaos engineering** | Production failure testing | HIGH | Random error injection |

### 3.4 Fuzz Testing Patterns

```cpp
// Pattern: CUDA-aware libFuzzer integration
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Parse fuzzer input
    if (size < sizeof(FuzzConfig)) return 0;
    
    auto config = *reinterpret_cast<const FuzzConfig*>(data);
    data += sizeof(FuzzConfig);
    size -= sizeof(FuzzConfig);
    
    // Initialize CUDA
    cuda::device::Context ctx(config.device_id);
    
    try {
        // Test memory allocation edge cases
        if (config.test_memory) {
            test_memory_allocation(data, size, config);
        }
        
        // Test algorithm with various sizes
        if (config.test_algorithm) {
            test_algorithm_correctness(data, size, config);
        }
        
        // Test stream operations
        if (config.test_streams) {
            test_stream_operations(data, size, config);
        }
        
        // Check for async errors
        cudaError_t async_err = cudaPeekAtLastError();
        if (async_err != cudaSuccess) {
            // Log but don't fail - async errors may be recoverable
            log_async_error(async_err);
        }
        
    } catch (const CudaException& e) {
        // Expected for some edge cases
        return 0;
    }
    
    return 0;
}

// Pattern: Edge case generation
struct FuzzConfig {
    uint32_t device_id : 4;
    uint32_t test_memory : 1;
    uint32_t test_algorithm : 1;
    uint32_t test_streams : 1;
    uint32_t size_hint : 26;  // Size hint for allocations
    uint32_t seed;  // For deterministic reproduction
};

// Pattern: Memory pressure testing
class MemoryPressureTest {
public:
    void stress_allocate_and_free(size_t total_size, 
                                  size_t chunk_size,
                                  int iterations) {
        std::vector<void*> allocations;
        allocations.reserve(total_size / chunk_size);
        
        for (int i = 0; i < iterations; ++i) {
            for (size_t allocated = 0; allocated < total_size; 
                 allocated += chunk_size) {
                void* ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, chunk_size);
                
                if (err == cudaSuccess) {
                    allocations.push_back(ptr);
                } else {
                    // Near OOM - test recovery
                    test_recovery_on_oom(allocations);
                    break;
                }
            }
            
            // Free all
            for (void* ptr : allocations) {
                cudaFree(ptr);
            }
            allocations.clear();
        }
    }
};
```

### 3.5 Property-Based Testing Patterns

```cpp
// Pattern: GPU algorithm property testing
template<typename Algorithm>
class GpuPropertyTest {
public:
    struct Property {
        std::string name;
        std::function<bool(const std::vector<T>&)> property_fn;
    };
    
    void register_property(std::string name, 
                          std::function<bool(const std::vector<T>&)> fn) {
        properties_.push_back({name, fn});
    }
    
    void test_algorithm(Algorithm& algo, size_t iterations) {
        RandomGenerator gen(seed_);
        
        for (size_t i = 0; i < iterations; ++i) {
            // Generate random input
            auto size = gen.Uniform<size_t>(1, 100000);
            auto input = gen.generate_vector<T>(size);
            
            // Run algorithm
            auto output = algo.execute(input);
            
            // Check all properties
            for (const auto& prop : properties_) {
                if (!prop.property_fn(output)) {
                    report_failure(prop.name, input, output, seed_ + i);
                }
            }
        }
    }
};

// Example properties for sorting
GpuPropertyTest<SortAlgorithm> test;
test.register_property("is_sorted", 
    [](const std::vector<int>& result) {
        return std::is_sorted(result.begin(), result.end());
    });

test.register_property("same_elements", 
    [](const std::vector<int>& result, 
       const std::vector<int>& original) {
        auto a = result; auto b = original;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        return a == b;
    });

test.register_property("stable_for_equal", 
    [](const std::vector<Record>& result) {
        // Check stability: equal keys maintain original order
        for (size_t i = 1; i < result.size(); ++i) {
            if (result[i].key == result[i-1].key &&
                result[i].index < result[i-1].index) {
                return false;
            }
        }
        return true;
    });
```

### 3.6 Stress Testing Patterns

```cpp
// Pattern: Concurrent multi-stream stress test
class ConcurrentStreamStressTest {
public:
    void stress_test(int num_streams, int iterations) {
        std::vector<cuda::stream::Stream> streams(num_streams);
        std::vector<std::vector<int>> results(num_streams);
        
        for (int iter = 0; iter < iterations; ++iter) {
            // Launch concurrent work
            std::vector<cudaEvent_t> events(num_streams);
            
            for (int s = 0; s < num_streams; ++s) {
                launch_kernel_on_stream(streams[s], iter);
                cudaEventRecord(&events[s], streams[s]);
            }
            
            // Wait for all
            for (int s = 0; s < num_streams; ++s) {
                cudaEventSynchronize(events[s]);
            }
            
            // Verify no data corruption
            verify_no_corruption(results);
        }
    }
    
    // Test for race conditions
    void test_race_conditions() {
        constexpr int THREADS = 4;
        std::atomic<int> conflict_count{0};
        
        // Multiple threads accessing same GPU resources
        std::vector<std::thread> threads;
        for (int t = 0; t < THREADS; ++t) {
            threads.emplace_back([&, t]() {
                cuda::stream::Stream stream;
                for (int i = 0; i < 1000; ++i) {
                    // Potential race: concurrent allocation
                    void* ptr = allocate_or_null();
                    if (!ptr) conflict_count++;
                    if (ptr) cudaFree(ptr);
                }
            });
        }
        
        for (auto& t : threads) t.join();
        
        // Report race frequency
        if (conflict_count > 0) {
            log_warning("Allocation conflicts: {}", conflict_count.load());
        }
    }
};

// Pattern: CUDA Graph stress testing
class GraphStressTest {
public:
    void test_graph_execution_stress(cudaGraphExec_t exec,
                                     int iterations) {
        for (int i = 0; i < iterations; ++i) {
            // Vary execution conditions
            set_random_stream_priority();
            inject_memory_pressure(i % 10 == 0);
            
            // Execute
            cudaError_t result = cudaGraphLaunch(exec, stream_);
            
            // Check result
            if (result != cudaSuccess) {
                test_partial_execution(exec);
            }
            
            // Verify output integrity
            verify_output_correctness();
        }
    }
    
    void test_graph_update_stress() {
        for (int i = 0; i < 1000; ++i) {
            // Update parameters repeatedly
            update_kernel_params();
            
            // Execute updated graph
            cudaGraphLaunch(exec_, stream_);
            
            // Verify no corruption
            verify_graph_integrity();
        }
    }
};
```

### 3.7 Anti-Patterns to Avoid

| Anti-Pattern | Why Avoid | Instead |
|--------------|-----------|---------|
| **Non-deterministic tests** | Flaky tests | Fix seeds, log state |
| **Ignoring async errors** | Silent corruption | Check after every sync |
| **No OOM testing** | Production crashes | Test near memory limits |
| **Single-device only** | Multi-GPU bugs | Test device mesh |

### Sources

- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html) — **HIGH confidence**
- [RapidCheck property-based testing](https://github.com/rapidcheck/rapidcheck) — **MEDIUM confidence**
- Existing `tests/fuzz/` and `tests/property/` implementations — **HIGH confidence**

---

## 4. Reliability Features

### 4.1 Existing Infrastructure

| Component | Location | Coverage |
|-----------|----------|----------|
| `CommErrorRecovery` | `include/cuda/comm/comm_error_recovery.h` | NCCL error handling |
| `HealthMonitor` | `include/cuda/comm/comm_error_recovery.h` | Health checks |
| `PreemptionHandler` | `include/cuda/preemption/preemption_handler.h` | Signal handling |
| `CheckpointManager` | `include/cuda/checkpoint/checkpoint_manager.h` | State save/restore |
| `MemoryErrorHandler` | `include/cuda/memory_error/memory_error_handler.h` | Memory errors |

### 4.2 Table Stakes (Required for Production)

| Feature | Status | Implementation | Complexity | Notes |
|---------|--------|----------------|------------|-------|
| **Device health checks** | Partial | Existing HealthMonitor | LOW | Extend for CUDA Graphs |
| **Memory diagnostics** | Partial | `memory_metrics.h` | LOW | Extend fragmentation analysis |
| **Timeout handling** | Partial | CommErrorRecovery | MEDIUM | Add stream-level timeouts |
| **Graceful degradation** | Missing | Fallback strategies | MEDIUM | Single-GPU fallback |

### 4.3 Differentiators (Advanced Reliability)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Runtime diagnostics** | Production issue detection | MEDIUM | Memory leaks, resource exhaustion |
| **Error telemetry** | Production monitoring | MEDIUM | Structured error logging |
| **Self-healing** | Automatic recovery | HIGH | Restart failed components |
| **Health dashboards** | Visual monitoring | MEDIUM | Web-based or CLI |

### 4.4 Health Monitoring Patterns

```cpp
// Pattern: Comprehensive health monitoring
class DeviceHealthMonitor {
public:
    struct HealthReport {
        int device_id;
        std::chrono::steady_clock::time_point timestamp;
        
        // Memory health
        size_t memory_used;
        size_t memory_total;
        float memory_utilization;
        
        // Compute health
        float gpu_utilization;
        float memory_bandwidth;
        float l2_hit_rate;
        
        // Thermal health
        int temperature_celsius;
        bool thermal_throttling;
        
        // Error health
        int uncorrectable_errors;
        int corrected_errors;
        bool ECC_enabled;
        
        // Recommendations
        std::vector<std::string> warnings;
        std::vector<std::string> recommendations;
    };
    
    HealthReport collect_report(int device_id) {
        HealthReport report;
        report.device_id = device_id;
        report.timestamp = std::chrono::steady_clock::now();
        
        // Memory metrics
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        report.memory_used = total - free;
        report.memory_total = total;
        report.memory_utilization = 
            static_cast<float>(report.memory_used) / total;
        
        // Compute utilization (requires nvml or profiler)
        report.gpu_utilization = query_gpu_utilization(device_id);
        
        // Thermal health
        report.temperature_celsius = query_temperature(device_id);
        report.thermal_throttling = 
            report.temperature_celsius > thermal_limit_;
        
        // ECC errors
        query_ecc_errors(device_id, 
            &report.uncorrectable_errors,
            &report.corrected_errors);
        report.ECC_enabled = query_ecc_mode(device_id);
        
        // Generate recommendations
        generate_recommendations(report);
        
        return report;
    }
    
    void generate_recommendations(HealthReport& report) {
        if (report.memory_utilization > 0.9f) {
            report.warnings.push_back(
                "High memory utilization - consider memory optimization");
            report.recommendations.push_back(
                "Enable memory pool reuse");
        }
        
        if (report.corrected_errors > 100) {
            report.warnings.push_back(
                "High rate of corrected ECC errors");
            report.recommendations.push_back(
                "Run memory test to identify failing hardware");
        }
        
        if (report.thermal_throttling) {
            report.warnings.push_back(
                "Thermal throttling detected");
            report.recommendations.push_back(
                "Reduce compute intensity or improve cooling");
        }
    }
    
    void export_telemetry(const HealthReport& report, 
                         const std::string& endpoint) {
        // JSON export for monitoring systems
        json j;
        j["device_id"] = report.device_id;
        j["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
            report.timestamp.time_since_epoch()).count();
        j["memory_utilization"] = report.memory_utilization;
        j["temperature"] = report.temperature_celsius;
        j["ecc_errors"] = report.uncorrectable_errors;
        
        // POST to telemetry endpoint
        http_post(endpoint, j.dump());
    }

private:
    int thermal_limit_ = 85;  // Celsius
};

// Pattern: Stream-level timeout monitoring
class StreamTimeoutMonitor {
public:
    struct WatchHandle {
        cudaStream_t stream;
        std::chrono::steady_clock::time_point start;
        std::chrono::milliseconds timeout;
        std::function<void()> on_timeout;
    };
    
    WatchHandle watch_stream(cudaStream_t stream,
                             std::chrono::milliseconds timeout,
                             std::function<void()> callback) {
        WatchHandle handle;
        handle.stream = stream;
        handle.start = std::chrono::steady_clock::now();
        handle.timeout = timeout;
        handle.on_timeout = callback;
        
        std::lock_guard<std::mutex> lock(mutex_);
        watches_[stream] = handle;
        
        return handle;
    }
    
    void check_timeouts() {
        auto now = std::chrono::steady_clock::now();
        
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [stream, watch] : watches_) {
            cudaError_t err = cudaStreamQuery(stream);
            
            if (err == cudaErrorNotReady) {
                auto elapsed = now - watch.start;
                if (elapsed > watch.timeout) {
                    // Timeout! Execute callback
                    watch.on_timeout();
                    
                    // Cancel watch
                    watches_.erase(stream);
                }
            } else if (err == cudaSuccess) {
                // Stream completed normally
                watches_.erase(stream);
            }
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<cudaStream_t, WatchHandle> watches_;
};
```

### 4.5 Diagnostics Patterns

```cpp
// Pattern: Runtime diagnostics for issue detection
class RuntimeDiagnostics {
public:
    enum class Issue {
        MemoryFragmentation,
        ResourceLeak,
        PerformanceRegression,
        DeviceError
    };
    
    struct Diagnostic {
        Issue issue;
        std::string description;
        std::string suggested_action;
        float severity;  // 0-1
    };
    
    std::vector<Diagnostic> run_diagnostics() {
        std::vector<Diagnostic> results;
        
        // Check memory fragmentation
        if (auto frag = check_memory_fragmentation()) {
            results.push_back(*frag);
        }
        
        // Check for resource leaks
        if (auto leak = check_resource_leaks()) {
            results.push_back(*leak);
        }
        
        // Check performance
        if (auto perf = check_performance()) {
            results.push_back(*perf);
        }
        
        return results;
    }
    
    std::optional<Diagnostic> check_memory_fragmentation() {
        auto stats = memory_pool_.get_fragmentation_stats();
        
        // Calculate fragmentation score
        float score = stats.largest_free_block / stats.total_free;
        
        if (score < 0.1f) {
            return Diagnostic{
                .issue = Issue::MemoryFragmentation,
                .description = fmt::format(
                    "Severe fragmentation: largest block {:.1f}% of total",
                    score * 100),
                .suggested_action = 
                    "Consider memory compaction or pool reset",
                .severity = 0.8f
            };
        }
        
        return std::nullopt;
    }
    
    std::optional<Diagnostic> check_resource_leaks() {
        // Track allocation/deallocation counts
        int alloc_count = allocations_.load();
        int free_count = frees_.load();
        
        if (alloc_count - free_count > 1000) {
            return Diagnostic{
                .issue = Issue::ResourceLeak,
                .description = fmt::format(
                    "{} potential leaks detected", 
                    alloc_count - free_count),
                .suggested_action = 
                    "Review allocation patterns for missing frees",
                .severity = 0.9f
            };
        }
        
        return std::nullopt;
    }
};
```

### 4.6 Graceful Degradation Patterns

```cpp
// Pattern: Graceful degradation for multi-GPU failures
class GracefulDegradationManager {
public:
    enum class FallbackStrategy {
        SingleDevice,      // Fall back to single GPU
        ReducedPrecision,  // Use lower precision
        CPUOffload,        // Fall back to CPU
        ReducedBatchSize,  // Process smaller batches
        Abort              // No recovery possible
    };
    
    struct DegradationResult {
        bool degraded;
        FallbackStrategy strategy;
        std::string reason;
    };
    
    DegradationResult handle_device_failure(
        int failed_device,
        const std::vector<int>& available_devices) 
    {
        // Strategy 1: Try other GPUs
        if (!available_devices.empty()) {
            // Redistribute work to surviving devices
            redistribute_work(failed_device, available_devices);
            return {
                .degraded = true,
                .strategy = FallbackStrategy::SingleDevice,
                .reason = fmt::format(
                    "Device {} failed, redistributed to {}",
                    failed_device, available_devices[0])
            };
        }
        
        // Strategy 2: Reduced precision
        if (precision_ != Precision::FP16) {
            precision_ = Precision::FP16;
            return {
                .degraded = true,
                .strategy = FallbackStrategy::ReducedPrecision,
                .reason = "Switched to FP16 to reduce memory pressure"
            };
        }
        
        // Strategy 3: Reduced batch size
        if (batch_size_ > 1) {
            batch_size_ /= 2;
            return {
                .degraded = true,
                .strategy = FallbackStrategy::ReducedBatchSize,
                .reason = fmt::format(
                    "Reduced batch size to {}", batch_size_)
            };
        }
        
        return {
            .degraded = false,
            .strategy = FallbackStrategy::Abort,
            .reason = "All recovery strategies exhausted"
        };
    }

private:
    Precision precision_ = Precision::FP32;
    int batch_size_ = 32;
};
```

### 4.7 Anti-Patterns to Avoid

| Anti-Pattern | Why Avoid | Instead |
|--------------|-----------|---------|
| **Silent failures** | Undetected corruption | Log and alert on failures |
| **No timeouts** | Indefinite hangs | Implement strict timeouts |
| **Ignoring ECC errors** | Hardware failure ignored | Alert and log patterns |
| **No fallback** | Complete service failure | Implement graceful degradation |

### Sources

- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — **HIGH confidence**
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/) — **MEDIUM confidence**
- Existing `include/cuda/comm/comm_error_recovery.h` — **HIGH confidence**

---

## Feature Dependencies

### Error Handling
```
CudaException
    └──requires──> cuda/device/error.h (existing)
    └──enhances──> CUSOLVER_CHECK (new)
    └──enhances──> CUFFT_CHECK (new)
    └──extends───> AsyncErrorCollector (new)
    
AsyncErrorCollector
    └──uses──────> cudaStreamQuery
    └──uses──────> cudaEventCreateWithFlags
    └──enables───> GraphExecutionMonitor (new)
```

### Performance Optimization
```
Stream (existing)
    └──extends───> PriorityStreamPool (new)
    
Profiler (existing)
    └──uses──────> CUDA Graphs
    └──enhances──> L2CacheOptimizer (new)
    
GraphBuilder (new)
    └──uses──────> cudaStreamBeginCapture
    └──uses──────> cudaGraphInstantiate
    └──enhances──> AdaptiveGraphExecutor (new)
```

### Stress Testing
```
Fuzz tests (existing in tests/fuzz/)
    └──extends───> CUDA-aware fuzzing (new)
    └──extends───> MemoryPressureTest (new)
    
Property tests (existing in tests/property/)
    └──extends───> GpuPropertyTest template (new)
    └──extends───> ConcurrentStreamStressTest (new)
```

### Reliability
```
HealthMonitor (existing)
    └──extends───> DeviceHealthMonitor (new)
    └──uses──────> ECC error queries
    └──uses──────> Thermal monitoring

CommErrorRecovery (existing)
    └──extends───> StreamTimeoutMonitor (new)
    └──extends───> GracefulDegradationManager (new)
```

---

## MVP Definition

### Phase 1: Error Handling Enhancement (Priority: HIGH)

Minimum viable error handling improvements:
- [ ] `CUSOLVER_CHECK` and `CUFFT_CHECK` macros
- [ ] Async error collection via events
- [ ] CUDA Graph error monitoring

### Phase 2: Performance Optimization (Priority: HIGH)

Minimum viable performance optimizations:
- [ ] Priority stream pool
- [ ] CUDA Graph capture for common pipelines
- [ ] L2 cache persistence hints

### Phase 3: Testing Enhancement (Priority: MEDIUM)

Core testing improvements:
- [ ] Memory pressure fuzz tests
- [ ] Concurrent stream stress tests
- [ ] GPU property-based test framework

### Phase 4: Reliability Enhancement (Priority: MEDIUM)

Core reliability features:
- [ ] Device health monitoring
- [ ] Stream timeout monitoring
- [ ] Graceful degradation strategies

---

## Complexity Assessment by Feature

| Domain | Table Stakes | Differentiators | Complexity Notes |
|--------|--------------|-----------------|------------------|
| **Error Handling** | CUDA_CHECK, CUBLAS_CHECK (existing) | Async errors, Graph errors | Low for macros; medium for async |
| **Performance** | Streams, Profiler (existing) | CUDA Graphs, priorities, L2 | Medium for graphs; low for priorities |
| **Testing** | Fuzz, Property tests (existing) | Memory pressure, Chaos | Medium for advanced patterns |
| **Reliability** | ErrorRecovery, HealthMonitor (existing) | Diagnostics, Degradation | Medium for diagnostics; high for self-healing |

---

## Sources

### Primary Sources (HIGH Confidence)

1. **NVIDIA CUDA Runtime API v13.2** - Error handling, streams, graphs
2. **NVIDIA CUDA C++ Programming Guide** - Best practices for error checking
3. **NVIDIA CUDA Best Practices Guide** - Production deployment patterns
4. Existing Nova infrastructure in `include/cuda/` and `tests/`

### Secondary Sources (MEDIUM Confidence)

5. **libFuzzer Documentation** - Fuzz testing patterns
6. **RapidCheck** - Property-based testing for C++
7. **NVIDIA DCGM** - Data center GPU management patterns

### Tertiary Sources (LOW Confidence, Verify)

8. **Production CUDA patterns** - Blog posts on CUDA reliability
9. **Fault injection literature** - GPU error simulation techniques

---

## Research Gaps & Future Investigation

| Gap | Why Needed | When to Investigate |
|-----|------------|---------------------|
| **Fault injection framework** | Test ECC error handling | Phase 3+ |
| **Self-healing algorithms** | Automatic component restart | Phase 4+ |
| **CUDA Graph debugging** | Production graph issues | Phase 3+ |
| **Memory pool compaction** | Fragmentation mitigation | Phase 4+ |

---

*Last updated: 2026-04-28 for Nova v2.4 Production Hardening*
