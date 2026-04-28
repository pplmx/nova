# CUDA Library Production Hardening Pitfalls

**Project:** Nova CUDA Library v2.4 Production Hardening
**Domain:** CUDA Production Readiness
**Researched:** 2026-04-28
**Confidence:** HIGH (based on NVIDIA official documentation, established production patterns, and codebase analysis)

## Executive Summary

This document catalogs common pitfalls when adding production hardening to CUDA libraries, organized into four categories: error handling, performance optimization, stress testing, and reliability. Each pitfall includes root cause analysis, production consequences, and actionable mitigation strategies.

Key cross-cutting themes:
- **Asynchronous operation errors** are frequently mishandled (cudaGetLastError semantics)
- **Profiling on synthetic data** leads to wrong optimization priorities
- **Flaky GPU tests** often indicate timing dependencies, not actual failures
- **False confidence from passing tests** is the most dangerous pitfall

---

## 1. CUDA Error Handling Pitfalls

### 1.1 Missing cudaGetLastError() After Kernel Launch

**What goes wrong:** CUDA API calls return `cudaSuccess` even when the kernel fails, masking errors that appear asynchronously.

**Why it happens:** Kernel launches are asynchronous. The CUDA runtime queues the launch but returns immediately. The actual error (out-of-resources, invalid configuration, illegal address) is recorded separately and must be retrieved via `cudaGetLastError()`.

```cpp
// WRONG: Assumes kernel succeeded if cudaLaunchKernel returns success
cudaLaunchKernel(&kernel, dim3, dim3, args, 0, 0);
// ... other code ...
// cudaSuccess here doesn't mean kernel succeeded!

// WRONG: Only checking API call, not kernel execution
cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
if (err != cudaSuccess) { /* handles memcpy error only */ }
```

**Consequences:**
- Program continues with undefined behavior after kernel failure
- `cudaErrorLaunchFailure` or `cudaErrorIllegalAddress` goes undetected
- Silent data corruption when downstream code assumes kernel produced valid output
- Extremely difficult to debug: error appears unrelated to actual failure

**Prevention:**
```cpp
// CORRECT: Check for kernel errors immediately after launch
cudaLaunchKernel(&kernel, dim3, dim3, args, 0, 0);
cudaError_t err = cudaGetLastError();  // MUST check this
if (err != cudaSuccess) {
    throw cuda_exception(err, "kernel launch", __FILE__, __LINE__);
}

// EVEN BETTER: Use NOVA_CHECK for automatic error checking
NOVA_CHECK(cudaLaunchKernel(&kernel, dim3, dim3, args, 0, 0));
NOVA_CHECK_WITH_STREAM(cudaLaunchKernel(&kernel, dim3, dim3, args, 0, 0), stream);
```

**Detection:** NVIDIA compute-sanitizer (`compute-sanitizer --tool memcheck`) detects illegal memory access in kernels.

**Phase Recommendation:** Phase 33 (Error Framework) — Ensure NOVA_CHECK wraps ALL kernel launches.

---

### 1.2 Ignoring Error State from Previous Operations

**What goes wrong:** `cudaGetLastError()` returns an error from an earlier unrelated operation, not the immediately preceding one.

**Why it happens:** CUDA maintains a single per-thread error state. Any un-checked CUDA call can pollute the error state:

```cpp
// WRONG: cudaGetLastError returns error from cudaFuncGetAttributes,
// not from the kernel launch
cudaFuncGetAttributes(&attrs, kernel);
cudaLaunchKernel(&kernel, dim3, dim3, args, 0, 0);
cudaError_t err = cudaGetLastError();  // Returns cudaSuccess (good!)
// BUT cudaFuncGetAttributes set error state to something else earlier
// and cudaLaunchKernel succeeded, so this is correct
```

**Consequences:**
- Incorrect error attribution when debugging
- Logic errors if code checks `cudaGetLastError()` after multiple operations

**Prevention:**
```cpp
// PATTERN: Clear error state before critical section, then check
cudaPeekAtLastError();  // Clear any pending error
// ... perform operations ...
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Error is from the operations above, not stale state
}
```

**Note:** Nova's `cuda_error_guard` handles this automatically by peeking at existing errors before checking new ones.

---

### 1.3 Error Handling in Async Callbacks

**What goes wrong:** Calling CUDA APIs from stream callbacks causes `cudaErrorNotPermitted` or undefined behavior.

**Why it happens:** CUDA stream callbacks execute on the host but in a restricted context. The CUDA Runtime API is partially blocked:

```cpp
// WRONG: CUDA API calls in callback are not permitted
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    int device;
    cudaGetDevice(&device);  // May fail!
    cudaMalloc(&ptr, size);  // Will likely fail!
    // Many runtime APIs are disallowed in callbacks
}
cudaStreamAddCallback(stream, myCallback, nullptr, 0);
```

**Consequences:**
- Intermittent failures depending on callback timing
- `cudaErrorNotPermitted` returned
- Resource leaks if allocation fails silently

**Prevention:**
```cpp
// CORRECT: Use cudaStreamQuery for error detection instead
void checkStreamErrors(cudaStream_t stream) {
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaErrorNotReady) {
        // Work still in progress, not an error
        return;
    }
    if (err != cudaSuccess) {
        // Actual error occurred
        throw cuda_exception(err, "stream", __FILE__, __LINE__);
    }
}

// CORRECT: For callbacks, only use allowed operations
void CUDART_CB safeCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    // Read-only operations may work, but safest to avoid CUDA APIs entirely
    // Signal another thread via atomics or condition variables instead
    std::atomic_store(&callbackCompleted, true);
}
```

**Phase Recommendation:** Phase 22 (Communication Error Recovery) — Use stream query patterns, not callback CUDA calls.

---

### 1.4 Memory Allocation Without Checking for Null

**What goes wrong:** `cudaMalloc` returns `cudaSuccess` even when allocation fails (pre-Pascal), leading to null pointer dereference.

**Why it happens:** Legacy behavior (compute capability < 6.0): `cudaMalloc` returns `cudaSuccess` but sets ptr to 0. Modern behavior (Pascal+): `cudaMalloc` returns `cudaErrorMemoryAllocation`.

```cpp
// DANGEROUS: Assuming ptr is non-null after cudaSuccess
void* ptr;
cudaError_t err = cudaMalloc(&ptr, size);
if (err == cudaSuccess) {
    // ptr might still be nullptr on older GPUs!
    memset(ptr, 0, size);  // Crash!
}
```

**Consequences:**
- Null pointer dereference crashes on Kepler/Maxwell GPUs
- Silent corruption or security vulnerabilities
- Hard to reproduce on modern hardware

**Prevention:**
```cpp
// ROBUST: Always check both error AND pointer
void* ptr = nullptr;
NOVA_CHECK(cudaMalloc(&ptr, size));
if (ptr == nullptr) {
    throw cuda_exception(cudaErrorMemoryAllocation, "cudaMalloc",
                         __FILE__, __LINE__);
}
// Now safe to use ptr
```

---

### 1.5 Forgetting to Free Resources on Error Paths

**What goes wrong:** Memory leaks and resource exhaustion when errors occur mid-function.

**Why it happens:** Early returns or exceptions bypass cleanup code:

```cpp
// WRONG: Leaks memory if check2 fails
void process(Buffer& buf) {
    float* d_temp;
    cudaMalloc(&d_temp, size1);  // Allocated
    
    check1();  // May throw
    cudaMalloc(&d_temp2, size2);  // Allocated
    
    check2();  // May throw - d_temp leaked!
    
    // cleanup never runs
}

// WRONG: Double-free on exception
void process() {
    float* d_a = nullptr, *d_b = nullptr;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    
    if (condition) throw std::runtime_error("fail");
    
    cudaFree(d_a);  // If exception here, d_b leaked
    cudaFree(d_b);
}
```

**Prevention:**
```cpp
// PATTERN: RAII wrappers (Buffer class already handles this)
Buffer temp1(size1);
Buffer temp2(size2);
// Exceptions automatically free resources

// PATTERN: Scope guards for C-style code
struct CudaFree {
    void* ptr;
    ~CudaFree() { if (ptr) cudaFree(ptr); }
};

void process() {
    CudaFree guard1{nullptr}, guard2{nullptr};
    cudaMalloc(&guard1.ptr, size1);
    cudaMalloc(&guard2.ptr, size2);
    // Exception-safe: both freed on unwind
}
```

**Phase Recommendation:** Phase 10 (Resource Management) — Enforce RAII patterns via code review.

---

### 1.6 Error Propagation Across CUDA Contexts

**What goes wrong:** Errors from one device context leak into another when using multi-GPU code.

**Why it happens:** Each CUDA context maintains its own error state. Explicit context switching (`cudaSetDevice`) doesn't automatically clear error state:

```cpp
// WRONG: Error from device 0 persists when switching to device 1
cudaSetDevice(0);
cudaMalloc(&ptr0, size);  // Fails on device 0
cudaSetDevice(1);
cudaMalloc(&ptr1, size);  // Error state still shows device 0 error!
// cudaGetLastError() returns device 0's error, not cudaSuccess
```

**Consequences:**
- Incorrect error attribution in multi-GPU scenarios
- Operations fail with misleading error messages
- Debugging complexity increases exponentially

**Prevention:**
```cpp
// CORRECT: Clear error after context switch
cudaSetDevice(0);
NOVA_CHECK(cudaMalloc(&ptr0, size));  // Check and clear

cudaSetDevice(1);
cudaPeekAtLastError();  // Clear any stale error from context switch
NOVA_CHECK(cudaMalloc(&ptr1, size));  // Now errors are from device 1
```

---

## 2. Performance Optimization Pitfalls

### 2.1 Premature Optimization Without Profiling

**What goes wrong:** Complex optimizations applied to code that isn't a bottleneck waste development time and reduce maintainability.

**Why it happens:** Intuition about performance is frequently wrong, especially for GPUs where memory access patterns dominate:

```cpp
// PREMATURE: Optimizing a kernel that's called once per frame
// while memory transfers dominate runtime
__global__ void complexKernel(float* data, int n) {
    // Complex optimized computation...
}

// Reality: This kernel takes 0.1ms, but cudaMemcpy takes 10ms
// Optimizing the kernel provides 0.1ms improvement, ignoring 10ms problem
```

**Consequences:**
- 10x development effort for 1% performance gain
- Code complexity increases without proportional benefit
- Maintainability decreases

**Prevention:**
```cpp
// CORRECT: Profile first, then optimize hotspots
void profileApplication() {
    // Use NVIDIA Nsight Compute/Visual Profiler
    // Or NVIDIA Tools Extension (NVTX) for custom markers
    
    // Identify top 3 hotspots:
    // 1. Memory transfer: 10ms (50% of runtime) - OPTIMIZE THIS
    // 2. Kernel A: 5ms (25%) - OPTIMIZE THIS
    // 3. Kernel B: 0.1ms (0.5%) - IGNORE
}
```

**Source:** NVIDIA Best Practices Guide explicitly states: "Before implementing lower priority recommendations, make sure all higher priority recommendations that are relevant have already been applied."

---

### 2.2 Profiling on Unrealistic Workloads

**What goes wrong:** Optimizations based on tiny or synthetic data sizes don't translate to production workloads.

**Why it happens:** GPU performance characteristics change dramatically with data size due to occupancy, cache effects, and memory bandwidth saturation:

```cpp
// WRONG: Testing with data that fits in L2 cache
void benchmark() {
    const int N = 1024;  // Fits in cache - unrepresentative!
    benchmarkKernel(data, N);  // Reports 100 GB/s
    
    // Production workload:
    const int N_PROD = 10 * 1024 * 1024;  // Cache thrashing
    // Same kernel reports 20 GB/s - 5x slower!
}
```

**Consequences:**
- Wrong optimization priorities (tuning for cache-resident data)
- Performance regression in production
- False confidence from benchmark results

**Prevention:**
```cpp
// CORRECT: Profile with production-representative sizes
void comprehensiveBenchmark() {
    std::vector<int> testSizes = {
        1024,           // Small boundary
        65536,          // L2 cache fitting
        1024 * 1024,    // L2 cache thrashing
        16 * 1024 * 1024,      // Working set > L2
        128 * 1024 * 1024,     // Memory pressure
    };
    
    for (int n : testSizes) {
        Buffer d_data(n);
        // Warm up
        kernel(d_data.data(), n);
        cudaDeviceSynchronize();
        
        // Measure steady-state performance
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            kernel(d_data.data(), n);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Report both throughput AND scaling behavior
        reportPerformance(n, duration);
    }
}
```

---

### 2.3 Ignoring Memory Transfer Overhead

**What goes wrong:** Optimizing kernel compute time while ignoring host-device transfer time.

**Why it happens:** Host-device bandwidth (20-50 GB/s) is 5-10x slower than device memory bandwidth (500-1000 GB/s):

```cpp
// WRONG: Focusing on kernel optimization
__global__ void optimizeMe(float* data, int n) {
    // Complex computation...
}

// Reality: This kernel is called with data transfer overhead
void process() {
    float* h_data = loadData();  // CPU memory
    
    float* d_data;
    cudaMalloc(&d_data, size);
    
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // SLOW
    optimizeMe<<<blocks, threads>>>(d_data, n);  // Fast kernel
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);  // SLOW
    
    // Transfer time: 20ms, Kernel time: 1ms
    // Optimizing kernel saves 1ms, optimizing transfer saves 20ms
}
```

**Consequences:**
- 20x opportunity cost from wrong optimization target
- Performance doesn't improve despite complex kernel optimization

**Prevention:**
```cpp
// CORRECT: Minimize transfers first
void optimizedProcess() {
    // Strategy 1: Keep data on device
    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // One-time
    
    for (int iteration = 0; iteration < 1000; iteration++) {
        // Process on device without transfers
        kernel<<<blocks, threads>>>(d_data, n);
    }
    
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);  // One-time
}

// Strategy 2: Overlap transfers with compute
void overlappedProcess() {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    
    cudaMemcpyAsync(d_next, h_next, size, H2D, s1);
    kernel<<<..., s2>>>(d_current, n);  // Compute while transferring
    cudaMemcpyAsync(h_output, d_current, size, D2H, s2);  // Transfer while computing next
}
```

---

### 2.4 Tuning for One Architecture

**What goes wrong:** Block size, shared memory usage, and unroll factors optimized for one GPU don't perform well on others.

**Why it happens:** Different compute capabilities have different resources (registers, shared memory, warp size, SM count):

```cpp
// WRONG: Hardcoded for RTX 3090 (sm_86)
__global__ void kernel(float* data) {
    __shared__ float shared[256];  // Fixed size
    // ...
}
dim3 block(256);  // Good for sm_86, bad for A100 (sm_80)

// WRONG: Assumes 32 threads/warp always
// Different architectures have different warp sizes (future-proofing concern)
```

**Consequences:**
- 2-5x performance regression on different GPUs
- Poor scaling across GPU generations
- Technical debt from architecture-specific code

**Prevention:**
```cpp
// CORRECT: Query device capabilities and adapt
struct KernelConfig {
    int blockSize;
    int sharedMemBytes;
    int unrollFactor;
};

KernelConfig autoTune(const cudaDeviceProp& props) {
    KernelConfig cfg;
    
    // Occupancy-based block size selection
    int maxBlocks;
    cudaOccupancyMaxPotentialBlockSize(&maxBlocks, &cfg.blockSize,
        kernel, cfg.sharedMemBytes, 0);
    
    // Shared memory scaled to device capability
    cfg.sharedMemBytes = std::min(props.sharedMemPerBlock, size_t{65536});
    
    // Unroll factor based on register pressure
    cfg.unrollFactor = (props.regsPerBlock > 65536) ? 8 : 4;
    
    return cfg;
}

// CORRECT: Use warp-native operations
// __shfl_sync works across all architectures
// Bank conflict avoidance via padding works across all shared memory configs
```

---

### 2.5 Microbenchmarking Without Warmup

**What goes wrong:** First-run timing includes JIT compilation, cache warmup, and allocation overhead.

**Why it happens:** CUDA JIT compiles PTX to SASS on first kernel launch. L2 cache is cold. Memory allocations trigger driver overhead:

```cpp
// WRONG: Timing includes JIT compilation
void badBenchmark() {
    auto start = now();
    kernel<<<blocks, threads>>>(d_data, n);  // First launch - JIT
    cudaDeviceSynchronize();
    auto end = now();  // Time includes JIT compilation!
}

// WRONG: Cold cache measurements
void badBenchmark2() {
    cudaFree(nullptr);  // Clear cache
    
    auto start = now();
    kernel<<<...>>>(d_data, n);  // Cold cache
    cudaDeviceSynchronize();
    auto end = now();
    // Not representative of steady-state
}
```

**Consequences:**
- 2-100x variance in first-run timing
- Incorrect optimization decisions
- CI flakiness from JIT timing variance

**Prevention:**
```cpp
// CORRECT: Warmup runs before measurement
void properBenchmark() {
    // Warmup: Force JIT compilation
    for (int i = 0; i < 10; i++) {
        kernel<<<blocks, threads>>>(d_data, n);
    }
    cudaDeviceSynchronize();
    
    // Clear L2 cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    
    // Now measure steady-state
    std::vector<double> timings;
    for (int i = 0; i < 100; i++) {
        auto start = now();
        kernel<<<blocks, threads>>>(d_data, n);
        cudaDeviceSynchronize();
        timings.push_back(elapsed(start, now()));
    }
    
    // Report statistics, not single timing
    reportPercentiles(timings);
}
```

---

## 3. Stress Testing Pitfalls

### 3.1 Flaky Tests from Uninitialized Memory

**What goes wrong:** Tests pass in isolation but fail in suites due to memory state bleeding between tests.

**Why it happens:** GPU memory persists across kernel launches within a process. Tests that don't fully initialize output buffers see stale data:

```cpp
// WRONG: Partially initialized output
__global__ void kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = compute(i);  // But what about out[n-1] on last block?
    }
}

// Test assumes output is initialized, but kernel doesn't write all elements
TEST_F(KernelTest, Basic) {
    float* d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    // d_out contains garbage!
    
    kernel<<<blocks, threads>>>(d_out, n);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(h_out[i], expected[i]);  // Garbage at uninitialized indices!
    }
}
```

**Consequences:**
- Tests pass locally, fail on CI (different memory patterns)
- Tests pass with one CUDA version, fail with another
- Non-deterministic failures that are hard to reproduce

**Prevention:**
```cpp
// CORRECT: Always initialize output buffers
TEST_F(KernelTest, Basic) {
    float* d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    
    // Initialize to safe value
    cudaMemset(d_out, 0xFF, n * sizeof(float));  // NaN for floats
    
    kernel<<<blocks, threads>>>(d_out, n);
    cudaDeviceSynchronize();  // Ensure kernel completes
    
    // Verify all elements were written
    std::vector<float> h_out(n);
    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        ASSERT_FALSE(std::isnan(h_out[i])) << "Index " << i << " not written";
        EXPECT_EQ(h_out[i], expected[i]);
    }
}

// CORRECT: Use initialization patterns in kernels
__global__ void safeKernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = compute(i);
    } else {
        out[i] = 0;  // Initialize unused elements
    }
}
```

---

### 3.2 Race Conditions in Stream-Based Tests

**What goes wrong:** Tests fail intermittently due to stream synchronization missing or incorrect.

**Why it happens:** CUDA streams execute asynchronously. Tests that don't wait for completion read stale data:

```cpp
// WRONG: No synchronization
TEST_F(StreamTest, AsyncCopy) {
    float *d_data, *h_data = new float[n];
    cudaMalloc(&d_data, n * sizeof(float));
    
    cudaMemcpyAsync(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // WRONG: Immediately reading from device after async call
    // Stream hasn't executed yet!
    float* h_result = new float[n];
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    // h_result contains garbage
}
```

**Consequences:**
- Intermittent test failures (depends on timing)
- Flaky CI jobs
- Difficult to debug timing-dependent issues

**Prevention:**
```cpp
// CORRECT: Always synchronize before reading results
TEST_F(StreamTest, AsyncCopy) {
    float *d_data, *h_data = new float[n];
    cudaMalloc(&d_data, n * sizeof(float));
    
    cudaMemcpyAsync(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Synchronize stream
    NOVA_CHECK_WITH_STREAM(cudaStreamSynchronize(stream), stream);
    
    // Now safe to read
    float* h_result = new float[n];
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(h_result[i], h_data[i]);
    }
}

// CORRECT: Use events for fine-grained synchronization
TEST_F(StreamTest, EventSync) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    kernel<<<..., stream>>>(d_data, n);
    cudaEventRecord(stop, stream);
    
    // Wait for specific event
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
}
```

---

### 3.3 Coverage Gaps for Edge Cases

**What goes wrong:** Tests cover happy paths but miss boundary conditions, leading to production failures.

**Why it happens:** Common edge cases are systematically omitted:

| Edge Case | Why Missed |
|-----------|------------|
| Empty input (n=0) | "Nobody uses this" |
| Single element | "Not worth testing" |
| Power-of-2 boundaries | "Shouldn't matter" |
| Maximum allocation | "Too expensive to test" |
| Alignment boundaries | "Unlikely to matter" |

**Consequences:**
- Production crashes on edge cases
- Security vulnerabilities from buffer overflows
- Numerical instability at boundaries

**Prevention:**
```cpp
// CORRECT: Explicit edge case tests
class EdgeCaseTest : public ::testing::Test {
protected:
    void TestEmptyInput() {
        float* d_out;
        NOVA_CHECK(cudaMalloc(&d_out, 0));
        kernel<<<1, 1>>>(d_out, 0);  // Should not crash
        NOVA_CHECK(cudaGetLastError());
        cudaFree(d_out);
    }
    
    void TestSingleElement() {
        float data = 42.0f;
        float* d_data;
        NOVA_CHECK(cudaMalloc(&d_data, sizeof(float)));
        NOVA_CHECK(cudaMemcpy(d_data, &data, sizeof(float), cudaMemcpyHostToDevice));
        
        kernel<<<1, 1>>>(d_data, 1);
        NOVA_CHECK(cudaGetLastError());
        
        float result;
        NOVA_CHECK(cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost));
        EXPECT_EQ(result, expected);
        
        cudaFree(d_data);
    }
    
    void TestPowerOfTwoBoundary(int size) {
        std::vector<float> h_data(size);
        std::iota(h_data.begin(), h_data.end(), 0);
        
        Buffer d_data(size);
        d_data.copyFrom(h_data.data(), size);
        
        kernel<<<(size + 255) / 256, 256>>>(d_data.data(), size);
        NOVA_CHECK(cudaGetLastError());
        
        // Verify all elements processed
    }
};

TEST_F(EdgeCaseTest, EmptyInput) { TestEmptyInput(); }
TEST_F(EdgeCaseTest, SingleElement) { TestSingleElement(); }
TEST_F(EdgeCaseTest, Size256) { TestPowerOfTwoBoundary(256); }
TEST_F(EdgeCaseTest, Size1024) { TestPowerOfTwoBoundary(1024); }
TEST_F(EdgeCaseTest, Size65536) { TestPowerOfTwoBoundary(65536); }
```

---

### 3.4 Resource Leak Tests Missing

**What goes wrong:** Memory leak tests don't exist, allowing gradual resource exhaustion in long-running applications.

**Why it happens:** GPU memory leaks are silent and accumulate over time. Single-run tests don't detect them:

```cpp
// WRONG: Test allocates but doesn't check cleanup
TEST_F(MemoryTest, Allocation) {
    float* ptr;
    cudaMalloc(&ptr, size);
    
    // Do work...
    
    // WRONG: Not verifying cleanup
    // cudaFree(ptr);  // Forgotten!
}
```

**Consequences:**
- Out-of-memory errors after hours of operation
- Gradual performance degradation
- Application crashes in production

**Prevention:**
```cpp
// CORRECT: Memory leak detection test
TEST_F(MemoryLeakTest, NoLeakInOperations) {
    size_t memBefore, memAfter;
    cudaMemGetInfo(&memBefore, nullptr);
    
    // Perform many allocations/deallocations
    for (int i = 0; i < 1000; i++) {
        Buffer temp(1024 * 1024);
        temp.copyFrom(data.data(), data.size());
        // temp automatically freed at end of iteration
    }
    
    cudaMemGetInfo(&memAfter, nullptr);
    
    // Allow small variance for fragmentation
    EXPECT_LE(memBefore - memAfter, 1024 * 1024)  // < 1MB leak
        << "Memory leak detected: " << (memBefore - memAfter) << " bytes";
}

// CORRECT: Test with leak sanitizer
// Run with: compute-sanitizer --tool leakcheck ./test
TEST_F(MemoryLeakTest, NoCudaLeaks) {
    float* ptr1, *ptr2;
    NOVA_CHECK(cudaMalloc(&ptr1, 1024));
    NOVA_CHECK(cudaMalloc(&ptr2, 1024));
    
    // ... use pointers ...
    
    NOVA_CHECK(cudaFree(ptr1));
    NOVA_CHECK(cudaFree(ptr2));
    // compute-sanitizer will report any leaks
}
```

---

### 3.5 Determinism Tests Missing

**What goes wrong:** Non-deterministic algorithms produce varying results, but tests don't verify determinism.

**Why it happens:** Floating-point non-associativity, parallel reduction order, and race conditions cause result variation:

```cpp
// WRONG: Test expects exact match but GPU reduction order varies
TEST_F(ReduceTest, Sum) {
    std::vector<float> data(1000);
    std::iota(data.begin(), data.end(), 1.0f);
    
    float result = reduce(data);
    
    EXPECT_EQ(result, 500500.0f);  // May fail on different GPUs!
}
```

**Consequences:**
- Tests fail on different GPU architectures
- Performance optimizations change reduction order and break tests
- Debug/release build differences

**Prevention:**
```cpp
// CORRECT: Test with tolerance for floating-point
TEST_F(ReduceTest, Sum) {
    std::vector<float> data(1000);
    std::iota(data.begin(), data.end(), 1.0f);
    
    float result = reduce(data);
    
    EXPECT_NEAR(result, 500500.0f, 0.001f)  // Allow small error
        << "Result: " << result << " expected: " << 500500.0f;
}

// CORRECT: Test determinism by running multiple times
TEST_F(ReduceTest, Determinism) {
    std::vector<float> data(1000);
    std::iota(data.begin(), data.end(), 1.0f);
    
    float result1 = reduce(data);
    float result2 = reduce(data);
    float result3 = reduce(data);
    
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(result2, result3);
}

// CORRECT: Integer comparisons for stable results
TEST_F(SortTest, StableSort) {
    // Use integer keys for deterministic sorting
    std::vector<uint32_t> keys(1000);
    std::iota(keys.begin(), keys.end(), 0);
    std::shuffle(keys.begin(), keys.end(), std::mt19937{42});  // Fixed seed
    
    sort(keys);
    
    for (int i = 1; i < keys.size(); i++) {
        EXPECT_LT(keys[i-1], keys[i]);
    }
}
```

---

## 4. Reliability Pitfalls

### 4.1 False Confidence from Passing Unit Tests

**What goes wrong:** Unit tests cover individual components but miss integration failures.

**Why it happens:** Unit tests run components in isolation with controlled inputs:

```cpp
// Unit test passes: component works in isolation
TEST_F(MatrixMultiplyTest, SinglePrecision) {
    Matrix a = generateMatrix(256, 256);
    Matrix b = generateMatrix(256, 256);
    Matrix c = matmul(a, b);
    EXPECT_TRUE(verify(c, expected));
    // PASSES
}

// Integration failure: memory alignment issue only occurs with large matrices
TEST_F(IntegrationTest, LargeMatrixMultiply) {
    Matrix a = generateMatrix(8192, 8192);  // Different code path
    Matrix b = generateMatrix(8192, 8192);
    Matrix c = matmul(a, b);  // CRASHES - alignment issue
}
```

**Consequences:**
- High unit test coverage, low integration reliability
- Production failures on edge cases
- Refactoring breaks production code without breaking tests

**Prevention:**
```cpp
// CORRECT: Layered testing pyramid
class TestPyramid {
    // Level 1: Unit tests (many, fast)
    void testComponentIsolation() { /* ... */ }
    
    // Level 2: Integration tests (fewer, slower)
    void testComponentInteraction() { /* ... */ }
    
    // Level 3: System tests (few, slowest)
    void testFullPipeline() { /* ... */ }
    
    // Level 4: Property-based tests (medium, catches edge cases)
    void testProperties() { /* random inputs */ }
};

// CORRECT: Property-based testing for edge cases
TEST_F(PropertyTest, MatrixMultiplyCorrectness) {
    // Generate random valid matrices
    for (int trial = 0; trial < 1000; trial++) {
        int m = randomInt(1, 16384);
        int n = randomInt(1, 16384);
        int k = randomInt(1, 16384);
        
        Matrix a = generateRandomMatrix(m, k);
        Matrix b = generateRandomMatrix(k, n);
        
        Matrix c = matmul(a, b);
        Matrix expected = cpuMatmul(a, b);  // Reference
        
        EXPECT_MATRIX_NEAR(c, expected, 1e-5f);
    }
}
```

---

### 4.2 Missing Device Capability Checks

**What goes wrong:** Code assumes capabilities that aren't available on all target GPUs.

**Why it happens:** Features have minimum compute capability requirements:

| Feature | Minimum CC | GPUs Affected |
|---------|-----------|---------------|
| Unified memory | 3.0 | Kepler (3.0-3.7) |
| Dynamic parallelism | 3.5 | Kepler (3.0-3.7) |
| FP16 tensor cores | 7.0 | Pascal (6.0-6.2) |
| BF16 | 8.0 | Ampere (8.0-8.9) |
| DP4a | 6.1 | Pascal+ |

**Consequences:**
- `cudaErrorNoKernelImageForDevice` at runtime
- Silent fallback to slow code paths
- Different results on different GPUs

**Prevention:**
```cpp
// CORRECT: Check capability before using features
struct DeviceCapabilities {
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int maxThreadsPerBlock;
    size_t sharedMemPerBlock;
    bool supportsFP16;
    bool supportsTensorCore;
    bool supportsUnifiedAddressing;
};

DeviceCapabilities queryDevice(int deviceId) {
    cudaDeviceProp prop;
    NOVA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    return {
        prop.major,
        prop.minor,
        prop.maxThreadsPerBlock,
        prop.sharedMemPerBlock,
        prop.major >= 5.3,  // Half-precision
        prop.major >= 7.0,  // Tensor cores
        prop.unifiedAddressing
    };
}

// CORRECT: Conditional feature usage
void process(Buffer& data) {
    auto caps = queryDevice(currentDevice());
    
    if (caps.supportsFP16) {
        // Use fast FP16 path
        kernel_fp16<<<blocks, threads, 0, stream>>>(data_fp16);
    } else {
        // Use compatible FP32 path
        kernel_fp32<<<blocks, threads, 0, stream>>>(data_fp32);
    }
}
```

---

### 4.3 Ignoring ECC Error Handling

**What goes wrong:** Production systems with ECC memory don't handle corrected errors gracefully.

**Why it happens:** ECC GPUs report corrected single-bit errors but programs ignore them:

```cpp
// WRONG: Ignoring ECC corrected errors
void process(Buffer& data) {
    // GPU memory has experienced corrected bit flip
    // cudaErrorCorrected ECC error not checked
    
    kernel<<<blocks, threads>>>(data.data(), data.size());
    // Results may be unreliable but error not detected
}
```

**Consequences:**
- Silent data corruption in scientific computing
- Reliability failures in long-running applications
- Difficult to debug intermittent failures

**Prevention:**
```cpp
// CORRECT: Monitor ECC status via NVML
#include <nvml.h>

nvmlReturn_t checkECCStatus(int device) {
    nvmlEccCounterType_t types[] = {NVML_VOLATILE_ECC, NVML_AGGREGATE_ECC};
    
    for (auto type : types) {
        nvmlEccCounter64_t singleBit, doubleBit;
        nvmlDeviceGetMemoryErrorCounter(device, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                        type, NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                                        &singleBit);
        nvmlDeviceGetMemoryErrorCounter(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                        type, NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                                        &doubleBit);
        
        if (singleBit > warningThreshold) {
            // Log warning, consider checkpointing
            logECCWarning(device, type, singleBit);
        }
        
        if (doubleBit > 0) {
            // Uncorrectable error - fail safely
            throw EccUncorrectableError(device, doubleBit);
        }
    }
    return NVML_SUCCESS;
}

// CORRECT: Periodic health monitoring
class GPUHealthMonitor {
    size_t lastSingleBitErrors_{0};
    
    void checkAndAlert() {
        size_t currentErrors = querySingleBitErrors();
        if (currentErrors > lastSingleBitErrors_) {
            size_t newErrors = currentErrors - lastSingleBitErrors_;
            logWarning("ECC corrected errors: ", newErrors);
            
            if (newErrors > threshold_) {
                // Consider checkpoint and restart
                requestCheckpoint("ECC error threshold exceeded");
            }
        }
        lastSingleBitErrors_ = currentErrors;
    }
};
```

---

### 4.4 Missing Timeout Handling

**What goes wrong:** GPU kernels run forever due to infinite loops or deadlocks, causing watchdog timer resets.

**Why it happens:** Infinite loops or improper synchronization cause GPU to hang:

```cpp
// DANGEROUS: Infinite loop in kernel
__global__ void badKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (true) {  // Infinite loop!
        data[i] = compute(data[i]);
    }
}

// DANGEROUS: Deadlock via improper barrier usage
__global__ void deadlockKernel(float* data) {
    __shared__ float shared[256];
    
    if (threadIdx.x < 128) {
        shared[threadIdx.x] = data[threadIdx.x];
    }
    __syncthreads();  // First barrier
    
    if (threadIdx.x >= 128) {
        // Only half threads reach here
        shared[threadIdx.x] = data[threadIdx.x];
    }
    __syncthreads();  // DEADLOCK - 128 threads waiting forever
}
```

**Consequences:**
- Watchdog timer reset (GPU reset by driver)
- Application crash
- Other GPUs in system affected
- Difficult recovery

**Prevention:**
```cpp
// CORRECT: Use cudaOccupancyMaxActiveBlocksPerMin to estimate kernel duration
void validateKernelLaunch(const void* kernel) {
    int blocks;
    int gridSize;
    NOVA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, 256, 0));
    
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    gridSize = blocks * prop.multiProcessorCount;
    
    // Sanity check: kernel should complete in reasonable time
    // Rough estimate: if gridSize * iterations > threshold, warn
    if (gridSize > 100000) {
        logWarning("Large kernel launch: ", gridSize, " blocks");
    }
}

// CORRECT: Watchdog timeout configuration
// Set CUDA_TIMEOUT to reasonable value
// cudaDeviceSetLimit(cudaLimitTimeout, timeout_ms);

// CORRECT: Stream timeout for testing
TEST_F(StreamTest, KernelTimeout) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    kernel<<<1000, 256, 0, stream>>>(data, n);
    
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaErrorNotReady) {
        // Wait with timeout
        for (int i = 0; i < 100; i++) {
            std::this_thread::sleep_for(100ms);
            err = cudaStreamQuery(stream);
            if (err == cudaSuccess) break;
            if (err != cudaErrorNotReady) {
                ADD_FAILURE() << "Kernel error: " << cudaGetErrorString(err);
                break;
            }
        }
        if (err == cudaErrorNotReady) {
            ADD_FAILURE() << "Kernel timeout exceeded";
        }
    }
}
```

---

### 4.5 Floating-Point Comparison Stability

**What goes wrong:** Floating-point comparisons use exact equality, failing on different GPUs or optimization levels.

**Why it happens:** IEEE 754 compliance varies slightly, and floating-point operations are not associative:

```cpp
// WRONG: Exact equality comparison
TEST_F(MathTest, Sum) {
    float result = compute();
    EXPECT_EQ(result, 1.0f);  // May be 0.9999999 or 1.0000001
}
```

**Consequences:**
- Tests fail on different GPU generations
- Tests fail between debug/release builds
- Tests fail with different CUDA versions

**Prevention:**
```cpp
// CORRECT: Tolerance-based comparison
TEST_F(MathTest, Sum) {
    float result = compute();
    EXPECT_NEAR(result, 1.0f, 1e-6f);  // Absolute tolerance
}

// CORRECT: Relative tolerance for large values
TEST_F(MathTest, LargeSum) {
    float result = compute();  // Result ~1e10
    EXPECT_FLOAT_EQ(result, expected);  // Uses relative tolerance
}

// CORRECT: Custom comparison for GPU-specific needs
MATCHER_P(FloatNear, expected, abs_error) {
    return std::abs(arg - expected) <= abs_error;
}

TEST_F(MathTest, CustomComparison) {
    float result = compute();
    EXPECT_THAT(result, FloatNear(1.0f, 1e-6f));
}

// CORRECT: For stable comparisons, use integer bit patterns
TEST_F(SortTest, StableSort) {
    // Sort by integer representation for stability
    std::vector<uint32_t> keys = floatAsUint(original);
    sort(keys);
    
    for (int i = 1; i < keys.size(); i++) {
        EXPECT_LE(keys[i-1], keys[i]);
    }
}
```

---

## Summary: Phase Recommendations

| Phase Topic | Likely Pitfall | Mitigation Strategy |
|-------------|----------------|---------------------|
| 33. Error Framework | Missing cudaGetLastError | Enforce NOVA_CHECK on all kernel launches |
| 22. Error Recovery | Ignoring async errors | Stream query patterns, not callback CUDA calls |
| Memory Pool | Resource leaks | RAII wrappers, leak sanitizer tests |
| Multi-GPU | Error state across contexts | Clear error after context switch |
| 27. Kernel Fusion | Profiling on tiny data | Production-representative sizes |
| Performance | Memory transfer overhead | Minimize transfers before kernel optimization |
| 26. Profiling | Microbenchmark warmup | JIT warmup runs, cache warmup |
| 29. Benchmarks | Tuning for one GPU | Query device capabilities, auto-tune |
| Test Infrastructure | Flaky async tests | Stream synchronization before reads |
| Fuzz Testing | Edge case gaps | Edge case parameterized tests |
| Memory Tests | No leak detection | cudaMemGetInfo before/after, compute-sanitizer |
| 31. Regression Tests | False confidence | Integration tests, property-based tests |
| 24. Signal Handling | Watchdog timeouts | Occupancy validation, timeout tests |
| 23. ECC Handling | Ignoring corrected errors | NVML health monitoring |
| Math Operations | FP comparison instability | Tolerance-based comparisons |

---

## Sources

- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - **HIGH confidence** (official NVIDIA documentation, v13.2)
- [NVIDIA CUDA Runtime API Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html) - **HIGH confidence** (official API documentation)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - **HIGH confidence** (official NVIDIA documentation)
- [Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/) - **HIGH confidence** (NVIDIA tools)
- [NVML Documentation](https://docs.nvidia.com/deploy/nvml-api/) - **HIGH confidence** (NVIDIA management library)
- [Google Test Best Practices](https://google.github.io/googletest/) - **MEDIUM confidence** (established testing framework)
- [CUDA Pro Tip: Unified Memory Performance](https://developer.nvidia.com/blog/unified-memory-cuda-pro-tip-memory-management/) - **HIGH confidence** (NVIDIA developer blog)

---

*Last updated: 2026-04-28 for Nova v2.4 Production Hardening milestone*
