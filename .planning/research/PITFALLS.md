# Pitfalls Research

**Domain:** CUDA Library Enhancement Common Mistakes

## Phase 1 Pitfalls (Performance & Quality)

### Device Capability Detection

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Hardcoded block sizes | Code has `256` everywhere | Centralize in device_info.h |
| Ignoring architecture differences | Works on one GPU, fails on another | Test on multiple compute capabilities |
| Assuming latest CUDA | Features don't work on older Toolkit | Check CUDA version at compile time |

**Detection:**
```cpp
// BAD: Hardcoded
kernel<<<grid, 256>>>(args);

// GOOD: Device-aware
int block_size = cuda::device::optimal_block_size<MyKernel>();
kernel<<<grid, block_size>>>(args);
```

### Memory Metrics

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Metrics overhead | Tracking degrades performance | Make metrics optional, off by default |
| Memory leaks in pool | Available memory decreases over time | Track all allocations, warn on imbalance |
| False accuracy | Metrics don't match reality | Verify with cuda-memcheck |

### Benchmark Pitfalls

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Warm-up runs missing | First run always slow | Add warm-up loop |
| Variable input sizes | Performance varies wildly | Test multiple sizes, document results |
| Ignoring variance | Perfect numbers every time | Report stddev, run multiple trials |
| GPU state pollution | Later benchmarks faster | Reset GPU state, use cudaDeviceReset |

**Correct Benchmark Pattern:**
```cpp
void benchmark_kernel() {
    // Warm-up
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block>>>(args);
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    const int iterations = 100;
    float times[iterations];
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<grid, block>>>(args);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<float, std::milli>(end - start).count();
    }
    
    // Report mean and stddev
    float mean = std::accumulate(times, times + iterations, 0.f) / iterations;
    float stddev = /* compute standard deviation */;
}
```

## Phase 2 Pitfalls (Async & Streaming)

### Stream Management

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Stream priority misuse | High priority blocking low priority | Use non-blocking streams for background work |
| Too many streams | `cudaErrorMemoryAllocation` | Limit stream count, reuse streams |
| Implicit synchronization | Code hangs, order wrong | Always specify stream for operations |

### Pinned Memory

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Overusing pinned memory | System becomes unresponsive | Use only for active transfers |
| Not freeing pinned memory | Memory leak visible in htop | RAII wrapper, always free |
| Synchronizing on pinned copy | Async not actually async | Verify with timeline profiler |

### Memory Pool

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Fragmentation | Pool works well, then degrades | Defragmentation strategy |
| Deadlock potential | Program hangs with multiple streams | Single-threaded pool, thread-safe if needed |
| Oversized allocations | Pool consumes too much memory | Set pool limits, fail gracefully |

## Phase 3+ Pitfalls (New Algorithms)

### FFT

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| cuFFT plan reuse | Changes take effect slowly | Rebuild plan when size changes |
| In-place vs out-of-place | Buffer corruption | Document API clearly |
| Power-of-2 assumption | Fails on arbitrary sizes | Support arbitrary sizes or reject gracefully |

### Ray Tracing

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Numerical precision | Rays miss surfaces | Use epsilon tolerance |
| Infinite loops | Hangs on certain inputs | Bound iteration count |
| NaN propagation | Entire image corrupted | Check for NaN, handle gracefully |

### Graph Algorithms

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Dense representation | Out of memory on large graphs | Force sparse format |
| Irregular memory access | Poor GPU utilization | Use CSR, coalesce access |
| Overflow | Distance becomes negative | Use 64-bit counters |

### Neural Net Primitives

| Pitfall | Warning Signs | Prevention |
|---------|---------------|------------|
| Numerical instability | Softmax returns NaN | Use log-sum-exp trick |
| Memory explosion | Out of memory on large models | Support memory-efficient modes |
| Half-precision bugs | Accuracy drops | Test against fp32 reference |

## Cross-Cutting Concerns

| Concern | Pitfall | Prevention |
|---------|---------|------------|
| **Testing** | Tests pass but code broken | Add integration tests |
| **Documentation** | API unclear | Doxygen, examples in header |
| **Backward compatibility** | Existing users broken | Deprecation warnings, version bump |
| **Performance regression** | New code slower than old | Always compare against baseline |
