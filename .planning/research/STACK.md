# Stack Research

**Domain:** CUDA Performance Optimization & New Algorithm Implementation

## Standard Stack (2025)

### Build & Development

| Component | Recommended | Notes |
|-----------|-------------|-------|
| **CMake** | 3.25+ | CUDA language support built-in |
| **CUDA Toolkit** | 12.x | Latest features, better diagnostics |
| **Compiler** | NVCC + clang | Use NVCC for device code, clang for host |
| **Formatter** | clang-format | Google style baseline |
| **Linter** | clang-tidy | CUDA-specific checks available |

### Performance Profiling

| Tool | Purpose |
|------|---------|
| **Nsight Systems** | System-wide timeline, identify bottlenecks |
| **Nsight Compute** | Kernel-level profiling, occupancy analysis |
| **nvprof** | Legacy profiler, still useful for quick checks |
| **cuda-memcheck** | Memory error detection |

## Device Capability-Aware Launch Configuration

### Key Formulas

```cpp
// Occupancy-optimal block size (must test empirically)
int block_size = 256; // Starting point
while (block_size > 32) {
    int max_blocks = max_active_blocks(compute_occupancy, block_size);
    if (max_blocks > 0) break;
    block_size /= 2;
}

// Grid size from data size
int grid_size = (N + block_size - 1) / block_size;

// For large datasets, use grid-stride loops
dim3 grid((N + block_size - 1) / block_size);
```

### Compute Capability Targets

| Capability | Architecture | Global Memory Bandwidth | Shared Memory |
|------------|--------------|------------------------|---------------|
| 6.0 | Pascal | 160 GB/s | 64 KB |
| 7.0 | Volta | 900 GB/s | 96 KB |
| 8.0 | Turing | 500 GB/s | 64 KB |
| 9.0 | Ampere | 900+ GB/s | 164 KB |

## Memory Optimization

### Pinned Memory Usage

```cpp
// Allocate pinned memory for async transfers
void* host_ptr;
cudaMallocHost(&host_ptr, size);  // Pinned, page-locked

// Async copy with streams
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

// Free pinned memory
cudaFreeHost(host_ptr);
```

### Memory Pool Best Practices

- Pre-allocate large blocks, sub-allocate
- Use memory arenas to reduce fragmentation
- Track allocation statistics (hits, misses, fragmentation %)
- Support device-local pools for multi-stream scenarios

## Async & Streams

### Stream Priorities

```cpp
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
```

| Priority | Use Case |
|----------|----------|
| High (lower number) | Interactive, low-latency |
| Low (higher number) | Background computation |

### Event-Based Synchronization

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## Version Recommendations

| Component | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| CMake | 3.25+ | 3.28+ | Better CUDA 12.x support |
| CUDA Toolkit | 17 | 12.6+ | Stable, good toolchain |
| C++ Standard | 20 | 20 | Stable, no regression needed |

## What NOT to Use

- **Thrust** — too high-level for our layer architecture
- **CUDA cooperative groups** — complex, premature abstraction
- **CUDA graph API** — overkill unless batch-processing identical graphs
- **OpenMP offload** — different model, less control
