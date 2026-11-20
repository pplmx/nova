# Performance Profiling Guide

How to measure and optimize Nova CUDA performance using benchmarks.

## Overview

Nova includes comprehensive benchmarks for:

- Memory operations (H2D, D2H, D2D)
- Algorithms (reduce, scan, sort, FFT)
- Neural net primitives (matmul, softmax)
- Multi-GPU collectives

## Running Benchmarks

```bash
# Build benchmarks. There is no separate NOVA_BUILD_BENCHMARKS option: benchmarks
# are part of NOVA_BUILD_TESTS_AND_BENCHMARKS, which is ON by default.
cmake -G Ninja -B build
cmake --build build

# Run all benchmarks
./build/bin/nova_benchmarks

# Run specific benchmark
./build/bin/nova_benchmarks --benchmark_filter=Memory

# Run with timing output
./build/bin/nova_benchmarks --benchmark_format=csv > results.csv
```

## Python Benchmark Harness

```bash
# Run with Python harness. It writes a timestamped JSON file into --output-dir
# (default results/ at the repository root), not to a single file named by the flag.
python scripts/benchmark/run_benchmarks.py --build-dir build --output-dir results

# Compare a run against a recorded baseline version, with a percentage tolerance
python scripts/benchmark/run_benchmarks.py --build-dir build --baseline v2.14 --tolerance 5

# Render a dashboard from a results directory
python scripts/benchmark/generate_dashboard.py --results results --output reports
```

`--gpu N` selects the device by setting `CUDA_VISIBLE_DEVICES` for the child
process; Google Benchmark itself has no device flag.

## Key Metrics

### Throughput

```cpp
// Measure throughput (GB/s)
double throughput_gbps = bytes / (elapsed_seconds * 1e9);
std::cout << "Throughput: " << throughput_gbps << " GB/s\n";
```

### Latency

```cpp
// Measure latency (microseconds)
auto start = std::chrono::high_resolution_clock::now();
kernel<<<...>>>(...);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Latency: " << us.count() << " us\n";
```

### Memory Bandwidth

```cpp
// Effective bandwidth
double bandwidth = (bytes_read + bytes_written) / (elapsed_seconds * 1e9);
std::cout << "Memory bandwidth: " << bandwidth << " GB/s\n";
```

## NVTX Profiling

Enable NVTX annotations for detailed timeline analysis:

```bash
cmake -B build -DNOVA_ENABLE_NVTX=ON
cmake --build build

# Profile with ncu (NVIDIA Nsight Compute)
ncu --set full ./build/bin/nova_benchmarks --benchmark_filter=Reduce
```

## Comparing Against Baseline

```python
import json

# Load baseline
with open('baseline.json') as f:
    baseline = json.load(f)

# Load current
with open('results.json') as f:
    current = json.load(f)

# Compare
for name, value in current.items():
    if name in baseline:
        delta = (value - baseline[name]) / baseline[name]
        if abs(delta) > 0.05:  # 5% threshold
            print(f"REGRESSION: {name} changed by {delta*100:.1f}%")
        else:
            print(f"OK: {name}")
```

## Optimization Tips

### 1. Memory Access Patterns

```cpp
// Good: Coalesced access
__global__ void good_kernel(const float* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        process(data[i]);  // Sequential
    }
}

// Bad: Strided access (except when needed)
__global__ void bad_kernel(const float* __restrict__ data, int n) {
    int i = threadIdx.x;  // Warp divergence
    while (i < n) {
        process(data[i]);
        i += blockDim.x;  // Strided
    }
}
```

### 2. Memory Coalescing

```cpp
// Ensure memory access is coalesced
// Threads should access consecutive memory locations
int tid = blockIdx.x * blockDim.x + threadIdx.x;
data[tid] = value;  // Good: consecutive
```

### 3. Shared Memory Usage

```cpp
// Use shared memory for frequently accessed data
__shared__ float shared_data[256];

// Load into shared memory
shared_data[threadIdx.x] = global_data[threadIdx.x];
__syncthreads();

// Use shared memory
float value = shared_data[threadIdx.x];
```

### 4. Device Selection

```cpp
// Select fastest device
int device = 0;
int max_throughput = 0;
for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (prop.memoryClockRate > max_throughput) {
        max_throughput = prop.memoryClockRate;
        device = i;
    }
}
cudaSetDevice(device);
```

## Performance Checklist

- [ ] Enable NVTX for timeline analysis
- [ ] Run with `--benchmark_format=csv` for analysis
- [ ] Compare against baseline for regression detection
- [ ] Profile with `ncu` for kernel-level analysis
- [ ] Check memory bandwidth utilization
- [ ] Verify coalesced memory access
- [ ] Enable GPU clocks for maximum performance: `nvidia-smi -pm 1`

## Common Issues

### Low memory bandwidth

- Check if data is in GPU memory (not CPU)
- Verify coalesced access patterns
- Consider pinned memory for H2D/D2H

### Kernel not scaling

- Check block size (typically 256-512 threads)
- Verify grid size covers problem size
- Avoid branch divergence

## Next Steps

- [Architecture](../ARCHITECTURE.md) - Layer and module reference
- [Examples](../../examples/) - More code examples
