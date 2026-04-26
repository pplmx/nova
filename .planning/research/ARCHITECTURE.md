# Architecture Research

**Domain:** CUDA/C++ Benchmarking and Performance Testing Infrastructure
**Researched:** 2026-04-26
**Confidence:** HIGH

## Executive Summary

This document defines the architecture for adding benchmarking infrastructure to an existing CUDA library with a five-layer architecture (device → memory → algo → api → distributed). The infrastructure will integrate Google Benchmark for C++ kernels, provide a Python harness for orchestration and regression detection, add NVTX profiling annotations to existing code, and generate HTML dashboards for trend visualization.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BENCHMARKING INFRASTRUCTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Python Harness Layer                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   Runner    │  │   Reporter  │  │   Baselines │  │  Dashboard  │ │   │
│  │  │  (invoke)   │  │  (JSON)     │  │  (storage)  │  │  (HTML)     │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │   │
│  └─────────┼────────────────┼────────────────┼────────────────┼────────┘   │
│            │                │                │                │             │
│  ┌─────────┴────────────────┴────────────────┴────────────────┴────────┐   │
│  │                    C++ Google Benchmark Layer                        │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │   │
│  │  │  benchmark/                                                     │ │   │
│  │  │  ├── benchmark_reduce.cpp       # algo layer kernels            │ │   │
│  │  │  ├── benchmark_scan.cpp         # algo layer kernels            │ │   │
│  │  │  ├── benchmark_matmul.cpp       # neural layer kernels          │ │   │
│  │  │  ├── benchmark_distributed.cpp  # distributed layer kernels     │ │   │
│  │  │  └── benchmark_memory.cpp       # memory layer kernels          │ │   │
│  │  └──────────────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│            │                                                                 │
│  ┌─────────┴─────────────────────────────────────────────────────────────┐   │
│  │                    Source Code with NVTX Annotations                   │   │
│  ├────────────────────────────────────────────────────────────────────────┤
│  │  nova/algo/      → NVTX ranges for reduce, scan, sort, FFT, matmul     │   │
│  │  nova/distributed/ → NVTX ranges for NCCL/MPI collectives              │   │
│  │  nova/memory/    → NVTX ranges for pool alloc, buffer ops              │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Recommended Project Structure

```
nova/
├── benchmark/                          # C++ Google Benchmark kernels (NEW)
│   ├── CMakeLists.txt                  # Build config for benchmarks
│   ├── benchmark_kernels.cu            # Kernel benchmark definitions
│   ├── benchmark_utils.cu              # Shared benchmark utilities
│   ├── benchmark_harness.cpp           # Main entry point
│   └── configs/                        # Benchmark configurations
│       ├── standard.json               # Standard workload configs
│       └── regression.json             # Regression test configs
│
├── scripts/benchmark/                  # Python harness (NEW)
│   ├── run_benchmarks.py               # Main orchestration script
│   ├── collect_results.py              # JSON result collection
│   ├── check_regression.py             # Baseline comparison logic
│   ├── generate_dashboard.py           # HTML dashboard generation
│   ├── baselines/                      # Stored baselines (committed)
│   │   └── v0.1.0/
│   │       ├── reduce.json
│   │       ├── scan.json
│   │       └── matmul.json
│   └── templates/                      # Dashboard templates
│       └── dashboard.html
│
├── results/                            # Current run results (gitignore)
│   ├── 2026-04-26/
│   │   ├── reduce.json
│   │   └── matmul.json
│   └── baselines/
│
├── reports/                            # Generated dashboards
│   ├── index.html                      # Main dashboard
│   ├── trends/
│   │   ├── reduce_trend.html
│   │   └── matmul_trend.html
│   └── regression/
│       └── latest.html
│
├── include/cuda/benchmark/             # Existing (keep)
│   ├── benchmark.h                     # Custom harness (consider migrating to Google Benchmark)
│   └── nvtx.h                          # NVTX wrapper utilities (NEW)
│
├── src/benchmark/                      # Existing but empty (populate)
│   └── nvtx.cu                         # NVTX implementation (NEW)
│
├── include/cuda/algo/                  # Add NVTX annotations
│   ├── reduce.h                        # + NVTX_RANGE in functions
│   ├── scan.h
│   ├── sort.h
│   └── ...
│
├── tests/benchmark/                    # Existing benchmark tests
│   ├── benchmark_test.cpp              # Keep for regression checks
│   ├── throughput_test.cpp
│   └── regression_test.cpp
│
└── CMakeLists.txt                      # Update with benchmark targets
```

## Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| `benchmark/` directory | C++ Google Benchmark kernels | Compile as separate executable, link against `cuda_impl` |
| `scripts/benchmark/` | Python orchestration | argparse CLI, subprocess calls to benchmark executable |
| `scripts/benchmark/run_benchmarks.py` | Invoke benchmarks with configurable workloads | Python subprocess with JSON config parsing |
| `scripts/benchmark/check_regression.py` | Compare results against baselines | JSON diff with configurable tolerance |
| `scripts/benchmark/generate_dashboard.py` | Create HTML reports | Jinja2 templates, Chart.js for visualization |
| `include/cuda/benchmark/nvtx.h` | NVTX helper macros | C++ header with RAII range guards |
| `results/` | Transient result storage | JSON files, gitignored |
| `scripts/benchmark/baselines/` | Versioned baselines | Committed to repo for regression tracking |

## Architectural Patterns

### Pattern 1: Google Benchmark Kernel Registration

**What:** Standard Google Benchmark pattern where each kernel registers itself via `BENCHMARK` macro.

**When to use:** All C++ benchmark kernels should follow this pattern for consistency with Google's conventions.

**Trade-offs:**
- Pro: Built-in parameterization, statistical analysis, and JSON output
- Pro: Integration with existing Google infrastructure (gbench_install.cmake)
- Con: Additional dependency (but already using Google Test)

**Example:**
```cpp
// benchmark/benchmark_reduce.cu
#include <benchmark/benchmark.h>
#include "cuda/algo/reduce.h"

static void BM_ReduceFloat(benchmark::State& state) {
    const size_t n = state.range(0);
    // ... setup, warmup, benchmark loop
    for (auto _ : state) {
        cuda::algo::reduce(d_data.get(), result, n);
    }
    state.SetBytesProcessed(n * sizeof(float));
    state.SetItemsProcessed(n * state.iterations());
}
BENCHMARK(BM_ReduceFloat)->RangeMultiplier(2)->Ranges({{1024, 1<<24}});
BENCHMARK(BM_ReduceFloat)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
```

### Pattern 2: RAII NVTX Range Guard

**What:** Scoped NVTX annotation using RAII destructor semantics.

**When to use:** Annotating existing functions without changing their signature.

**Trade-offs:**
- Pro: Zero-cost abstraction, automatic end on scope exit
- Pro: Exception-safe (destructor runs even on exceptions)
- Con: Requires include of nvtx.h in all annotated files

**Example:**
```cpp
// include/cuda/benchmark/nvtx.h
#pragma once
#include <nvtx3/nvtx3.hpp>

class NVTXRange {
public:
    explicit NVTXRange(const char* name, nvtx3::color color = nvtx3::color::blue) 
        : marker_(name) {
        marker_.start();
    }
    ~NVTXRange() { marker_.end(); }
    
    // Prevent copying, allow moving
    NVTXRange(const NVTXRange&) = delete;
    NVTXRange& operator=(const NVTXRange&) = delete;
    NVTXRange(NVTXRange&&) = default;
    NVTXRange& operator=(NVTXRange&&) = default;

private:
    nvtx3::scoped_marker marker_;
};

// Macro for convenience
#define NVTX_SCOPED_RANGE(name) NVTXRange _nvtx_range_(name)
```

### Pattern 3: Configuration-Driven Benchmark Execution

**What:** JSON configuration files that control which benchmarks run with what parameters.

**When to use:** Standardizing benchmark execution across different environments (local, CI, release).

**Trade-offs:**
- Pro: Easy to add new configurations without code changes
- Pro: Reproducible benchmarks across runs
- Con: Additional parsing layer

**Example:**
```json
// benchmark/configs/standard.json
{
    "name": "standard",
    "description": "Standard workload for CI regression",
    "filters": ["*/Reduce/*", "*/Scan/*"],
    "workloads": [
        {"name": "small", "size": 1024},
        {"name": "medium", "size": 1024*1024},
        {"name": "large", "size": 1024*1024*64}
    ],
    "iterations": 10,
    "warmup": 3,
    "tolerance_percent": 10.0
}
```

### Pattern 4: Baseline Versioning

**What:** Store baselines with version tags for regression comparison.

**When to use:** Tracking performance across releases and identifying when regressions occurred.

**Trade-offs:**
- Pro: Clear history of performance expectations
- Pro: Easy to rollback baselines if needed
- Con: Requires discipline to update baselines on intentional changes

**Directory structure:**
```
scripts/benchmark/baselines/
├── v0.1.0/           # Initial baseline
│   └── reduce.json
├── v0.2.0/           # After optimization
│   └── reduce.json
└── main/             # Current main branch baseline (symlink or copy)
    └── reduce.json
```

### Pattern 5: Dashboard Generation with Trend Analysis

**What:** HTML dashboards with embedded Chart.js visualizations showing performance over time.

**When to use:** Visualizing benchmark trends and communicating performance to stakeholders.

**Trade-offs:**
- Pro: Self-contained HTML that can be served statically
- Pro: Interactive charts for exploration
- Con: Static generation means no real-time updates

**Data flow:**
```
Benchmark Run → JSON Results → Python Parser → Jinja2 Template → HTML Dashboard
                     ↓
              Baseline Comparison
                     ↓
              Regression Alerts
```

## Data Flow

### Benchmark Execution Flow

```
User runs: python scripts/benchmark/run_benchmarks.py --config standard --gpu 0

1. Python harness parses config
         ↓
2. Invokes ./build/bin/benchmark_kernels --benchmark_filter="*/Reduce/*" --benchmark_format=json
         ↓
3. C++ kernels execute with NVTX annotations (visible in Nsight Graphics/Compute)
         ↓
4. Google Benchmark outputs JSON to stdout or file
         ↓
5. Python harness parses JSON, stores in results/YYYY-MM-DD/
         ↓
6. Regression check compares against baselines/
         ↓
7. Dashboard generation creates HTML reports in reports/
```

### NVTX Integration Flow

```
Source Code (nova/algo/reduce.h)
    │
    ├─ #include "cuda/benchmark/nvtx.h"
    │
    └─ NVTX_SCOPED_RANGE("reduce_float")
           │
           ├─ nvtx3::scoped_marker created
           │
           ├─ marker.start() called (NVTX event pushed)
           │
           ├─ Actual reduce operation executes
           │
           └─ ~NVTXRange() destructor calls marker.end()
                   │
                   └─ NVTX event popped
```

### Multi-GPU/Multi-Node Benchmark Flow

```
# Single GPU benchmark
python scripts/benchmark/run_benchmarks.py --gpu 0 --config single_gpu

# Multi-GPU benchmark
python scripts/benchmark/run_benchmarks.py --gpu all --config multi_gpu --nnodes 2

# For multi-node, SSH to each node and invoke via MPI
mpirun -n 4 python scripts/benchmark/run_benchmarks.py --gpu 0 --config multi_node
```

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| NVIDIA Nsight Compute | nvcc --generate-line-info + NVTX ranges | Profile kernel-level performance |
| NVIDIA Nsight Systems | NVTX annotations visible in timeline | System-wide timeline view |
| CUDA Profiler | NVTX domains categorize ranges | Use nvtx.* options for profiling |
| GitHub Actions | Python harness exit codes | Non-zero on regression detection |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| benchmark/ → cuda_impl | Link-time dependency | Benchmarks link against cuda_impl library |
| Python harness → benchmark binary | stdin/stdout JSON | Subprocess invocation |
| scripts/benchmark/ → results/ | File system | JSON files written/read |
| Dashboard generator → results/ | File system | Reads and transforms data |

### Layer Integration (Existing Codebase)

```
┌──────────────────────────────────────────────────────────────┐
│                    algo layer (nova/algo/)                   │
├──────────────────────────────────────────────────────────────┤
│  reduce.h     → benchmark_reduce.cpp invokes cuda::algo::reduce│
│  scan.h       → benchmark_scan.cpp invokes cuda::algo::scan    │
│  sort.h       → benchmark_sort.cpp invokes cuda::algo::sort    │
│                                                              │
│  + NVTX_SCOPED_RANGE("reduce") annotations added to headers   │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│               distributed layer (nova/distributed/)          │
├──────────────────────────────────────────────────────────────┤
│  all_reduce.h → benchmark_distributed.cpp tests NCCL ops     │
│                                                              │
│  + NVTX_SCOPED_RANGE("all_reduce") for collective ops        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                memory layer (nova/memory/)                   │
├──────────────────────────────────────────────────────────────┤
│  memory_pool.h → benchmark_memory.cpp tests pool alloc       │
│                                                              │
│  + NVTX_SCOPED_RANGE("pool_alloc") for memory ops            │
└──────────────────────────────────────────────────────────────┘
```

## Anti-Patterns

### Anti-Pattern 1: Benchmarking Without Warm-up

**What people do:** Run benchmarks immediately without GPU warm-up, leading to inflated first-run times.

**Why it's wrong:** CUDA kernels experience initialization overhead on first launch (kernel compilation, cache warming).

**Do this instead:**
```cpp
// Google Benchmark handles warmup automatically
// For custom harness, ensure warmup iterations before measurement:
for (int i = 0; i < options.warmup_iterations; ++i) {
    kernel();
}
cudaDeviceSynchronize();
```

### Anti-Pattern 2: Ignoring Variance

**What people do:** Reporting only mean execution time without standard deviation.

**Why it's wrong:** GPU performance varies due to dynamic frequency scaling, memory timing, and kernel launch overhead.

**Do this instead:**
```cpp
// Use Google Benchmark's built-in statistics
// Or custom implementation:
BenchmarkResult result = compute_statistics(measurements);
std::cout << result.mean_ms << " +/- " << result.stddev_ms << " ms\n";
```

### Anti-Pattern 3: Benchmarking Small Data Sizes

**What people do:** Testing with sizes too small to saturate GPU resources.

**Why it's wrong:** GPU benefits only manifest at sufficient parallelism; small workloads hide algorithmic inefficiencies.

**Do this instead:**
```cpp
// Test across meaningful sizes
BENCHMARK(BM_Kernel)->Range(1<<10, 1<<28);  // 1KB to 256MB
// Include throughput metric (GB/s) not just latency
state.SetBytesProcessed(data_size * state.iterations());
```

### Anti-Pattern 4: Not Isolating Benchmark Code

**What people do:** Mixing benchmark logic with production code in the same translation unit.

**Why it's wrong:** Benchmark instrumentation (timing, NVTX) adds overhead to production builds.

**Do this instead:**
```cpp
// Production: include headers only
#include "cuda/algo/reduce.h"
auto result = cuda::algo::reduce(...);

// Benchmark: separate executable
// benchmark/benchmark_reduce.cpp
// - Includes same headers
// - Adds BENCHMARK macros
// - Compiled as separate target
```

### Anti-Pattern 5: Hardcoding Regression Thresholds

**What people do:** Using tight tolerances (e.g., 1%) that trigger false positives.

**Why it's wrong:** GPU performance has inherent variance; tight thresholds cause flaky CI failures.

**Do this instead:**
```cpp
// Use reasonable tolerances based on operation type
constexpr double TOLERANCE_MEMORY_OPS = 5.0;    // Memory ops are stable
constexpr double TOLERANCE_COMPUTE = 10.0;       // Compute varies more
constexpr double TOLERANCE_DISTRIBUTED = 15.0;   // Network-bound ops vary most
```

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single GPU, single benchmark | Simple: single binary, direct JSON output |
| Multi-GPU benchmarks | Add `--gpu` flag, aggregate results from each GPU |
| Multi-node benchmarks | Use MPI wrapper, results aggregated via rank 0 |
| Large test matrix | Parallelize benchmark runs via `xargs -P` or Python `concurrent.futures` |

### Scaling Priorities

1. **First bottleneck: CI time** — Use selective filtering to run only changed benchmarks in PRs
2. **Second bottleneck: Result storage** — Prune old results, keep only baseline-tagged versions

## CMake Integration

The benchmark targets should integrate with the existing CMake setup:

```cmake
# benchmark/CMakeLists.txt
find_package(benchmark REQUIRED)

add_executable(nova_benchmarks
    benchmark_kernels.cu
    benchmark_utils.cu
    benchmark_harness.cpp
)

target_link_libraries(nova_benchmarks PRIVATE
    cuda_impl
    benchmark::benchmark
    CUDA::cudart
)

target_include_directories(nova_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install benchmark binary for Python harness to invoke
install(TARGETS nova_benchmarks DESTINATION bin)
```

## Recommended Execution Flow

```bash
# Full benchmark suite
python scripts/benchmark/run_benchmarks.py --all --output results/latest

# Regression check against baselines
python scripts/benchmark/run_benchmarks.py --config regression --check

# Generate dashboard
python scripts/benchmark/generate_dashboard.py --results results/latest --output reports/

# Update baselines (after intentional changes)
python scripts/benchmark/update_baselines.py --results results/latest --version v0.2.0
```

## Sources

- [Google Benchmark Documentation](https://google.github.io/benchmark/)
- [NVTX Documentation](https://docs.nvidia.com/gameworks/content/gameworkslibrary/cudart/nvtx3.html)
- [NVIDIA Nsight Systems profiling](https://developer.nvidia.com/nsight-systems)
- [CMU 15-418 GPU benchmarking best practices](https://www.cs.cmu.edu/~gahan/15418/)
- [Google Benchmark GitHub](https://github.com/google/benchmark)

---

*Architecture research for: CUDA/C++ Benchmarking Infrastructure*
*Researched: 2026-04-26*
