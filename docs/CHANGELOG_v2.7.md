# Nova v2.7 Testing & Validation

**Version:** 2.7
**Date:** 2026-04-30

## Overview

This release adds comprehensive robustness testing, enhanced profiling capabilities, and new algorithm implementations to the Nova CUDA library.

## New Features

### Observability & Profiling (Phase 75)

#### Timeline Visualization

Export NVTX annotations to Chrome trace format for visualization in `chrome://tracing`.

```cpp
#include "cuda/observability/timeline.h"

NOVA_TIMELINE_SCOPED("kernel_name", "category");
// ... GPU operations ...
NOVA_TIMELINE_EXPORT("/path/to/trace.json");
```

#### Memory Bandwidth Measurement

Measure H2D, D2H, and D2D memory bandwidth.

```cpp
#include "cuda/observability/bandwidth_tracker.h"

cuda::observability::BandwidthTracker tracker;
auto result = tracker.measure_host_to_device(1024 * 1024);
std::cout << "Bandwidth: " << result.bandwidth_gbps << " GB/s\n";
```

#### Kernel Statistics

Collect per-kernel latency, throughput, and occupancy.

```cpp
#include "cuda/observability/kernel_stats.h"

cuda::observability::KernelStatsCollector collector;
collector.record_kernel("my_kernel", start, end, blocks, threads);
auto stats = collector.get_stats("my_kernel");
```

#### Occupancy Analyzer

Real-time block size recommendations.

```cpp
#include "cuda/observability/occupancy_analyzer.h"

cuda::observability::OccupancyAnalyzer analyzer;
auto rec = analyzer.recommend(kernel_func, dynamic_smem);
std::cout << "Optimal block size: " << rec.recommended_block_size << "\n";
```

### Algorithm Extensions (Phase 76)

#### Segmented Sort

Sort elements within groups without full array copy.

```cpp
#include "cuda/algo/segmented_sort.h"

cuda::algo::segmented::sort_by_key(keys, segment_ids, out_keys, out_segments,
                                    count, num_segments);
```

#### Sparse Matrix-Vector Multiply

SpMV using CSR/CSC formats from v2.1.

```cpp
#include "cuda/algo/spmv.h"

cuda::algo::spmv::multiply_csr(values, row_offsets, col_indices, x, y, num_rows);
```

#### Sample Sort

For large datasets beyond radix sort efficiency.

```cpp
#include "cuda/algo/sample_sort.h"

auto result = cuda::algo::sample_sort::sort_large_dataset(input, count);
```

#### Delta-Stepping SSSP

Single-source shortest path for weighted graphs.

```cpp
#include "cuda/algo/sssp.h"

auto distances = cuda::algo::sssp::compute_distances(graph, source);
```

### Robustness & Testing (Phase 77)

#### Memory Safety Validation

```cpp
#include "cuda/testing/memory_safety.h"

MemorySafetyValidator::instance().validate_allocation(ptr, size);
```

#### Test Isolation

```cpp
#include "cuda/testing/test_isolation.h"

TestIsolationContext::execute_isolated([]() {
    // Test code with isolated CUDA context
});
```

#### Layer-Aware Error Injection

```cpp
#include "cuda/testing/layer_error_injection.h"

auto& injector = LayerAwareErrorInjector::instance();
injector.inject_at_layer(LayerBoundary::Memory,
                         cuda::production::ErrorTarget::Allocation,
                         cudaErrorMemoryAllocation);
```

#### Boundary Condition Testing

```cpp
#include "cuda/testing/boundary_testing.h"

is_warp_aligned(size);           // Check 32-byte alignment
is_memory_aligned(ptr);          // Check 256-byte alignment
is_valid_block_size(dim3(256, 1, 1));  // Check valid block size
```

#### FP Determinism Control

```cpp
#include "cuda/testing/fp_determinism.h"

FPDeterminismControl::instance().set_level(DeterminismLevel::GpuToGpu);
```

## Build Requirements

- C++23
- CUDA 20
- CMake 4.0+

## Dependencies

- CCCL 2.6.0+ (replaces archived CUB)
- Google Test for testing
- Optional: NVIDIA Compute Sanitizer for memory validation

## Migration Notes

### CUB to CCCL Migration

Replace CUB includes:

```cpp
// Before
#include <cub/cub.cuh>

// After
#include <thrust/system/cuda/detail/cccl/cub.cuh>
```

## Performance Baselines

| Operation          | Baseline           |
| ------------------ | ------------------ |
| Timeline export    | < 1ms overhead     |
| Memory bandwidth   | ~90% peak HBM      |
| Kernel stats       | < 0.1ms per kernel |
| Occupancy analysis | < 5ms              |

## Testing

This section records how the v2.7 release was built. `NOVA_BUILD_TESTS` no longer
exists: tests and benchmarks are built by default and disabled together with
`-DNOVA_BUILD_TESTS_AND_BENCHMARKS=OFF`.

```bash
# Build with tests (v2.7: -DNOVA_BUILD_TESTS=ON; current trees need no flag)
cmake -B build
cmake --build build

# Run observability tests
ctest -R observability

# Run algorithm tests
ctest -R algo_

# Run robustness tests
ctest -R robustness
```

## Known Limitations

- FP determinism GPU-to-GPU requires matching CUDA version and GPU architecture
- Memory safety validation uses poison patterns; Compute Sanitizer CLI provides more comprehensive checking
- Sample sort threshold is configurable; adjust based on dataset characteristics
