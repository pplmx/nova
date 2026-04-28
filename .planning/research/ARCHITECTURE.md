# Architecture Research: Algorithm Domain Integration

**Domain:** Nova CUDA Library - Algorithm Extension Architecture
**Researched:** 2026-04-28
**Confidence:** MEDIUM-HIGH
**For:** v2.3 Extended Algorithms

## Executive Summary

This document maps four new algorithm domains (Sorting/Searching, Linear Algebra Extras, Numerical Methods, Signal Processing) onto the existing five-layer CUDA architecture. The existing infrastructure—memory pools, stream management, device capabilities—is well-suited for extension, requiring minimal modification. Each domain should follow a consistent pattern: thin API wrappers in domain-specific namespaces, leveraging existing memory/device layers, with algorithms implemented as composable primitives.

## Current Five-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          API Layer                                   │
│              (include/cuda/api/) - Public interface                 │
│                     Depends on: algo layer                          │
├─────────────────────────────────────────────────────────────────────┤
│                       Algorithm Layer                                │
│         (include/cuda/algo/) - Parallel algorithm wrappers          │
│                     Uses: device, memory                            │
├─────────────────────────────────────────────────────────────────────┤
│                        Device Layer                                  │
│           (include/cuda/device/) - Device management                │
│              Shared: reduce kernels, warp/block primitives          │
├─────────────────────────────────────────────────────────────────────┤
│                       Memory Layer                                   │
│           (include/cuda/memory/) - Buffer, MemoryPool               │
│         Reusable by: all algorithm domains                          │
├─────────────────────────────────────────────────────────────────────┤
│                        Stream Layer                                  │
│            (include/cuda/stream/) - Async operations                │
│         Reusable by: all algorithm domains                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Domain Mapping to Five-Layer Architecture

### 1. Sorting & Searching

| Layer | Mapping | Rationale |
|-------|---------|-----------|
| **Memory** | Reuse `Buffer<T>`, `MemoryPool` | Sorting requires temporary buffers; existing memory layer provides RAII abstractions |
| **Device** | Extend with comparison/swap primitives | Warp-level primitives exist in `device_utils.h`; extend for sort-specific ops |
| **Algo** | New `cuda/sort/` namespace | Sort algorithms (radix, bitonic, merge sort) as algo primitives |
| **API** | New `cuda/sort/` public headers | User-facing API for sorting, searching, top-k |

**Architecture:**
```
include/cuda/sort/
├── sort.h              # Public API (radix sort, top-k)
├── search.h            # Binary search, partition
├── types.h             # Sort config, comparator types
└── sort_kernels.cuh    # Device kernels (internal)

src/cuda/sort/
├── sort.cu             # Implementation
└── search.cu           # Search implementation
```

**Integration with Existing:**
- Reuse `memory::Buffer<T>` for working memory
- Extend `device::ReduceOp` with comparison operations
- Stream-based async sorting via `stream::Stream`

### 2. Linear Algebra Extras (SVD, Eigendecomposition, Factorization)

| Layer | Mapping | Rationale |
|-------|---------|-----------|
| **Memory** | Reuse `Buffer<T>`, extend for matrix storage | Matrix factorization requires contiguous matrix buffers |
| **Device** | Reuse `cuda::neural::MatmulOptions` pattern | Already wraps cuBLAS; extend for MAGMA/LAPACK integration |
| **Algo** | New `cuda/linalg/` namespace | SVD, eigenvalue decomposition as algo primitives |
| **API** | New `cuda/linalg/` public headers | User-facing API for matrix factorization |

**Architecture:**
```
include/cuda/linalg/
├── svd.h               # SVD decomposition (full, thin, truncated)
├── eigendecomposition.h # Eigenvalue/eigenvector computation
├── qr.h                # QR decomposition
├── cholesky.h          # Cholesky factorization
├── factorization.h     # LU, LDLT decomposition
└── types.h             # Matrix layout, job options

src/cuda/linalg/
├── svd.cu              # SVD implementation via cuSOLVER
├── eigendecomposition.cu
├── qr.cu
└── cholesky.cu
```

**Integration with Existing:**
- Reuse cuBLAS handle pattern from `neural/matmul.h`
- Extend `cuda::stream::Stream` for factorization plans
- Reuse `memory::Buffer<T>` for matrix storage

**External Dependencies:**
- **cuSOLVER** (required): Dense linear algebra, already part of CUDA toolkit
- **MAGMA** (optional): GPU-accelerated LAPACK for advanced factorization

### 3. Numerical Methods (Monte Carlo, Integration, Root Finding)

| Layer | Mapping | Rationale |
|-------|---------|-----------|
| **Memory** | Reuse `Buffer<T>`, `MemoryPool` | Monte Carlo requires large random number buffers |
| **Device** | New PRNG device utilities | Extend device layer with curand-based RNG |
| **Algo** | New `cuda/numeric/` namespace | Numerical algorithms as algo primitives |
| **API** | New `cuda/numeric/` public headers | User-facing API for numerical methods |

**Architecture:**
```
include/cuda/numeric/
├── monte_carlo.h       # MC integration, simulation
├── integration.h       # Numerical integration (trapezoid, Simpson, quadrature)
├── root_finding.h      # Bisection, Newton-Raphson, Brent
├── interpolation.h     # Linear, cubic spline, Lagrange
├── random.h            # PRNG utilities (curand wrapper)
└── types.h             # Config structures

src/cuda/numeric/
├── monte_carlo.cu      # Monte Carlo implementation
├── integration.cu      # Numerical integration
├── root_finding.cu     # Root finding algorithms
├── interpolation.cu
└── random.cu           # PRNG implementation
```

**Integration with Existing:**
- Reuse `memory::Buffer<T>` for large-scale Monte Carlo
- Extend stream management for parallel task execution
- Reuse reduction primitives from device layer

### 4. Signal Processing (Wavelets, Filters, Convolution)

| Layer | Mapping | Rationale |
|-------|---------|-----------|
| **Memory** | Reuse `Buffer<T>` for signal data | Signal processing operates on 1D/2D arrays |
| **Device** | Extend with FFT/filter primitives | Reuse FFT plan pattern from `cuda/fft/` |
| **Algo** | New `cuda/signal/` namespace | Wavelet transforms, FIR/IIR filters as primitives |
| **API** | New `cuda/signal/` public headers | User-facing API for signal processing |

**Architecture:**
```
include/cuda/signal/
├── wavelet.h           # DWT, IDWT (Haar, Daubechies, etc.)
├── filter.h            # FIR, IIR filters, filter design
├── convolution.h       # Direct, overlap-add, overlap-save
├── spectral.h          # Power spectrum, spectrogram
└── types.h             # Signal types, window functions

src/cuda/signal/
├── wavelet.cu          # Wavelet transform implementation
├── filter.cu           # Filter implementation
├── convolution.cu      # Convolution algorithms
└── spectral.cu         # Spectral analysis
```

**Integration with Existing:**
- Reuse FFT infrastructure from `cuda/fft/`
- Reuse `Buffer<T>` for signal storage
- Extend `stream::Stream` for pipeline processing

## Integration Points Analysis

### 1. Memory Pool Reuse

**Can algorithms reuse existing memory pool?**
- **YES, with minor extensions**

| Algorithm Domain | Buffer Pattern | Pool Compatibility |
|-----------------|----------------|-------------------|
| Sorting | Working buffers, temporary storage | Full compatibility |
| Linear Algebra | Matrix buffers, workspace | Full compatibility |
| Numerical Methods | Large simulation arrays | Full compatibility |
| Signal Processing | Signal buffers, FFT workspace | Full compatibility |

**Required Extensions:**
```cpp
// Extend MemoryPool for algorithm-specific patterns
namespace cuda::memory {
    // Existing pattern - fully reusable
    Buffer<float> workspace(1024);
    
    // Algorithm-specific: pre-allocated scratch space
    class ScratchPool {
    public:
        void* allocate_scratch(size_t bytes);
        void release_scratch();
        
        // Support for algorithm-specific alignment requirements
        void* allocate_aligned(size_t bytes, size_t alignment);
    };
}
```

### 2. Device/Stream Management

**Can algorithms share device/stream management?**
- **YES, fully compatible**

| Stream Feature | Current Status | Algorithm Need |
|----------------|----------------|----------------|
| `Stream` class | Available | Extend with algorithm-specific options |
| Async operations | Supported | All domains benefit |
| Event synchronization | Available | Required for multi-kernel algorithms |
| Priority streams | Available | Useful for numerical methods |

**Integration Pattern:**
```cpp
// All algorithms use existing stream pattern
namespace cuda::stream {
    // Existing - fully reusable
    auto stream = make_stream();
    
    // Algorithm can accept external streams
    void sort_async(float* data, size_t n, cudaStream_t stream);
    void svd_async(const float* A, float* U, float* S, float* VT, 
                   cudaStream_t stream);
    void mc_simulate(float* results, size_t n, cudaStream_t stream);
}
```

### 3. Standalone vs. Integrated with Algo Layer

**Decision Framework:**

| Criterion | Sorting | Linear Algebra | Numerical Methods | Signal Processing |
|-----------|---------|----------------|-------------------|-------------------|
| Reuses device primitives | Yes | Partial (cuBLAS) | Yes | Partial (FFT) |
| Reuses memory patterns | Yes | Yes | Yes | Yes |
| Requires external libs | No (thrust) | Yes (cuSOLVER) | No (thrust) | No (thrust) |
| Fits algo layer | Yes | **Standalone** | Yes | **Standalone** |

**Recommendation:**
- **Sorting**: Integrate into `cuda::algo` layer (pure CUDA, no external deps)
- **Linear Algebra Extras**: **Standalone** `cuda::linalg` layer (depends on cuSOLVER, distinct from algo primitives)
- **Numerical Methods**: Integrate into `cuda::algo` layer (pure CUDA/thrust)
- **Signal Processing**: **Standalone** `cuda::signal` layer (FFT specialization, distinct from algo primitives)

## New Components vs. Modifications

### New Components (Create)

| Component | Location | Purpose | Dependencies |
|-----------|----------|---------|--------------|
| `cuda_sort` | `include/cuda/sort/`, `src/cuda/sort/` | Sorting and searching algorithms | memory, device |
| `cuda_linalg` | `include/cuda/linalg/`, `src/cuda/linalg/` | Linear algebra extras | memory, device, cuBLAS, cuSOLVER |
| `cuda_numeric` | `include/cuda/numeric/`, `src/cuda/numeric/` | Numerical methods | memory, device, curand |
| `cuda_signal` | `include/cuda/signal/`, `src/cuda/signal/` | Signal processing | memory, device, fft |

### Modifications to Existing (Minimal)

| File | Change | Rationale |
|------|--------|-----------|
| `include/cuda/device/device_utils.h` | Add comparison/swap primitives | Required for sorting algorithms |
| `include/cuda/device/device_utils.h` | Add random number generation helpers | Required for Monte Carlo |
| `include/cuda/algo/kernel_launcher.h` | Extend for algorithm-specific launch patterns | Optional, improves DX |
| `CMakeLists.txt` | Add new library targets (sort, linalg, numeric, signal) | Build system integration |

### No Changes Required

| Layer | Status | Reason |
|-------|--------|--------|
| `memory/Buffer<T>` | Unchanged | Already provides required abstractions |
| `memory/MemoryPool` | Unchanged | Fully reusable |
| `stream/Stream` | Unchanged | Accepts external streams |
| `stream/Event` | Unchanged | Fully reusable |
| `error/cuda_error.hpp` | Unchanged | Wraps all CUDA APIs |

## Recommended Build Order

### Phase Dependencies Graph

```
Phase 1: Foundation
    │
    ├── Sorting Infrastructure ──────────────────┐
    │   (device primitives, Buffer usage)        │
    │                                           │
    ├── Numerical Methods Foundation ───────────┤
    │   (random numbers, array operations)      │
    │                                           │
    └── Signal Processing Foundation ───────────┤
        (FFT extension, filter primitives)      │
                                                  │
                                                  ▼
Phase 2: Core Algorithms                    Phase 3: Linear Algebra
    │                                           │
    ├── Sorting Algorithms ─────────────────────┼──►
    │   (radix sort, top-k)                     │    (SVD, eigenvalue, QR)
    │                                           │
    ├── Numerical Methods ──────────────────────┤
    │   (Monte Carlo, integration)              │
    │                                           │
    └── Signal Processing Core ────────────────┤
        (wavelets, basic filtering)             │
                                                  │
                                                  ▼
Phase 4: Advanced Features
    │
    ├── Sorting: Binary search, partition
    ├── Linear Algebra: Cholesky, factorization
    ├── Numerical Methods: Root finding, interpolation
    └── Signal Processing: Advanced filters, convolution

Phase 5: Integration
    │
    ├── Performance profiling
    ├── Memory optimization
    └── API polish
```

### Detailed Build Order

**Phase 1: Device Layer Extensions** (Week 1-2)
```
1. Extend include/cuda/device/device_utils.h
   - Add comparison primitives (>, <, swap)
   - Add warp-level sorting primitives
   - Add PRNG device utilities

2. Create include/cuda/numeric/random.h
   - Wrap cuRAND for device-side random numbers
   - Provide common distributions (uniform, normal)
```

**Phase 2: Sorting & Searching** (Week 2-3)
```
3. Create include/cuda/sort/types.h
4. Create include/cuda/sort/sort.h
5. Create src/cuda/sort/sort.cu
   - Radix sort (primary)
   - Top-k selection
6. Create include/cuda/sort/search.h
7. Create src/cuda/sort/search.cu
   - Binary search
   - Partition
8. Add tests for sorting algorithms
```

**Phase 3: Numerical Methods** (Week 3-4)
```
9. Create include/cuda/numeric/types.h
10. Create include/cuda/numeric/monte_carlo.h
11. Create src/cuda/numeric/monte_carlo.cu
    - MC integration
    - Option pricing framework
12. Create include/cuda/numeric/integration.h
13. Create src/cuda/numeric/integration.cu
    - Trapezoid rule
    - Simpson's rule
    - Gaussian quadrature
14. Create include/cuda/numeric/root_finding.h
15. Create src/cuda/numeric/root_finding.cu
    - Bisection
    - Newton-Raphson
    - Brent's method
16. Add tests for numerical methods
```

**Phase 4: Linear Algebra Extras** (Week 4-6)
```
17. Create include/cuda/linalg/types.h
18. Create include/cuda/linalg/svd.h
19. Create src/cuda/linalg/svd.cu
    - Full SVD (gesvd)
    - Truncated SVD for dimensionality reduction
20. Create include/cuda/linalg/eigendecomposition.h
21. Create src/cuda/linalg/eigendecomposition.cu
    - Eigenvalue computation (syevd, geevd)
    - Eigenvectors
22. Create include/cuda/linalg/qr.h
23. Create src/cuda/linalg/qr.cu
24. Create include/cuda/linalg/cholesky.h
25. Create src/cuda/linalg/cholesky.cu
26. Add tests for linear algebra
```

**Phase 5: Signal Processing** (Week 6-8)
```
27. Create include/cuda/signal/types.h
28. Create include/cuda/signal/wavelet.h
29. Create src/cuda/signal/wavelet.cu
    - Haar wavelet
    - Daubechies wavelets
30. Create include/cuda/signal/filter.h
31. Create src/cuda/signal/filter.cu
    - FIR filter
    - IIR filter (optional)
32. Create include/cuda/signal/convolution.h
33. Create src/cuda/signal/convolution.cu
    - Direct convolution
    - FFT-based convolution
    - Overlap-add/save
34. Add tests for signal processing
```

**Phase 6: Integration & Optimization** (Week 8-10)
```
35. Add CUDA library dependency management
    - Link cuSOLVER for linalg
    - Link cuRAND for numeric
36. Add performance benchmarks
37. Add memory optimization
    - Scratch space reuse
    - Stream-based parallelism
38. API documentation
39. Integration tests
```

## CMake Integration

### Required CMake Additions

```cmake
# Add to CMakeLists.txt

# New algorithm libraries
set(CUDA_SORT_DIR ${CMAKE_SOURCE_DIR}/include/cuda/sort)
set(SRC_CUDA_SORT ${CMAKE_SOURCE_DIR}/src/cuda/sort)

set(CUDA_NUMERIC_DIR ${CMAKE_SOURCE_DIR}/include/cuda/numeric)
set(SRC_CUDA_NUMERIC ${CMAKE_SOURCE_DIR}/src/cuda/numeric)

set(CUDA_LINALG_DIR ${CMAKE_SOURCE_DIR}/include/cuda/linalg)
set(SRC_CUDA_LINALG ${CMAKE_SOURCE_DIR}/src/cuda/linalg)

set(CUDA_SIGNAL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/signal)
set(SRC_CUDA_SIGNAL ${CMAKE_SOURCE_DIR}/src/cuda/signal)

# Source file sets
set(SORT_SOURCES
    ${SRC_CUDA_SORT}/sort.cu
    ${SRC_CUDA_SORT}/search.cu
)

set(NUMERIC_SOURCES
    ${SRC_CUDA_NUMERIC}/monte_carlo.cu
    ${SRC_CUDA_NUMERIC}/integration.cu
    ${SRC_CUDA_NUMERIC}/root_finding.cu
    ${SRC_CUDA_NUMERIC}/random.cu
)

set(LINALG_SOURCES
    ${SRC_CUDA_LINALG}/svd.cu
    ${SRC_CUDA_LINALG}/eigendecomposition.cu
    ${SRC_CUDA_LINALG}/qr.cu
    ${SRC_CUDA_LINALG}/cholesky.cu
)

set(SIGNAL_SOURCES
    ${SRC_CUDA_SIGNAL}/wavelet.cu
    ${SRC_CUDA_SIGNAL}/filter.cu
    ${SRC_CUDA_SIGNAL}/convolution.cu
)

# Add to cuda_impl library sources
set(ALL_CUDA_SOURCES
    ${SORT_SOURCES}
    ${NUMERIC_SOURCES}
    ${LINALG_SOURCES}
    ${SIGNAL_SOURCES}
    # ... existing sources
)
```

### Library Target Hierarchy

```
cuda_impl (static library)
├── cuda_memory (INTERFACE)
├── cuda_device (INTERFACE)
├── cuda_algo (INTERFACE)
├── cuda_sort (STATIC) ────────┐
│   └── cuda_memory, cuda_device
├── cuda_numeric (STATIC) ────┤
│   └── cuda_memory, cuda_device, CUDA::curand
├── cuda_linalg (STATIC) ──────┼── cuda_impl
│   └── cuda_memory, cuda_device, CUDA::cublas, CUDA::cusolver
└── cuda_signal (STATIC) ──────┘
    └── cuda_memory, cuda_device, CUDA::cufft
```

## Architecture Patterns

### Pattern 1: Algorithm Domain with External Library Dependency

**When:** Algorithm requires cuSOLVER, cuFFT, cuRAND, etc.

**Structure:**
```cpp
// include/cuda/linalg/svd.h
#pragma once

#include <cusolverDn.h>
#include "cuda/device/error.h"
#include "cuda/memory/buffer.h"

namespace cuda::linalg {

struct SVDOptions {
    cusolverDnHandle_t handle = nullptr;  // cuSOLVER handle
    int workspace_bytes = 0;
    void* workspace = nullptr;
    bool truncated = false;
    int k = 0;  // Rank for truncated SVD
};

class SVDPlan {
public:
    SVDPlan(int m, int n);
    ~SVDPlan();
    
    // Prevent copying (GPU resources)
    SVDPlan(const SVDPlan&) = delete;
    SVDPlan& operator=(const SVDPlan&) = delete;
    
    // Allow moving
    SVDPlan(SVDPlan&& other) noexcept;
    SVDPlan& operator=(SVDPlan&& other) noexcept;
    
    void compute(const float* A, float* U, float* S, float* VT,
                 cudaStream_t stream = nullptr);
    
private:
    cusolverDnHandle_t handle_;
    int m_, n_;
    float* d_work_ = nullptr;
    int* d_info_ = nullptr;
};

}  // namespace cuda::linalg
```

### Pattern 2: Pure CUDA Algorithm (No External Dependencies)

**When:** Algorithm can be implemented with CUDA kernels only

**Structure:**
```cpp
// include/cuda/sort/sort.h
#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/stream/stream.h"

namespace cuda::sort {

template <typename T>
void radix_sort(Buffer<T>& data, bool ascending = true);

template <typename T>
void top_k(const T* input, T* output, size_t n, size_t k);

void binary_search(const int* sorted, int value, size_t size, int* result);

// Async variants
template <typename T>
void radix_sort_async(Buffer<T>& data, cudaStream_t stream);

}  // namespace cuda::sort
```

### Pattern 3: Stateless Algorithm Functions

**When:** Simple algorithm that doesn't need persistent state

**Structure:**
```cpp
// include/cuda/numeric/integration.h
#pragma once

#include "cuda/memory/buffer.h"

namespace cuda::numeric {

// Numerical integration functions
float trapezoid(const float* f, size_t n, float a, float b);
float simpson(const float* f, size_t n, float a, float b);
float gauss_legendre(const float* f, size_t n, float a, float b);

// Async versions
void trapezoid_async(const float* f, float* result, size_t n, 
                     float a, float b, cudaStream_t stream = nullptr);

}  // namespace cuda::numeric
```

## Anti-Patterns

### Anti-Pattern 1: Duplicate Memory Management

**What:** Each algorithm domain creates its own memory allocation pattern.

**Why bad:** Inconsistent API, potential memory leaks, no unified memory pool.

**Do this instead:** Require all algorithms to use `cuda::memory::Buffer<T>` or accept device pointers.

### Anti-Pattern 2: Hard-coded Stream Management

**What:** Algorithm manages its own streams internally.

**Why bad:** No user control over async behavior, stream pool exhaustion.

**Do this instead:** Accept `cudaStream_t` as parameter, use default stream as fallback.

### Anti-Pattern 3: Algorithm-Specific Error Types

**What:** Each algorithm domain defines its own error enum.

**Why bad:** Inconsistent error handling for users.

**Do this instead:** Use `std::error_code` with categories per library (e.g., `cuda::linalg::error_category`).

### Anti-Pattern 4: Tight Coupling to External Libraries

**What:** Algorithm directly calls cuSOLVER/cuFFT internal functions.

**Why bad:** API changes in CUDA toolkit break user code.

**Do this instead:** Wrap external library calls in opaque plan classes.

## Scalability Considerations

| Scale | Architecture Adjustments |
|-------|-------------------------|
| Single GPU | Standard implementation, no changes needed |
| Multi-GPU | Sorting benefits from multi-GPU; numerical methods parallelize over streams |
| Multi-Node | Linear algebra (SVD) can distribute across nodes; Monte Carlo parallelizes trivially |

### Future Extension Points

1. **Multi-GPU Sorting**: Extend `sort` namespace with `multi_gpu_sort()` using NCCL
2. **Distributed Linear Algebra**: Extend `linalg` namespace for SVD across nodes
3. **Real-time Signal Processing**: Add streaming filter API for pipeline processing

## Sources

- [cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [cuRAND Documentation](https://docs.nvidia.com/curand/)
- [Thrust Algorithms](https://nvidia.github.io/thrust/)
- Existing Nova codebase patterns
- CUDA Best Practices Guide

---

*Architecture research for: Nova CUDA Library v2.3 Extended Algorithms*
*Researched: 2026-04-28*
