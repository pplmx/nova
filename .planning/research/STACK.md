# Technology Stack for v2.8 Numerical Computing & Performance

**Analysis Date:** 2026-05-01
**Confidence:** MEDIUM-HIGH
**Note:** Some library capabilities require verification with current CUDA 12.x documentation; cuSPARSE iterative solver support has evolved.

---

## Executive Summary

v2.8 adds three capability areas: GPU-based iterative linear solvers (Krylov methods), Roofline performance model, and advanced sparse matrix formats (ELL/HYB). 

**Key finding:** NVIDIAcuSOLVER does NOT provide general iterative solvers (CG, GMRES, BiCGSTAB) — these must be implemented using cuBLAS SpMV primitives. cuSPARSE provides native ELL format support and conversion to HYB. The Roofline model builds on existing bandwidth measurement from v2.7.

---

## 1. Krylov Solver Stack Requirements

### What NVIDIA Provides

| Library | CG | GMRES | BiCGSTAB | Notes |
|---------|-----|-------|----------|-------|
| cuSOLVER | No | No | No | Provides only direct solvers (LU, QR, Cholesky, SVD) |
| cuSPARSE | No | No | No | Provides SpMV primitives, not iterative solvers |
| AmgX | Yes | Yes | Yes | NVIDIA's algebraic multigrid library, separate download |

### Recommendation: Custom Implementation using cuBLAS/cuSPARSE

**Rationale:** AmgX is a separate product requiring enterprise licensing and is overly complex for basic iterative solvers. CG, GMRES, and BiCGSTAB can be implemented using existing SpMV from v2.7 plus cuBLAS BLAS-1/2 operations (dot product, axpy, norm).

**Stack additions needed:**

| Component | Source | Purpose | Integration |
|-----------|--------|---------|-------------|
| cuBLAS | CUDA toolkit (already linked) | BLAS-1/2 operations for Krylov kernels | `CUDA::cublas` already in `cuda_impl` |
| Existing SpMV | v2.7 CSR/CSC | Sparse matrix-vector products | Already in `src/algo/spmv.cpp` |
| Custom Krylov kernels | Implementation required | CG/GMRES/BiCGSTAB algorithms | New `src/cuda/solvers/` directory |

### Custom Implementation Architecture

```
src/cuda/solvers/
├── krylov_solver.hpp      # Base class interface
├── cg_solver.cu           # Conjugate Gradient
├── gmres_solver.cu        # Generalized Minimal Residual  
├── bicgstab_solver.cu     # Biconjugate Gradient Stabilized
└── solver_utils.cu        # Common kernels (dot, norm, axpy)
```

**Dependencies:**
- `cuda_runtime.h` — GPU allocation, streams
- `cublas_v2.h` — BLAS-1/2 operations (already available)
- `<cuda/sparse/sparse_ops.hpp>` — SpMV for matrix-vector products (existing)

**No new external dependencies required.**

### cuBLAS Operations Needed (Already Available)

| Operation | cuBLAS Function | Purpose in Krylov |
|-----------|-----------------|-------------------|
| Dot product | `cublasDot` | Residual computation |
| NRM2 | `cublasDnrm2` | Convergence check |
| AXPY | `cublasAxpy` | Vector updates |
| GEMV | `cublasGemv` | Dense matrix operations |
| SpMV | Custom kernel (v2.7) | Sparse iterations |

### CMake Integration

```cmake
# Add solver sources to cuda_impl
set(SOLVER_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/cg_solver.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/gmres_solver.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/bicgstab_solver.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/solver_utils.cu
)
list(APPEND ALL_CUDA_SOURCES ${SOLVER_SOURCES})
```

---

## 2. Roofline Stack Requirements

### What Already Exists (v2.7)

The project already has:
- `BandwidthTracker` in `include/cuda/observability/bandwidth_tracker.h`
- Device memory bandwidth measurement (H2D/D2H/D2D)
- Kernel timing via CUDA events

### What's Needed for Roofline Model

The Roofline model requires:

| Component | Source | Purpose |
|-----------|--------|---------|
| Arithmetic intensity calculation | Implementation required | FLOPs per byte of memory traffic |
| Device memory bandwidth | `BandwidthTracker` (existing) | Memory bound ceiling |
| Peak FLOP/s | Device properties API | Compute bound ceiling |
| FLOP/s counter | Custom instrumentation | Actual achieved performance |

### Implementation Stack

**No external dependencies required.** Roofline model can be implemented using:

1. **CUDA device properties** (`cudaDeviceGetAttribute`) — Peak FLOPS
2. **Memory bandwidth measurement** (existing `BandwidthTracker`)
3. **Kernel FLOP counter** — Custom per-kernel instrumentation
4. **Visualization** — Potentially matplotlib/Python harness from v1.7

### Data Sources

| Metric | Source | Confidence |
|--------|--------|------------|
| Device peak FP32 GFLOP/s | `cudaDeviceGetAttribute(cudaDevAttrSingleKEngineClockRate)` | HIGH |
| Device peak FP64 GFLOP/s | `cudaDeviceGetAttribute(cudaDevAttrDoubleKEngineClockRate)` | HIGH |
| Memory bandwidth (HBM) | `BandwidthTracker::DeviceMemoryBandwidth` | HIGH (v2.7) |
| Achieved FLOP/s | Custom kernel instrumentation | MEDIUM |

### CMake Integration

```cmake
# Add Roofline sources
set(ROOFLINE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/roofline/roofline_model.cu
)
list(APPEND ALL_CUDA_SOURCES ${ROOFLINE_SOURCES})
```

---

## 3. Sparse Format Stack Requirements

### cuSPARSE Format Support

| Format | cuSPARSE Native | Nova Implementation | Notes |
|--------|-----------------|---------------------|-------|
| CSR | `CUSPARSE_FORMAT_CSR` | Existing (v2.1) | Works with SpMV |
| CSC | `CUSPARSE_FORMAT_CSC` | Existing (v2.1) | Works with SpMV |
| ELL | `CUSPARSE_INDEX_RAW` | **Required** | Padded format for regular sparsity |
| HYB | `CUSPARSE_MATRIX_TYPE_GENERAL` + analyze | **Required** | Hybrid ELL/COO format |

### ELL Format (Equal-Length Lazy)

**Purpose:** Efficient for matrices with regular, moderate sparsity (e.g., structured grids). Each row padded to max nnz per row.

**Structure:**
```
values: [a00, a01, a02, pad, b00, b01, b02, pad, ...]  // max_nnz_per_row elements per row
indices: [0, 2, 4, pad, 1, 3, 5, pad, ...]             // column indices, padded with -1 or row length
```

**cuSPARSE support:** 
- `cusparseSpMV` with `CUSPARSE_FORMAT_ELL` via `cusparseCreateCsr` + conversion
- Native ELL format in cuSPARSE 12.x via `cusparseCsr2Ell`, `cusparseEll2Csr`

### HYB Format (Hybrid ELL/COO)

**Purpose:** Handles irregular sparsity in first K columns with ELL, remainder with COO.

**cuSPARSE support:**
- `cusparseCreateHybMat` / `cusparseCscToHyb`
- Automatic conversion: `cusparseCsrToHyb`
- `cusparseSpMV` with HYB handle

### Required Dependencies

| Component | Source | Version | Purpose |
|-----------|--------|---------|---------|
| cuSPARSE | CUDA toolkit | 12.x | ELL/HYB format operations |
| `CUDA::cusparse` | CMake target | — | Link to library |

### CMake Integration

```cmake
# Add cusparse to cuda_impl (verify it's not already linked)
target_link_libraries(cuda_impl PUBLIC
    # ... existing links ...
    CUDA::cusparse  # Add if not already present
)

# Add sparse format sources
set(SPARSE_FORMAT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/sparse/ell_matrix.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/sparse/hyb_matrix.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/sparse/format_conversion.cu
)
list(APPEND ALL_CUDA_SOURCES ${SPARSE_FORMAT_SOURCES})
```

### Header Integration

```cpp
// include/cuda/sparse/sparse_matrix.hpp
#include <cusparse.h>  // Add if not present

enum class SparseFormat { CSR, CSC, ELL, HYB };

template<typename T>
class SparseMatrixELL { /* ... */ };

template<typename T>
class SparseMatrixHYB { /* ... */ };
```

---

## 4. Integration Points

### With Existing cuSOLVER (linalg.h)

The Krylov solvers should integrate with existing linear algebra:

```cpp
// New header: include/cuda/solvers/krylov_solver.hpp
#include "cuda/linalg/linalg.h"
#include "cuda/sparse/sparse_matrix.hpp"

namespace cuda::solvers {

class KrylovSolver {
public:
    virtual ~KrylovSolver() = default;
    virtual bool solve(const float* A, float* x, const float* b, size_t n) = 0;
};

// Factory for creating solvers
std::unique_ptr<KrylovSolver> create_cg_solver();
std::unique_ptr<KrylovSolver> create_gmres_solver(int restart = 50);
std::unique_ptr<KrylovSolver> create_bicgstab_solver();
}  // namespace cuda::solvers
```

### With Existing SpMV (spmv.cpp)

The ELL/HYB formats should share the sparse matrix hierarchy:

```cpp
// include/cuda/sparse/sparse_ops.hpp extensions
namespace nova::sparse {

template<typename T>
void spmv(const SparseMatrixELL<T>& matrix, const T* x, T* y);

template<typename T>
void spmv(const SparseMatrixHYB<T>& matrix, const T* x, T* y);

}  // namespace nova::sparse
```

### With Existing Observability (bandwidth_tracker.h)

Roofline model should use existing bandwidth infrastructure:

```cpp
// include/cuda/roofline/roofline_model.hpp
#include "cuda/observability/bandwidth_tracker.h"

namespace cuda::roofline {

class RooflineModel {
public:
    RooflineModel();
    
    // Add data point from kernel execution
    void add_kernel(const std::string& name, 
                    double flops, 
                    double bytes_transferred,
                    double elapsed_ms);
    
    // Generate Roofline chart data
    std::vector<RooflinePoint> compute_roofline() const;
    
private:
    std::vector<KernelMeasurement> measurements_;
    observability::DeviceMemoryBandwidth bandwidth_;
    double peak_flops_fp32_;
};

}  // namespace cuda::roofline
```

---

## 5. Recommended Dependencies (with versions)

### No New External Dependencies

The v2.8 capabilities can be implemented using existing CUDA toolkit components:

| Component | Status | Justification |
|-----------|--------|---------------|
| cuBLAS | Already linked | BLAS-1/2 for Krylov kernels |
| cuSPARSE | **Add linking** | ELL/HYB format support |
| CUDA events | Already available | Timing for Roofline |
| Device properties | CUDA runtime API | Peak FLOP/s for Roofline |
| Memory bandwidth | v2.7 `BandwidthTracker` | Bandwidth ceiling |

### CMake Changes Required

```cmake
# Verify cusparse is linked in cuda_impl target
# From CMakeLists.txt line 573-589:
target_link_libraries(cuda_impl PUBLIC
    cuda_device
    cuda_algo
    cuda_nccl
    cuda_mpi
    cuda_topology
    cuda_multinode
    cuda_checkpoint
    cuda_comm
    cuda_memory_error
    cuda_preemption
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    CUDA::cufft
    CUDA::cusparse  # ADD THIS LINE
)
```

---

## 6. Anti-Dependencies (What NOT to Add)

| Library | Why Avoid | Alternative |
|---------|-----------|-------------|
| **AmgX** | Separate enterprise product, overkill for basic Krylov | Custom implementation using cuBLAS |
| **MAGMA** | CPU-GPU hybrid, not pure GPU iterative solvers | Custom GPU implementation |
| **PETSc** | CPU-focused, CUDA support is secondary | Custom GPU implementation |
| **ViennaCL** | Unmaintained, incomplete CUDA support | Custom GPU kernels |
| **Eigen** | CPU-focused, CUDA support experimental | cuBLAS already available |
| **Thrust** (new) | Not needed — existing CCCL covers | Keep existing CCCL |
| **cuCollections** | Concurrent data structures, not needed | — |

---

## 7. Summary: Stack Changes for v2.8

### Additions to CMake

```cmake
# 1. Add cusparse if not already linked
find_package(CUDAToolkit REQUIRED)  # Already present
# CUDA::cusparse target available after CUDAToolkit find

# 2. New source directories
set(CUDA_SOLVER_DIR ${CMAKE_SOURCE_DIR}/include/cuda/solvers)
set(CUDA_ROOFLINE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/roofline)
set(CUDA_SPARSE_EXT_DIR ${CMAKE_SOURCE_DIR}/include/cuda/sparse)

# 3. New source files
set(SOLVER_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/cg_solver.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/gmres_solver.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/solvers/bicgstab_solver.cu
)

set(ROOFLINE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/roofline/roofline_model.cu
)

set(SPARSE_EXT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/sparse/ell_matrix.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/sparse/hyb_matrix.cu
)

list(APPEND ALL_CUDA_SOURCES 
    ${SOLVER_SOURCES}
    ${ROOFLINE_SOURCES}
    ${SPARSE_EXT_SOURCES}
)

# 4. Update cuda_impl link libraries
target_link_libraries(cuda_impl PUBLIC CUDA::cusparse)
```

### New Include Directories

| Directory | Purpose |
|-----------|---------|
| `include/cuda/solvers/` | Krylov solver interface |
| `include/cuda/roofline/` | Roofline model implementation |
| `include/cuda/sparse/` (extensions) | ELL/HYB sparse matrix classes |

### No Version Upgrades Needed

| Component | Current | v2.8 | Notes |
|-----------|---------|------|-------|
| CUDA | 20 | 20 | No change needed |
| C++ | 23 | 23 | No change needed |
| CMake | 4.0+ | 4.0+ | No change needed |
| cuBLAS | toolkit | toolkit | Already linked |
| cuSPARSE | toolkit | toolkit | Add link only |
| CCCL | 2.6.0 | 2.6.0 | No change needed |

---

## 8. Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Krylov solver approach | HIGH | CG/GMRES/BiCGSTAB well-documented, implementable with cuBLAS |
| ELL format support | HIGH | cuSPARSE has mature ELL support |
| HYB format support | MEDIUM | HYB is deprecated in cuSPARSE 12.x in favor of CSR+COO merge; verify |
| Roofline model | HIGH | Uses existing infrastructure, no external deps |
| No external deps needed | HIGH | All capabilities available in CUDA toolkit |

---

## Sources

- [cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/) — Direct solvers only, no iterative
- [cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/) — ELL/HYB format support
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) — BLAS operations for Krylov
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) — All libraries included
- [Roofline Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineSideNoVir.pdf) — Berkeley benchmark methodology

---

*Stack research: 2026-05-01 for v2.8 Numerical Computing & Performance*
