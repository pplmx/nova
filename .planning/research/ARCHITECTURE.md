# Architecture Research: v2.8 Numerical Computing & Performance

**Research Date:** 2026-05-01
**Overall Confidence:** HIGH
**Research Basis:** Existing codebase analysis + standard CUDA numerical computing patterns

---

## Executive Summary

The v2.8 features (Krylov solvers, Roofline model, advanced sparse formats) integrate cleanly with Nova's five-layer architecture through targeted additions to the **Algorithm layer** (Krylov, ELL/HYB SpMV), **Observability layer** (Roofline model), and **Sparse module** (new format classes). All three features share common dependencies: memory buffers, streams, and the existing SpMV primitive.

---

## 1. Krylov Solver Architecture

### 1.1 Integration with Existing Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                                     │
│              krylov_solvers.cpp, iterative_solve.cpp                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Algorithm Layer (NEW)                                 │
│   krylov.h ──► ConjugateGradient, GMRES, BiCGSTAB                          │
│                KrylovSpace (vector management), ConvergenceCriteria        │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Core CUDA Layer (EXISTING)                           │
│   algo/spmv.h ─────────────────────────────────────────────────────────────│
│                     │                      │                                │
│   memory/buffer.h   │  linalg/linalg.h     │  stream/stream.h               │
│                     │  (preconditioners)   │                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 New Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `KrylovContext` | `include/cuda/solvers/krylov_context.h` | Workspace allocation, workspace reuse, iteration state |
| `KrylovSpace<T>` | `include/cuda/solvers/krylov_space.hpp` | Vector management (Ax, r, p, Ap), orthogonality tracking |
| `ConjugateGradient` | `include/cuda/solvers/conjugate_gradient.h` | CG solver for SPD systems |
| `GMRES` | `include/cuda/solvers/gmres.h` | GMRES solver for general systems |
| `BiCGSTAB` | `include/cuda/solvers/bicgstab.h` | BiCGSTAB solver for nonsymmetric systems |
| `ConvergenceCriteria` | `include/cuda/solvers/convergence.h` | Residual, relative residual, divergence stopping |

### 1.3 Data Flow: CG Iteration

```
Initial guess x₀
       │
       ▼
r₀ = b - Ax₀  ──► SpMV via algo/spmv.h
       │
       ▼
p₀ = r₀
       │
       ▼
┌─────────────────────────────────────────┐
│         ITERATION LOOP                  │
│  ┌───────────────────────────────────┐  │
│  │ α = (r·r) / (p·Ap)                │  │
│  │ x = x + α·p                       │  │
│  │ r_new = r - α·Ap                  │  │
│  │ β = (r_new·r_new) / (r·r)         │  │
│  │ p = r_new + β·p                   │  │
│  └───────────────────────────────────┘  │
│       │                                 │
│       ▼                                 │
│  convergence_check()                    │
│       │                                 │
│       └───► max_iterations? ──► EXIT   │
└─────────────────────────────────────────┘
       │
       ▼
   Solution x
```

### 1.4 Key Interfaces

```cpp
// include/cuda/solvers/krylov.h
namespace cuda::solvers {

template<typename T>
struct KrylovResult {
    memory::Buffer<T> x;
    int iterations;
    T residual;
    bool converged;
};

template<typename T>
struct KrylovConfig {
    int max_iterations = 1000;
    T tolerance = T{1e-6};
    bool verbose = false;
};

// Main entry point - dispatches to appropriate solver
template<typename T>
KrylovResult<T> solve(const SparseMatrix& A, const T* b, 
                      SolverType type, const KrylovConfig<T>& config);

// Direct solver interfaces
template<typename T>
KrylovResult<T> cg(const SparseMatrix& A, const T* b, 
                   const KrylovConfig<T>& config = {});

template<typename T>                   
KrylovResult<T> gmres(const SparseMatrix& A, const T* b,
                      int restart = 50,
                      const KrylovConfig<T>& config = {});

template<typename T>
KrylovResult<T> bicgstab(const SparseMatrix& A, const T* b,
                         const KrylovConfig<T>& config = {});

}  // namespace cuda::solvers
```

### 1.5 Dependencies

| Dependency | Source | Usage |
|------------|--------|-------|
| SpMV | `algo/spmv.h`, `sparse/sparse_ops.hpp` | Matrix-vector products |
| Memory | `memory/buffer.h` | Workspace allocation |
| Streams | `stream/stream.h` | Async execution |
| DOT product | `algo/reduce.h` | Inner products |
| NVTX | `observability/nvtx_extensions.h` | Profiling |

---

## 2. Roofline Model Architecture

### 2.1 Integration with Existing Observability Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Production Layer                                     │
│              profiler.h, health_metrics.h, autotuner.h                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Observability Layer (EXTEND)                         │
│  kernel_stats.h ──► roofline.h ──► RooflineModel                          │
│  bandwidth_tracker.h ──────────────────────────────────────────────────────│
│                     │                                                        │
│  nvtx_extensions.h  │  device_info.h (peak FLOP/s, memory bandwidth)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 New Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `RooflineModel` | `include/cuda/observability/roofline.h` | Peak computation/bandwidth, operational intensity |
| `RooflinePoint` | `include/cuda/observability/roofline.h` | Single kernel measurement point |
| `RooflineReport` | `include/cuda/observability/roofline.h` | Aggregate analysis, bottlenecks |

### 2.3 Roofline Computation

```
Operational Intensity (I) = FLOPs / Bytes Accessed

Theoretical Peak Performance:
  Compute-bound:  P_peak = min(Peak_FLOPS, Peak_Bandwidth × I)
  
Achieved Performance:
  P_actual = Measured_FLOPs / Measured_Time

AI: Arithmetic Intensity (FLOPs/Byte)
    I > Peak_FLOPS / Peak_Bandwidth → Compute-bound
    I < Peak_FLOPS / Peak_Bandwidth → Memory-bound

     GFLOP/s
        ▲
        │        /
        │       /  Peak FLOP/s
        │      /
        │     /••••••••• Roofline
        │    / •
        │   /  • <-- Kernel points
        │  /    •
        │ /      •••••••••••••••••••••••• Peak Bandwidth × I
        └────────────────────────────────────────►
              0    I_peak    Operational Intensity (FLOP/Byte)
```

### 2.4 Key Interfaces

```cpp
// include/cuda/observability/roofline.h
namespace cuda::observability {

struct RooflinePoint {
    std::string kernel_name;
    double flops;                    // Total FLOPs in kernel
    double bytes_accessed;           // Memory bytes accessed
    double time_us;                  // Kernel execution time
    double operational_intensity;    // Computed: flops / bytes
    double achieved_gflops;          // Computed: flops / time
    double peak_gflops;              // Device peak FLOP/s
    double peak_bandwidth_gbps;      // Device memory bandwidth
    bool is_compute_bound;           // True if compute-bound
};

class RooflineModel {
public:
    RooflineModel(int device_id = 0);
    
    // Record a kernel execution
    void record_kernel(const char* name, double flops, 
                       double bytes, double time_us);
    
    // Record from existing KernelStats
    void record_from_stats(const KernelStats& stats, 
                           double flops, double bytes);
    
    // Compute and return analysis
    std::vector<RooflinePoint> analyze() const;
    
    // Export for visualization
    void export_json(const std::string& path) const;
    void export_csv(const std::string& path) const;

private:
    std::vector<RooflinePoint> points_;
    double peak_flops_gf_;
    double peak_bandwidth_gbps_;
};

}  // namespace cuda::observability
```

### 2.5 Integration Points

| Existing Component | Integration Method |
|--------------------|-------------------|
| `KernelStatsCollector` | Extend to track FLOPs/bytes per kernel |
| `BandwidthTracker` | Query peak bandwidth for roofline normalization |
| `DeviceProperties` | Query peak FLOP/s (tensor/FP64/FP32) |
| `nvtx_extensions.h` | Add "nova.roofline" NVTX domain |

---

## 3. Sparse Format Architecture (ELL/HYB)

### 3.1 Integration with Existing Sparse Module

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Algorithm Layer                                      │
│              algo/spmv.h (existing CSR/CSC SpMV)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Sparse Module (EXTEND)                               │
│  sparse_matrix.hpp ──► SparseMatrixELL, SparseMatrixHYB                    │
│  sparse_ops.hpp ──► ELL/HYB SpMV kernels                                   │
│  format_converter.hpp ──► CSR ↔ ELL ↔ HYB conversion                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Format Structures

**ELL (Ellpack)** - Fixed-width per row:
```
Values:  [a00, a01, a02, ..., a0k-1, a10, a11, ..., a1k-1, ...]
Indices: [c00, c01, c02, ..., c0k-1, c10, c11, ..., c1k-1, ...]
        where k = max_row_nnz (padded to this)
```

**HYB (Hybrid ELL + COO)** - ELL for regular part, COO for irregular tail:
```
ELL part:  Same as ELL for (nnz - coo_nnz) nonzeros
COO part:  Remaining irregular entries stored as (row, col, val)
```

### 3.3 New Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `SparseMatrixELL<T>` | `include/cuda/sparse/sparse_matrix.hpp` | ELL format storage and accessors |
| `SparseMatrixHYB<T>` | `include/cuda/sparse/sparse_matrix.hpp` | Hybrid ELL+COO format |
| `FormatConverter` | `include/cuda/sparse/format_converter.hpp` | CSR ↔ ELL ↔ HYB conversion |
| `FormatAnalyzer` | `include/cuda/sparse/format_analyzer.hpp` | Pattern analysis for format selection |
| `ell_spmv()` | `include/cuda/sparse/sparse_ops.hpp` | ELL sparse-matrix-vector product |
| `hyb_spmv()` | `include/cuda/sparse/sparse_ops.hpp` | HYB sparse-matrix-vector product |

### 3.4 Key Interfaces

```cpp
// Extended sparse_matrix.hpp
namespace nova::sparse {

enum class SparseFormat { CSR, CSC, ELL, HYB };

template<typename T>
class SparseMatrixELL {
public:
    SparseMatrixELL() = default;
    
    SparseMatrixELL(std::vector<T> values, std::vector<int> col_indices,
                    int num_rows, int num_cols, int padded_nnz_per_row);
    
    // Conversion from CSR
    static std::optional<SparseMatrixELL<T>> FromCSR(const SparseMatrixCSR<T>& csr);
    
    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int padded_nnz_per_row() const { return padded_nnz_per_row_; }
    
    const T* values() const { return values_.data(); }
    const int* col_indices() const { return col_indices_.data(); }

private:
    std::vector<T> values_;
    std::vector<int> col_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
    int padded_nnz_per_row_ = 0;
};

template<typename T>
class SparseMatrixHYB {
public:
    SparseMatrixHYB() = default;
    
    SparseMatrixHYB(SparseMatrixELL<T> ell, std::vector<int> coo_rows,
                    std::vector<int> coo_cols, std::vector<T> coo_vals,
                    int num_rows, int num_cols);
    
    static std::optional<SparseMatrixHYB<T>> FromCSR(const SparseMatrixCSR<T>& csr,
                                                      int ell_threshold);
    
    int num_rows() const { return num_rows_; }
    int num_cols() const { return num_cols_; }
    int coo_nnz() const { return static_cast<int>(coo_vals_.size()); }
    
    // Accessors for ELL and COO parts...

private:
    SparseMatrixELL<T> ell_part_;
    std::vector<int> coo_rows_;
    std::vector<int> coo_cols_;
    std::vector<T> coo_vals_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};

// format_converter.hpp
class FormatConverter {
public:
    template<typename T>
    static SparseMatrixELL<T> csr_to_ell(const SparseMatrixCSR<T>& csr);
    
    template<typename T>
    static SparseMatrixHYB<T> csr_to_hyb(const SparseMatrixCSR<T>& csr,
                                         int ell_threshold = 16);
    
    template<typename T>
    static SparseMatrixCSR<T> ell_to_csr(const SparseMatrixELL<T>& ell);
    
    template<typename T>
    static SparseMatrixCSR<T> hyb_to_csr(const SparseMatrixHYB<T>& hyb);
};

// format_analyzer.hpp
class FormatAnalyzer {
public:
    struct PatternAnalysis {
        double max_nnz_per_row;
        double avg_nnz_per_row;
        double stddev_nnz;
        double variance_coefficient;  // stddev / avg (for format selection)
        bool is_regular;              // variance_coefficient < threshold
    };
    
    static PatternAnalysis analyze(const SparseMatrixCSR<T>& csr);
    
    // Recommend best format based on pattern
    static SparseFormat recommend_format(const SparseMatrixCSR<T>& csr,
                                         int ell_threshold = 16);
};

}  // namespace nova::sparse
```

### 3.5 SpMV Integration

The sparse_ops.hpp already has a `SparseOps<T>::spmv()` pattern. Extend with:

```cpp
// In sparse_ops.hpp
template<typename T>
class SparseOps {
public:
    static void spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);
    static void spmv(const SparseMatrixELL<T>& matrix, const T* x, T* y);
    static void spmv(const SparseMatrixHYB<T>& matrix, const T* x, T* y);
    // ...
};
```

---

## 4. Integration with Existing Layers

### 4.1 Five-Layer Integration Summary

| Layer | Integration | New Files |
|-------|-------------|-----------|
| **Memory** | `Buffer<T>` for Krylov workspace, ELL/HYB storage | None (reuse) |
| **Device** | Existing error handling (`cuda/error.h`) | None (reuse) |
| **Algorithm** | SpMV for Krylov, ELL/HYB SpMV kernels | `solvers/krylov.h`, `sparse/sparse_ops.hpp` |
| **API** | Unified solver interface, format selection | `sparse/format_analyzer.hpp` |
| **Distributed** | (Future) Block iterative solvers with NCCL | None yet |

### 4.2 NVTX Domain Extension

Add to `nvtx_extensions.h`:

```cpp
struct NVTXDomains {
    // ... existing domains ...
    static constexpr nvtx3::domain_handle_t Solvers = nvtx3::domain_create("nova.solvers");
    static constexpr nvtx3::domain_handle_t Sparse = nvtx3::domain_create("nova.sparse");
    static constexpr nvtx3::domain_handle_t Roofline = nvtx3::domain_create("nova.roofline");
};
```

### 4.3 Observability Integration

```
kernel_stats.h ─────────────────────────────────────────────┐
                                                          │
bandwidth_tracker.h ──────────────────────────────┐        │
                                                   │        │
roofline.h (NEW) ─────────────────────────────────┴────────┤
                                                   ▲        │
nvtx_extensions.h ────────────────────────────────┘        │
                                                          │
           ┌───────────────────────────────────────────────┘
           │
           ▼
   RooflineModel::analyze()
           │
           ▼
   JSON/CSV export for visualization
```

---

## 5. Suggested Build Order

The following order respects dependencies and enables parallel development of independent components:

### Phase 1: Sparse Format Foundation (Lowest Dependency)
**Rationale:** ELL/HYB classes have no dependencies on solvers or roofline.

1. `sparse_matrix.hpp` - Add `SparseMatrixELL<T>`, `SparseMatrixHYB<T>`
2. `format_converter.hpp` - CSR ↔ ELL ↔ HYB conversion (NEW FILE)
3. `sparse_ops.hpp` - Add ELL/HYB SpMV functions
4. `format_analyzer.hpp` - Pattern analysis for format selection (NEW FILE)
5. Tests for sparse formats

### Phase 2: Krylov Solver Core
**Rationale:** Depends on SpMV from Phase 1, but independent of Roofline.

1. `krylov_context.h` - Workspace management (NEW FILE)
2. `krylov_space.hpp` - Vector space operations (NEW FILE)
3. `convergence.h` - Convergence criteria (NEW FILE)
4. `conjugate_gradient.h` - CG implementation (NEW FILE)
5. `gmres.h` - GMRES implementation (NEW FILE)
6. `bicgstab.h` - BiCGSTAB implementation (NEW FILE)
7. `krylov.h` - Unified entry point (NEW FILE)
8. Tests for Krylov solvers

### Phase 3: Roofline Model
**Rationale:** Independent of both Phase 1 and 2, but integrates with existing observability.

1. `observability/roofline.h` - Roofline model (NEW FILE)
2. Extend `kernel_stats.h` - Track FLOPs/bytes per kernel
3. Extend `nvtx_extensions.h` - Add roofline domain
4. Integrate with `profiler.h` - Auto-generate roofline from profile runs
5. Tests and visualization

### Phase 4: Integration
**Rationale:** Combines all components with production concerns.

1. Update `algo_wrapper.h` - Wrap Krylov solvers with error handling
2. Update `profiler.h` - Auto-run roofline analysis post-profile
3. Update `autotuner.h` - Use roofline to select optimal kernel configs
4. Update `test_integration.h` - E2E tests for full pipeline
5. Benchmark suite updates

---

## 6. New Files Required

### Solver Module
| File | Purpose |
|------|---------|
| `include/cuda/solvers/krylov.h` | Unified solver interface |
| `include/cuda/solvers/krylov_context.h` | Workspace and state management |
| `include/cuda/solvers/krylov_space.hpp` | Vector space template |
| `include/cuda/solvers/convergence.h` | Convergence criteria |
| `include/cuda/solvers/conjugate_gradient.h` | CG implementation |
| `include/cuda/solvers/gmres.h` | GMRES implementation |
| `include/cuda/solvers/bicgstab.h` | BiCGSTAB implementation |
| `tests/cuda/solvers/*.cpp` | Solver tests |

### Sparse Format Extension
| File | Purpose |
|------|---------|
| `include/cuda/sparse/format_converter.hpp` | Format conversion utilities |
| `include/cuda/sparse/format_analyzer.hpp` | Pattern analysis, format selection |
| `tests/cuda/sparse/format_tests.cpp` | Format conversion tests |

### Observability Extension
| File | Purpose |
|------|---------|
| `include/cuda/observability/roofline.h` | Roofline model implementation |
| `tests/cuda/observability/roofline_tests.cpp` | Roofline tests |

---

## 7. Compatibility Notes

### Backward Compatibility
- All existing sparse APIs remain unchanged
- New format classes are additive
- NVTX domains are additive (compile-time toggle preserved)

### API Stability
- `krylov_solve()` returns `KrylovResult<T>` - new struct, no breaking change
- `SparseFormat` enum extended with ELL/HYB values
- Existing `SparseMatrixCSR<T>` and `SparseMatrixCSC<T>` unchanged

### Performance Considerations
- ELL/HYB formats can reduce SpMV kernel launch overhead for structured matrices
- GMRES uses preallocated workspace (avoid per-iteration malloc)
- Roofline analysis is post-hoc (no runtime overhead)

---

## 8. Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Krylov Architecture | HIGH | Standard numerical computing patterns, well-understood algorithms |
| ELL/HYB Integration | HIGH | Straight extension of existing CSR/CSC patterns |
| Roofline Model | HIGH | Existing observability infrastructure, standard Roofline model |
| Build Order | MEDIUM | Dependency analysis, but parallel workstreams may enable reordering |
| API Design | MEDIUM | Follows existing Nova patterns, but may need adjustment based on user feedback |

---

*Architecture research complete: 2026-05-01*
