---
phase: 80
plan: 01
type: summary
wave: 1
status: complete
files_modified:
  - include/cuda/sparse/krylov.hpp
  - include/cuda/sparse/roofline.hpp
  - tests/sparse/krylov_test.cpp
---

# Phase 80, Plan 01: Krylov Solvers + Roofline

## Status: Complete

### Files Created

**include/cuda/sparse/krylov.hpp:**
- `SolverResult<T>` struct with convergence status, iterations, residual
- `SolverConfig<T>` struct with tolerance, max iterations settings
- `KrylovSolver<T>` base class (abstract)
- `ConjugateGradient<T>` for SPD systems
- `GMRES<T>` with restart for non-symmetric systems
- `BiCGSTAB<T>` for non-symmetric systems using van der Vorst formulation

**include/cuda/sparse/roofline.hpp:**
- `DevicePeaks` struct with FP64/FP32/FP16 peaks and bandwidth
- `get_device_peaks()` function computing theoretical peaks
- `RooflineMetrics` struct for performance analysis
- `RooflineAnalyzer` class with classification logic
- `arithmetic_intensity()` template for FLOPs/byte calculation
- `spmv_arithmetic_intensity()` helper for sparse matrix operations

**tests/sparse/krylov_test.cpp:**
- CG solver tests (trivial, Laplacian, convergence rate)
- GMRES tests (non-symmetric, restart behavior)
- BiCGSTAB tests (non-symmetric, convergence)
- Roofline tests (device peaks, AI calculation, classification)

### Key Implementation Details

**CG Algorithm:**
- Work vectors: r, p, Ap (3n storage)
- Convergence: ||r||/||b|| < tolerance
- Tested on 10x10 tridiagonal matrix → converged in 5 iterations

**GMRES Algorithm:**
- Configurable restart (default 50)
- Full reorthogonalization for stability
- Hessenberg matrix for least squares

**BiCGSTAB Algorithm:**
- Work vectors: r, r_tilde, p, p_hat, s, t (6n storage)
- Stabilized formulation avoids BiCG oscillations

### Verification

- CG on 10x10 Laplacian: Converged in 5 iterations, residual 2.29e-18
- SpMV arithmetic intensity: 0.124 FLOPs/byte (memory-bound)

---
*Summary generated: 2026-05-01*
