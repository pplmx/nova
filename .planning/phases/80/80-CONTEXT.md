# Phase 80: Krylov Solver Core + Roofline - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can solve linear systems Ax=b using CG, GMRES, and BiCGSTAB iterative methods, and analyze kernel performance using the Roofline model to compare achieved vs theoretical performance.

</domain>

<decisions>
## Implementation Decisions

### Krylov Solver API
- Base class `KrylovSolver<T>` with templated solve method
- Derived classes: `ConjugateGradient<T>`, `GMRES<T>`, `BiCGSTAB<T>`
- Configuration via builder pattern or constructor parameters:
  - `relative_tolerance` (default: 1e-6)
  - `max_iterations` (default: 1000)
  - `restart` for GMRES (default: 50)
- Returns `SolverResult<T>` struct with:
  - `converged` flag
  - `iterations` used
  - `residual_norm` final
  - `error_code` if failed

### CG Solver (Symmetric Positive-Definite)
- Requires symmetric matrix (enforce via template or runtime check)
- Work vectors: r, p, Ap (3 vectors of size n)
- Algorithm: classic CG with dot product convergence check

### GMRES Solver (General Arnoldi)
- Flexible GMRES with restart capability
- Work vectors: Krylov basis (m+1 vectors), Hessenberg matrix
- Memory proportional to restart * n
- Full reorthogonalization for stability

### BiCGSTAB Solver (Non-symmetric)
-适合非对称系统，不需要完整Krylov子空间
- Work vectors: r, r_tilde, p, p_hat, s, t (6 vectors)
- Stabilized formulation避免BiCG的波动

### Convergence Criteria
- Relative residual: ||r|| / ||b|| < tolerance
- Absolute residual option: ||r|| < tolerance
- Maximum iterations hard limit
- Early exit on stagnation detection

### Roofline Model Infrastructure
- Device peaks from compute capability lookup table
- Peak FLOP/s: determined by clock rate and FMA throughput
- Peak bandwidth: already available in device_info.h
- Arithmetic intensity: computed from kernel FLOPs / memory traffic
- Roofline plot support: calculate operational intensity vs performance

### Device Peak Computation
Compute theoretical peak for each precision:
- FP64: 2 FMA/clock = 2 ops/clock
- FP32: 2 FMA/clock = 2 ops/clock
- FP16: Tensor cores if available, else 2 ops/clock
- Peak = num_SM * clock_MHz * ops_per_clock * 1e6 / 1e9 GFLOPS

### Performance Measurement
- Warm-up iterations before timing
- Synchronized timing with CUDA events
- Memory bandwidth: use device memory bandwidth formula
- FLOP count: instrumented counters or known kernel FLOPs

### the agent's Discretion
- Solver convergence parameters (tolerances, max iterations)
- Internal algorithm details (preconditioning not in scope)
- Test matrix generation strategies
- Performance measurement methodology

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cuda::performance::get_device_properties()` — existing device queries
- `cuda::performance::get_memory_bandwidth_gbps()` — pre-computed bandwidth
- `SparseMatrixCSR<T>`, `SparseMatrixELL<T>`, `SparseMatrixSELL<T>` — from Phase 79
- `sparse_mv()` function for SpMV operations

### Established Patterns
- Configuration builder pattern in other nova components
- Return type struct with status and data
- Template-based implementations for type flexibility
- Namespace: `nova::sparse::krylov`

### Integration Points
- New files: `include/cuda/sparse/krylov.hpp`, `include/cuda/sparse/roofline.hpp`
- Tests: `tests/sparse/krylov_test.cpp`
- Depends on Phase 79 SpMV operations

</code_context>

<specifics>
## Specific Ideas

No specific requirements — follow standard approaches:
- CG: Standard formulation from Saad's Iterative Methods
- GMRES: Classic Arnoldi with restart
- BiCGSTAB: van der Vorst formulation
- Roofline: Standard compute intensity model

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
