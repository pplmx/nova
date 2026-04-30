# Milestone v2.8 Requirements

**Project:** Nova CUDA Library Enhancement
**Milestone:** v2.8 Numerical Computing & Performance
**Date:** 2026-05-01
**Total Requirements:** 17

## Requirements by Phase

### Phase 79: Sparse Format Foundation

- [ ] **SPARSE-01**: User can store sparse matrices in ELL (ELLPACK) format with row-wise padding
- [ ] **SPARSE-02**: User can store sparse matrices in SELL (Sliced ELLPACK) format with slice organization
- [ ] **SPARSE-03**: User can convert CSR matrices to ELL/SELL format with automatic format selection
- [ ] **SPARSE-04**: User can perform SpMV operations using ELL and SELL formats

### Phase 80: Krylov Solver Core + Roofline

- [ ] **KRY-01**: User can solve symmetric positive-definite linear systems using Conjugate Gradient (CG) method
- [ ] **KRY-02**: User can solve general non-symmetric linear systems using Generalized Minimal Residual (GMRES) method
- [ ] **KRY-03**: User can solve non-symmetric linear systems using Biconjugate Gradient Stabilized (BiCGSTAB) method
- [ ] **KRY-04**: User can configure convergence criteria (relative residual tolerance, max iterations)
- [ ] **RF-01**: User can query device peak FLOP/s for FP64, FP32, and FP16 precision
- [ ] **RF-02**: User can measure achieved memory bandwidth and compare against device peak
- [ ] **RF-03**: User can calculate arithmetic intensity (FLOPs / Bytes accessed) for kernel operations

### Phase 81: Extended Formats + Roofline Analysis

- [ ] **SPARSE-05**: User can store sparse matrices in HYB (Hybrid ELL+COO) format with automatic partition
- [ ] **RF-04**: User can classify performance limiters (compute-bound vs memory-bound) using Roofline model
- [ ] **RF-05**: User can export Roofline analysis data in JSON format for external visualization

### Phase 82: Integration & Production

- [ ] **KRY-05**: User can reuse solver workspace across multiple solves via memory pool integration
- [ ] **KRY-06**: User can access solver diagnostic information (iteration count, residual history, convergence status)
- [ ] **INT-01**: User can run end-to-end tests validating all three feature categories
- [ ] **INT-02**: User can run benchmarks comparing sparse format performance and solver convergence
- [ ] **INT-03**: User can profile solver and format operations via NVTX annotations
- [ ] **INT-04**: User can access updated documentation covering Krylov solvers, Roofline model, and sparse formats

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SPARSE-01 | Phase 79 | Pending |
| SPARSE-02 | Phase 79 | Pending |
| SPARSE-03 | Phase 79 | Pending |
| SPARSE-04 | Phase 79 | Pending |
| KRY-01 | Phase 80 | Pending |
| KRY-02 | Phase 80 | Pending |
| KRY-03 | Phase 80 | Pending |
| KRY-04 | Phase 80 | Pending |
| RF-01 | Phase 80 | Pending |
| RF-02 | Phase 80 | Pending |
| RF-03 | Phase 80 | Pending |
| SPARSE-05 | Phase 81 | Pending |
| RF-04 | Phase 81 | Pending |
| RF-05 | Phase 81 | Pending |
| KRY-05 | Phase 82 | Pending |
| KRY-06 | Phase 82 | Pending |
| INT-01 | Phase 82 | Pending |
| INT-02 | Phase 82 | Pending |
| INT-03 | Phase 82 | Pending |
| INT-04 | Phase 82 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

## Out of Scope

| Feature | Reason |
|---------|--------|
| Direct sparse solvers (LU/Cholesky) | Use cuDSS for direct solves; separate module |
| Full preconditioner library | Scope creep; just Jacobi as starting point |
| Matrix-free solvers | Premature abstraction for v2.8 |
| DIA/JDS sparse formats | Rarely optimal; defer if real need emerges |
| Real-time Roofline monitoring | Batch analysis mode sufficient |
| Multi-GPU distributed solvers | Future milestone |

## Future Requirements

### Deferred from v2.8

- ILU preconditioner for Krylov solvers
- cuDSS wrapper for direct sparse solves
- Multi-GPU distributed Krylov
- BSR (Block Sparse Row) format support

---
*Requirements defined: 2026-05-01*
*Research completed: 2026-05-01*
*Ready for planning: yes*
