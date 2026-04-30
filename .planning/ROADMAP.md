# Milestone v2.8 Roadmap

**Project:** Nova CUDA Library Enhancement
**Milestone:** v2.8 Numerical Computing & Performance
**Granularity:** Standard
**Coverage:** 20/20 requirements mapped (100%)

---

## Phases

- [ ] **Phase 79: Sparse Format Foundation** - ELL/SELL storage, CSR conversion, sparse matrix-vector multiplication
- [ ] **Phase 80: Krylov Solver Core + Roofline** - CG/GMRES/BiCGSTAB solvers, device peak metrics, arithmetic intensity
- [ ] **Phase 81: Extended Formats + Roofline Analysis** - HYB format, performance classification, JSON export
- [ ] **Phase 82: Integration & Production** - Memory pool integration, diagnostics, E2E tests, benchmarks, NVTX, docs

---

## Phase Details

### Phase 79: Sparse Format Foundation

**Goal:** Users can store sparse matrices in ELL/SELL formats, convert from CSR, and perform SpMV operations

**Depends on:** Phase 78 (existing CSR/CSC SpMV from Phase 76)

**Requirements:** SPARSE-01, SPARSE-02, SPARSE-03, SPARSE-04

**Success Criteria** (what must be TRUE):

1. User can store sparse matrices in ELL (ELLPACK) format with row-wise padding to max_nnz_per_row
2. User can store sparse matrices in SELL (Sliced ELLPACK) format with configurable slice height
3. User can convert existing CSR matrices to ELL or SELL format with automatic padding calculation
4. User can perform SpMV operations using ELL and SELL formatted matrices and compare results against CSR

**Plans:** 4 plans in 3 waves

**Plan list:**
- [ ] 79-00-PLAN.md — Interface contracts for ELL/SELL classes
- [ ] 79-01-PLAN.md — Implement ELL and SELL matrix classes with FromCSR factory
- [ ] 79-02-PLAN.md — Implement ELL and SELL SpMV operations
- [ ] 79-03-PLAN.md — Comprehensive tests for ELL/SELL formats

---

### Phase 80: Krylov Solver Core + Roofline

**Goal:** Users can solve linear systems with CG/GMRES/BiCGSTAB and analyze kernel performance using Roofline model

**Depends on:** Phase 79 (SpMV operations required for Krylov iteration)

**Requirements:** KRY-01, KRY-02, KRY-03, KRY-04, RF-01, RF-02, RF-03

**Success Criteria** (what must be TRUE):

1. User can solve symmetric positive-definite linear systems using Conjugate Gradient (CG) method
2. User can solve general non-symmetric linear systems using Generalized Minimal Residual (GMRES) method
3. User can solve non-symmetric linear systems using Biconjugate Gradient Stabilized (BiCGSTAB) method
4. User can configure convergence criteria including relative residual tolerance and maximum iterations
5. User can query device peak FLOP/s for FP64, FP32, and FP16 precision from device properties
6. User can measure achieved memory bandwidth and compare against device theoretical peak
7. User can calculate arithmetic intensity (FLOPs per byte accessed) for any kernel operation

**Plans:** 1 plan in 1 wave

**Plan list:**
- [ ] 80-01-PLAN.md — Krylov solvers (CG/GMRES/BiCGSTAB) + Roofline infrastructure

---

### Phase 81: Extended Formats + Roofline Analysis

**Goal:** Users can use HYB format for irregular sparse matrices and classify kernel performance using Roofline model

**Depends on:** Phase 79, Phase 80

**Requirements:** SPARSE-05, RF-04, RF-05

**Success Criteria** (what must be TRUE):

1. User can store sparse matrices in HYB (Hybrid ELL+COO) format with automatic partition based on row density
2. User can classify performance limiters (compute-bound vs memory-bound) using Roofline model analysis
3. User can export Roofline analysis data in JSON format for external visualization tools

**Plans:** TBD

---

### Phase 82: Integration & Production

**Goal:** Users can reuse solver workspace, access diagnostics, run comprehensive tests and benchmarks, and access documentation

**Depends on:** Phase 79, Phase 80, Phase 81

**Requirements:** KRY-05, KRY-06, INT-01, INT-02, INT-03, INT-04

**Success Criteria** (what must be TRUE):

1. User can reuse solver workspace across multiple consecutive solves via memory pool integration
2. User can access solver diagnostic information including iteration count, residual history, and convergence status
3. User can run end-to-end tests validating sparse formats, Krylov solvers, and Roofline analysis together
4. User can run benchmarks comparing sparse format performance and solver convergence rates
5. User can profile solver and format operations via NVTX annotations with dedicated domains
6. User can access updated documentation covering Krylov solvers, Roofline model, and sparse formats

**Plans:** TBD

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 79. Sparse Format Foundation | 0/4 | Planned | - |
| 80. Krylov Solver Core + Roofline | 0/1 | Planned | - |
| 81. Extended Formats + Roofline Analysis | 0/1 | Not started | - |
| 82. Integration & Production | 0/1 | Not started | - |

---

## Coverage Map

```
SPARSE-01 → Phase 79 (ELL format storage)
SPARSE-02 → Phase 79 (SELL format storage)
SPARSE-03 → Phase 79 (CSR→ELL/SELL conversion)
SPARSE-04 → Phase 79 (ELL/SELL SpMV kernels)
KRY-01 → Phase 80 (CG solver)
KRY-02 → Phase 80 (GMRES solver)
KRY-03 → Phase 80 (BiCGSTAB solver)
KRY-04 → Phase 80 (Convergence criteria)
RF-01 → Phase 80 (Device peak FLOP/s)
RF-02 → Phase 80 (Memory bandwidth comparison)
RF-03 → Phase 80 (Arithmetic intensity calculation)
SPARSE-05 → Phase 81 (HYB format)
RF-04 → Phase 81 (Performance limiter classification)
RF-05 → Phase 81 (JSON export)
KRY-05 → Phase 82 (Memory pool integration)
KRY-06 → Phase 82 (Solver diagnostics)
INT-01 → Phase 82 (E2E tests)
INT-02 → Phase 82 (Performance benchmarks)
INT-03 → Phase 82 (NVTX integration)
INT-04 → Phase 82 (Documentation updates)

Mapped: 20/20 ✓
No orphaned requirements ✓
```

---

*Roadmap created: 2026-05-01*
*v2.8 phases: 79, 80, 81, 82*
