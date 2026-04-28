# Roadmap: Nova v2.3 Extended Algorithms

## Milestones

- ✅ **v2.2 Comprehensive Enhancement** - Phases 1-53 (shipped 2026-04-27)
- 🚧 **v2.3 Extended Algorithms** - Phases 54-58 (in progress)
- 📋 **v2.4 Production Hardening** - Phases 59+ (planned)

## Phase Summary

- [x] **Phase 54: Foundation & Sorting** - GPU radix sort, top-k selection, binary search ✅
- [x] **Phase 55: Linear Algebra Extras** - SVD, eigenvalue decomposition, matrix factorization ✅
- [x] **Phase 56: Numerical Methods** - Monte Carlo, integration, root finding, interpolation ✅
- [x] **Phase 57: Signal Processing** - FFT convolution, wavelet transform, FIR filters ✅
- [x] **Phase 58: Integration & Polish** - Performance, benchmarks, documentation ✅

## Phase Details

### Phase 54: Foundation & Sorting
**Goal**: Users can efficiently sort key-value pairs and search sorted data using GPU primitives
**Depends on**: Nothing (first phase of milestone)
**Requirements**: SORT-01, SORT-02, SORT-03
**Success Criteria** (what must be TRUE):
  1. User can sort arrays of key-value pairs in ascending or descending order using GPU radix sort
  2. User can find the k largest elements in a dataset without performing a full sort
  3. User can perform binary search on sorted arrays using warp shuffle primitives
  4. Sorting operations integrate cleanly with existing Buffer and MemoryPool patterns
**Plans**: TBD

### Phase 55: Linear Algebra Extras
**Goal**: Users can compute advanced matrix decompositions for scientific computing applications
**Depends on**: Phase 54
**Requirements**: LINALG-01, LINALG-02, LINALG-03
**Success Criteria** (what must be TRUE):
  1. User can compute SVD with standard and randomized modes for matrix decomposition
  2. User can compute eigenvalues and eigenvectors of symmetric matrices
  3. User can compute QR, Cholesky, and LDL matrix factorizations
  4. All decompositions provide condition number estimation for numerical stability feedback
**Plans**: TBD

### Phase 56: Numerical Methods
**Goal**: Users can run numerical computations for scientific simulations and data analysis
**Depends on**: Phase 55
**Requirements**: NUM-01, NUM-02, NUM-03, NUM-04
**Success Criteria** (what must be TRUE):
  1. User can run Monte Carlo simulations with variance reduction via antithetic variates
  2. User can compute numerical integrals using trapezoidal and Simpson rules
  3. User can find roots of functions using bisection and Newton-Raphson methods
  4. User can perform interpolation using linear and cubic spline methods
**Plans**: TBD
**UI hint**: yes

### Phase 57: Signal Processing
**Goal**: Users can process signals using FFT-based convolution, wavelet transforms, and FIR filters
**Depends on**: Phase 56
**Requirements**: SIGNAL-01, SIGNAL-02, SIGNAL-03
**Success Criteria** (what must be TRUE):
  1. User can compute FFT-based convolution for efficient large-kernel filtering
  2. User can compute Haar wavelet transform with forward and inverse operations
  3. User can apply FIR filters to signals with configurable coefficients
  4. Signal processing operations handle boundary conditions gracefully
**Plans**: TBD

### Phase 58: Integration & Polish
**Goal**: All new algorithms are tested, benchmarked, and documented for production use
**Depends on**: Phase 57
**Requirements**: (Cross-cutting)
**Success Criteria** (what must be TRUE):
  1. All new algorithms pass unit tests with >80% coverage
  2. Performance benchmarks exist for each algorithm domain
  3. API documentation covers all new functions with usage examples
  4. CMake integration properly links CUDA library dependencies (cuSOLVER, cuRAND)
**Plans**: TBD

## Progress

| Phase | Goal | Requirements | Success Criteria | Status |
|-------|------|--------------|------------------|--------|
| 54. Foundation & Sorting | GPU sorting primitives | SORT-01, SORT-02, SORT-03 | 4 criteria | ✅ Complete |
| 55. Linear Algebra Extras | Matrix decompositions | LINALG-01, LINALG-02, LINALG-03 | 4 criteria | ✅ Complete |
| 56. Numerical Methods | Scientific computing | NUM-01, NUM-02, NUM-03, NUM-04 | 4 criteria | ✅ Complete |
| 57. Signal Processing | Signal transforms | SIGNAL-01, SIGNAL-02, SIGNAL-03 | 4 criteria | ✅ Complete |
| 58. Integration & Polish | Production readiness | (cross-cutting) | 4 criteria | ✅ Complete |

**Coverage:** 13/13 requirements mapped ✓

---

*Roadmap created: 2026-04-28 for v2.3 Extended Algorithms*
