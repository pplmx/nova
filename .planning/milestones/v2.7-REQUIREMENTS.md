# Milestone v2.7 Requirements

**Project:** Nova CUDA Library Enhancement
**Milestone:** v2.7 Comprehensive Testing & Validation
**Date:** 2026-04-30
**Total Requirements:** 16

## Requirements by Phase

### Phase 75: Observability & Profiling

- [ ] **OBS-01**: User can export timeline visualizations in Chrome trace format from NVTX annotations
- [ ] **OBS-02**: User can measure memory bandwidth (H2D/D2H/D2D) via NVbandwidth integration
- [ ] **OBS-03**: User can collect kernel statistics (latency, throughput, occupancy) per kernel launch
- [ ] **OBS-04**: User can analyze real-time occupancy and receive feedback on block size selection

### Phase 76: Algorithm Extensions

- [ ] **ALGO-01**: User can perform segmented sort to sort elements within groups without full array copy
- [ ] **ALGO-02**: User can compute sparse matrix-vector multiply using CSR/CSC formats from v2.1
- [ ] **ALGO-03**: User can sort large datasets using sample sort when radix sort is inefficient
- [ ] **ALGO-04**: User can compute single-source shortest paths using delta-stepping algorithm

### Phase 77: Robustness & Testing

- [ ] **ROB-01**: User can validate memory safety using Compute Sanitizer (UAF, double-free, uninitialized memory)
- [ ] **ROB-02**: User can run tests in isolated CUDA contexts without state pollution between tests
- [ ] **ROB-03**: User can inject errors at specific layer boundaries (Memory, Device, Algorithm, Stream, Inference)
- [ ] **ROB-04**: User can test CUDA-specific boundary conditions (256-byte alignment, warp size, SM limits)
- [ ] **ROB-05**: User can control FP determinism levels (not_guaranteed, run_to_run, gpu_to_gpu)

### Phase 78: Integration & Validation

- [ ] **INT-01**: User can run end-to-end robustness tests with simultaneous profiling enabled
- [ ] **INT-02**: User can validate memory safety across all algorithm implementations
- [ ] **INT-03**: User can establish performance regression baselines for comparison
- [ ] **INT-04**: User can access updated documentation for new observability and algorithm features

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| OBS-01 | Phase 75 | Pending |
| OBS-02 | Phase 75 | Pending |
| OBS-03 | Phase 75 | Pending |
| OBS-04 | Phase 75 | Pending |
| ALGO-01 | Phase 76 | Pending |
| ALGO-02 | Phase 76 | Pending |
| ALGO-03 | Phase 76 | Pending |
| ALGO-04 | Phase 76 | Pending |
| ROB-01 | Phase 77 | Pending |
| ROB-02 | Phase 77 | Pending |
| ROB-03 | Phase 77 | Pending |
| ROB-04 | Phase 77 | Pending |
| ROB-05 | Phase 77 | Pending |
| INT-01 | Phase 78 | Pending |
| INT-02 | Phase 78 | Pending |
| INT-03 | Phase 78 | Pending |
| INT-04 | Phase 78 | Pending |

## Out of Scope

- Chaos engineering (ECC error simulation, PCIe fault injection) — defer to v2.8
- Krylov subspace methods (CG, GMRES) — requires SpMV first, defer to v2.8
- Roofline model visualization — requires bandwidth baseline, defer to v2.8
- cuCollections — sm_70+ only, Pascal support incompatible

## Future Requirements

### Deferred from v2.7

- Chaos engineering framework
- Krylov subspace methods
- Roofline model integration

---
*Requirements defined: 2026-04-30*
*Roadmap created: 2026-04-30*
*Ready for planning: yes*
