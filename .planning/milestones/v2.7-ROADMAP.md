# Milestone v2.7 Roadmap

**Project:** Nova CUDA Library Enhancement
**Milestone:** v2.7 Comprehensive Testing & Validation
**Granularity:** Standard
**Coverage:** 16/16 requirements mapped (100%)

---

## Phases

- [ ] **Phase 75: Observability & Profiling** - Timeline visualization, bandwidth measurement, kernel statistics, occupancy analysis
- [ ] **Phase 76: Algorithm Extensions** - Segmented sort, SpMV, sample sort, delta-stepping SSSP
- [ ] **Phase 77: Robustness & Testing** - Memory safety validation, isolated contexts, error injection, boundary testing, FP determinism
- [ ] **Phase 78: Integration & Validation** - End-to-end robustness, memory safety validation, baselines, documentation

---

## Phase Details

### Phase 75: Observability & Profiling

**Goal:** Users can visualize kernel execution timelines, measure memory bandwidth, collect kernel statistics, and analyze occupancy

**Depends on:** Phase 74 (InferenceGraphExecutor, NVTX domains)

**Requirements:** OBS-01, OBS-02, OBS-03, OBS-04

**Success Criteria** (what must be TRUE):

1. User can export Chrome trace format files from NVTX annotations and load them in chrome://tracing
2. User can measure H2D, D2H, and D2D memory bandwidth using NVbandwidth integration
3. User can collect per-kernel statistics (latency, throughput, achieved occupancy) via profiler integration
4. User can receive real-time feedback on block size selection with occupancy recommendations

**Plans:** TBD

**UI hint:** yes

---

### Phase 76: Algorithm Extensions

**Goal:** Users can leverage advanced algorithms for segmented operations, sparse computation, large-scale sorting, and graph shortest paths

**Depends on:** Phase 75

**Requirements:** ALGO-01, ALGO-02, ALGO-03, ALGO-04

**Success Criteria** (what must be TRUE):

1. User can sort elements within arbitrary segments without full array copy using segmented sort
2. User can perform sparse matrix-vector multiply with CSR/CSC formatted matrices from v2.1
3. User can sort datasets exceeding single-pass capacity using sample sort when radix sort is inefficient
4. User can compute single-source shortest paths using delta-stepping for weighted graphs

**Plans:** TBD

---

### Phase 77: Robustness & Testing

**Goal:** Users can validate memory safety, run isolated tests, inject errors, test boundary conditions, and control floating-point determinism

**Depends on:** Phase 76

**Requirements:** ROB-01, ROB-02, ROB-03, ROB-04, ROB-05

**Success Criteria** (what must be TRUE):

1. User can run Compute Sanitizer to detect UAF, double-free, and uninitialized memory access across all algorithms
2. User can execute tests in isolated CUDA contexts with no state pollution between test cases
3. User can inject errors at layer boundaries (Memory, Device, Algorithm, Stream, Inference) via error injection framework
4. User can test CUDA-specific boundaries including 256-byte alignment, warp size, and SM limits
5. User can control FP determinism at three levels: not_guaranteed, run_to_run, gpu_to_gpu

**Plans:** TBD

---

### Phase 78: Integration & Validation

**Goal:** Users can run comprehensive end-to-end tests with profiling, validate all algorithms for memory safety, establish baselines, and access documentation

**Depends on:** Phase 77

**Requirements:** INT-01, INT-02, INT-03, INT-04

**Success Criteria** (what must be TRUE):

1. User can run end-to-end robustness tests with simultaneous NVTX profiling enabled
2. User can validate memory safety across all algorithm implementations using automated test suite
3. User can establish performance regression baselines and compare subsequent runs against them
4. User can access updated documentation covering new observability features, algorithms, and testing capabilities

**Plans:** TBD

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 75. Observability & Profiling | 0/1 | Not started | - |
| 76. Algorithm Extensions | 0/1 | Not started | - |
| 77. Robustness & Testing | 0/1 | Not started | - |
| 78. Integration & Validation | 0/1 | Not started | - |

---

## Coverage Map

```
OBS-01 → Phase 75 (Timeline visualization)
OBS-02 → Phase 75 (Memory bandwidth)
OBS-03 → Phase 75 (Kernel statistics)
OBS-04 → Phase 75 (Occupancy analysis)
ALGO-01 → Phase 76 (Segmented sort)
ALGO-02 → Phase 76 (SpMV CSR/CSC)
ALGO-03 → Phase 76 (Sample sort)
ALGO-04 → Phase 76 (Delta-stepping SSSP)
ROB-01 → Phase 77 (Compute Sanitizer)
ROB-02 → Phase 77 (Isolated contexts)
ROB-03 → Phase 77 (Error injection)
ROB-04 → Phase 77 (Boundary testing)
ROB-05 → Phase 77 (FP determinism)
INT-01 → Phase 78 (E2E robustness + profiling)
INT-02 → Phase 78 (Memory safety validation)
INT-03 → Phase 78 (Performance baselines)
INT-04 → Phase 78 (Documentation)

Mapped: 16/16 ✓
No orphaned requirements ✓
```

---

*Roadmap created: 2026-04-30*
*v2.7 phases: 75, 76, 77, 78*
