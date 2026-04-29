# Project Research Summary

**Project:** Nova CUDA Library v2.7
**Domain:** GPU robustness testing, performance profiling, and advanced algorithms
**Researched:** 2026-04-30
**Confidence:** HIGH

## Executive Summary

The v2.7 milestone extends the Nova CUDA library's production readiness through three complementary areas: (1) enhanced robustness testing building on the v2.4 error injection framework, (2) GPU profiling enhancements for memory bandwidth and timeline visualization, and (3) advanced algorithms extending v2.3 numerical methods work. Research confirms that most required tooling is already in the stack or available from NVIDIA at no additional cost—the only critical migration is CUB to CCCL. Experts in this domain rely on NVIDIA's Compute Sanitizer for memory safety (ASan does not work with CUDA GPU code), Nsight tools for profiling, and layered testing approaches that isolate fault injection at layer boundaries.

Key risks include test isolation failures from shared CUDA context state, profiler overhead distorting measurements, and known CONCERNS.md issues (SyncBatchNorm backward pass, memory leaks in error paths) that will surface under stress testing. The recommended approach sequences observability infrastructure first, then algorithms, then robustness framework—this enables each phase to validate the previous one while avoiding the pitfall of testing without baselines.

## Key Findings

### Recommended Stack

**Summary:** v2.7 stack additions are minimal—most tooling is already available via NVIDIA's CUDA Toolkit. The critical action is migrating CUB to CCCL (CUB is archived); everything else is optional or already present.

**Core technologies:**
- **CCCL 2.6.0** — Required migration from archived CUB; provides backward-compatible headers and active maintenance
- **NVIDIA Compute Sanitizer** — Memory safety validation (memcheck, racecheck, initcheck, synccheck); part of CUDA 12+, no installation needed
- **Nsight Compute CLI** — Kernel-level profiling with detailed metrics; integrated via CMake detection
- **Nsight Systems v2026.2** — Timeline visualization for multi-stream interactions; separate download (free registration)
- **NVbandwidth** — Memory bandwidth measurement (H2D/D2H/D2D); for Phase 2 roofline model integration
- **Tracy Profiler v0.13.1** — Optional open-source alternative to Nsight for CI/continuous profiling

**Already in stack (do NOT re-add):** Google Test v1.17.0, libFuzzer, property-based tests, NVTX, CUDA Events, Google Benchmark v1.9.0, error injection framework (v2.4), memory pressure tests (v2.4).

### Expected Features

**Must have (table stakes):**
- **FP Determinism Control** — CCCL 3.1 provides `not_guaranteed`, `run_to_run`, `gpu_to_gpu` levels; essential for reproducible scientific computing
- **Memory Safety Validation** — Compute Sanitizer integration; detects UAF, double-free, uninitialized memory on GPU
- **Boundary Condition Tests** — Must include CUDA-specific boundaries (256-byte alignment, warp size=32, SM limits)
- **Memory Bandwidth Measurement** — NVbandwidth or custom timing kernels; bandwidth limits most GPU workloads
- **Timeline Visualization** — Chrome trace format export from existing NVTX infrastructure
- **Segmented Sort** — Sort within groups without full copy; foundation for sparse operations
- **SpMV (CSR/CSC)** — Sparse matrix-vector multiply for scientific computing; uses v2.1 formats

**Should have (competitive):**
- **Statistical Testing** — Hypothesis testing for numerical algorithm outputs
- **Determinism Verification** — Automated regression suite for bit-exact output
- **Occupancy Calculator Integration** — Real-time occupancy feedback via cudaOccupancyMaxPotentialBlockSize
- **Sample Sort** — For large datasets beyond radix sort efficiency
- **Delta-Stepping SSSP** — Single-source shortest path for graph algorithms

**Defer (v2+):**
- **Chaos Engineering** — High complexity; requires deep integration; simulates ECC errors, PCIe faults
- **Krylov Subspace Methods** — Requires SpMV first; conjugate gradient, GMRES
- **Roofline Model** — Requires bandwidth baseline first; high complexity
- **cuCollections** — Not needed for basic sorting/scanning; only for concurrent hash tables (sm_70+ only, no Pascal support)

### Architecture Approach

Research identifies three distinct architectural patterns for v2.7 features:

1. **Robustness Testing** is a **horizontal cross-cutting concern** that must inject faults at every layer boundary (Memory, Device, Algorithm, Stream, Inference, Production). Primary modifications in Production layer via `fault_injector.h`, `chaos_scenarios.h`, `memory_safety.h`.

2. **Profiling Enhancements** are an **observability extension** building on existing v1.6/v1.7/v2.4 infrastructure (NVTX, NVBench, CUDA Events). Components include `timeline.h`, `bandwidth_tracker.h`, `kernel_stats.h`; all modifications are additive.

3. **Advanced Algorithms** are a **vertical extension** to the Algorithm layer, adding new algorithm families (sample sort, SSSP, iterative methods) following existing `Buffer<T>` API patterns. No data flow changes.

### Critical Pitfalls

1. **Test Isolation Failures** — CUDA context state persists between tests; shared singletons cause pollution. Must reset per-test CUDA context and singletons. Tests pass in isolation but fail in CI.

2. **Error Injection Timing** — Injected errors at wrong layer (CUDA API vs. exception) never exercise recovery code; cleanup paths never tested. Must use layer-aware injection and verify memory cleanup after failures.

3. **Memory Safety Tool Mismatch** — ASan/MSan don't track GPU memory. Must use Compute Sanitizer or poison patterns. SyncBatchNorm memory leaks (CONCERNS.md:430-432) will surface.

4. **Profiler Overhead Distortion** — NVTX calls per iteration add overhead that distorts timing measurements. Must measure overhead separately and subtract; warmup runs required.

5. **Boundary Condition Blind Spots** — Tests use CPU boundaries (0, 1, 1024, 65536) but miss CUDA-specific boundaries (256-byte alignment, warp size, SM limits). Will cause production failures.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Observability Foundation
**Rationale:** Profiling tools help validate correctness and performance of other features. NVTX annotations enable fault detection in Phase 3.
**Delivers:** Timeline visualization (Chrome trace export), bandwidth_tracker.h, kernel_stats.h, expanded NVTX domains
**Addresses:** Table stakes timeline visualization, memory bandwidth measurement
**Avoids:** Profiler overhead distortion (establish overhead budgets early), timeline misinterpretation (document hardware concurrency limits)
**Research Flag:** MEDIUM — Timeline format (Chrome vs custom) needs decision; investigate NVTX domain naming conventions

### Phase 2: Algorithm Extensions
**Rationale:** Algorithms are independent additions building on core layers; no risk to existing functionality. Enables Phase 3 chaos testing targets.
**Delivers:** device/warp_graph.h, algo/sample_sort.h, algo/graph/sssp.h, algo/numerical/iterative.h, segmented sort, SpMV implementation
**Addresses:** Should-have competitive algorithms, deferred v2+ features (partial)
**Avoids:** Algorithm API inconsistency (establish unified Algorithm base class), numerical instability (NaN-aware comparisons, Kahan summation)
**Research Flag:** MEDIUM — Specific algorithm variants need validation; CUDA architecture target (sm_60 Pascal support) affects cuCollections addition

### Phase 3: Robustness Framework
**Rationale:** Testing framework benefits from having target algorithms ready; observability from Phase 1 needed for fault detection.
**Delivers:** fault_injector.h, chaos_scenarios.h, memory_safety.h, boundary condition tests, CI integration
**Addresses:** Must-have memory safety, determinism control, statistical testing
**Avoids:** Test isolation failures (per-test CUDA context reset), error injection timing (layer-aware injection), memory safety gaps (GPU-specific tools, not ASan)
**Research Flag:** HIGH — Known CONCERNS.md issues (SyncBatchNorm:528-543, memory leaks:430-432) must be fixed during this phase; requires coordination with existing issues

### Phase 4: Integration & Validation
**Rationale:** Final integration ensures all pieces work together; establishes performance regression baselines.
**Delivers:** End-to-end robustness tests with profiling, memory safety validation on all algorithms, documentation, regression baselines
**Addresses:** Verification of all previous phases
**Avoids:** Profiling without baselines (store in version control), adding tests without fixing issues (policy: discovery = bug filing)
**Research Flag:** LOW — Validation phase follows established patterns

### Phase Ordering Rationale

- **Observability first:** Validates all subsequent work; profiling annotations for Phase 2 algorithms enable Phase 3 fault detection
- **Algorithms before robustness:** Chaos testing needs targets; testing empty framework provides no value
- **Robustness before validation:** Must have tests that exercise new algorithms before validating
- **Known issues fixed in Phase 3:** Memory leaks and empty backward pass will surface under stress testing; address alongside test framework

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Timeline format preference (Chrome Traces vs custom binary); NVTX domain naming conventions for consistency
- **Phase 2:** Algorithm variant selection (sample sort vs adaptive radix sort); sm_60 Pascal compatibility decision
- **Phase 3:** Deterministic fault replay requirements (state capture architecture); CUDA-MEMCHECK vs runtime detection preference

Phases with standard patterns (skip research-phase):
- **Phase 1:** Compute Sanitizer and NVTX are well-documented by NVIDIA; established patterns
- **Phase 4:** Validation follows existing test infrastructure; no novel patterns needed

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | NVIDIA official sources; CCCL migration path documented; tools are CUDA Toolkit components |
| Features | HIGH | Based on industry testing patterns and NVIDIA guidance; existing v2.x patterns inform scope |
| Architecture | HIGH | Integration points from explicit layer documentation; clear separation of concerns |
| Pitfalls | HIGH | Based on established CUDA testing patterns, NVIDIA profiling guides, and existing codebase issues |

**Overall confidence:** HIGH

### Gaps to Address

- **CCCL 3.1 availability:** FP determinism control requires CCCL 3.1; verify version in current CUDA Toolkit before relying on this feature
- **sm_60 Pascal support:** cuCollections and potentially other features require sm_70+; clarify if Pascal support is still needed
- **Fault injection replay:** Architecture mentions deterministic replay as open question; this affects test design
- **CONCERNS.md alignment:** Known issues (SyncBatchNorm backward pass, memory leaks) predate v2.7; ensure cleanup tracked in roadmap

## Sources

### Primary (HIGH confidence)
- [NVIDIA Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/) — Memory safety validation
- [NVIDIA Nsight Compute v13.2](https://docs.nvidia.com/nsight-compute/) — Kernel profiling
- [NVIDIA Nsight Systems v2026.2](https://docs.nvidia.com/nsight-systems/) — Timeline visualization
- [CCCL GitHub](https://github.com/nvidia/cccl) — CUB migration, determinism control
- [NVIDIA NVbandwidth](https://developer.nvidia.com/blog/nvidia-nvbandwidth-your-essential-tool-for-measuring-gpu-interconnect-and-memory-performance/) — Memory bandwidth measurement

### Secondary (MEDIUM confidence)
- [cuCollections GitHub](https://github.com/NVIDIA/cuCollections) — Concurrent data structures; active development, API may change
- [Tracy Profiler v0.13.1](https://github.com/wolfpld/tracy) — Alternative profiler; less CUDA-native than Nsight
- [cuGraph GitHub](https://github.com/rapidsai/cugraph) — Graph algorithm reference patterns

### Tertiary (LOW confidence)
- [cuCollections sm_60 support] — Dropped Feb 2026; verify if current systems require Pascal support

---

*Research completed: 2026-04-30*
*Ready for roadmap: yes*
