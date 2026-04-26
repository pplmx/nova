# Project Research Summary

**Project:** Nova CUDA Library — v1.7 Benchmarking & Testing
**Domain:** CUDA/C++ Benchmarking and Performance Testing Infrastructure
**Researched:** 2026-04-26
**Confidence:** HIGH

## Executive Summary

v1.7 adds comprehensive benchmarking infrastructure to a production CUDA library. The existing codebase (C++23, CUDA 20, five-layer architecture, 444 tests) has strong foundations — a custom `cuda::benchmark::Benchmark` class already provides warmup, statistical aggregation, and throughput metrics. The gap is CI integration, baseline storage, NVTX profiling hooks, and a Python harness for orchestration and regression detection.

The recommended approach: Google Benchmark for C++ kernels (extend existing patterns), a Python pytest harness for orchestration/regression checking, NVTX annotations as optional instrumentation, and HTML dashboards for trend visualization. Key risks are GPU frequency scaling (unstable baselines) and NVTX overhead distorting measurements — both addressable with correct architecture in Phase 1.

## Key Findings

### Recommended Stack

**Core technologies:**
- **Google Benchmark v1.9.x** — Industry standard for C++ microbenchmarks with CMake integration, statistical analysis, JSON output. Extend the existing `cuda::benchmark::Benchmark` patterns rather than replacing them.
- **NVTX nvtx3 (CUDA bundled)** — Header-only C++ library for GPU profiling annotations. No linking required. Integrates with Nsight Systems/Compute automatically.
- **Python harness (pytest 8.x)** — Subprocess orchestration of C++ benchmark binaries, JSON result collection, statistical regression testing with `scipy.stats`.
- **HTML dashboards (plotly + chevron/Jinja2)** — Static HTML generation with interactive trend charts. No real-time complexity.

**Supporting stack:**
- `pandas` for JSON data processing
- `scipy.stats` for statistical significance testing (Welch's t-test)
- `plotly` for interactive HTML charts
- `pytest-xdist` for parallel benchmark execution
- `requests` for CI artifact storage

### Expected Features

**Must have (table stakes):**
- Baseline storage system (JSON files committed to repo)
- CI regression gates (GitHub Actions with threshold-based failure)
- Algorithmic benchmarks (reduce, scan, sort, FFT, matmul) with throughput + latency
- JSON export for all benchmark results

**Should have (competitive):**
- Memory profiling suite (leak detection, pool metrics, fragmentation analysis)
- Multi-GPU NCCL benchmark harness (all-reduce, broadcast, all-gather scaling curves)
- HTML report generator with trend charts and baseline comparison
- NVTX annotation framework for kernel timeline visualization in Nsight Systems

**Defer (v2+):**
- Pipeline parallelism benchmark suite
- Tensor parallelism benchmark suite
- Automated alert routing (Slack/Teams)
- Interactive Nsight CLI integration

### Architecture Approach

Four-layer integration into existing five-layer architecture:

1. **C++ Google Benchmark kernels** in `benchmark/` directory — compile as separate executable, link against `cuda_impl`, invoke from Python harness via subprocess
2. **Python harness** in `scripts/benchmark/` — orchestrates benchmark runs, parses JSON results, checks regression against baselines, generates dashboards
3. **NVTX annotations** in `include/cuda/benchmark/nvtx.h` — RAII scoped range guards, optional (compile-time toggle to avoid overhead in pure throughput mode)
4. **HTML dashboards** in `reports/` — generated from benchmark results and baselines using Jinja2 + plotly

Data flow: `benchmark binary → JSON → Python parser → regression check → HTML dashboard`

### Critical Pitfalls

1. **GPU frequency scaling** — Results vary 15-40% without clock locking. Fix: Phase 1 must establish fixed clock methodology and warmup patterns.
2. **Missing CUDA synchronization** — Async kernels report misleading times. Fix: Always `cudaEventSynchronize()` before reading timing.
3. **NVTX overhead distortion** — Annotations add microseconds to fine-grained kernels. Fix: Never wrap timing code inside NVTX ranges; separate measurement from annotation.
4. **CI non-determinism** — Cloud GPU variance causes false regressions. Fix: Statistical significance testing (Welch's t-test), multiple iterations, normalized comparison.
5. **Memory pool state contamination** — Pool effects cause first runs to be slower. Fix: Explicit warmup and allocator state reset.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Benchmark Infrastructure Foundation
**Rationale:** Establishes measurement methodology, CUDA event patterns, NVTX separation, and warmup protocols. All downstream phases depend on these being correct.
**Delivers:** Stable measurement infrastructure, NVTX annotation framework, CUDA event timing patterns, warmup protocol, Google Benchmark kernel template
**Avoids:** Pitfalls 1 (frequency scaling), 2 (synchronization), 6 (NVTX overhead), 8 (pool contamination)

### Phase 2: Comprehensive Benchmark Suite
**Rationale:** Core algorithmic benchmarks provide the measurement data for all downstream features. Input size coverage and realistic workloads prevent "looks done but isn't."
**Delivers:** Algorithmic benchmarks (reduce, scan, sort, FFT, matmul, memory ops), multi-GPU NCCL harness, scaling curve framework, baseline capture infrastructure
**Avoids:** Pitfall 3 (input size gaps)

### Phase 3: CI Regression Testing
**Rationale:** CI integration without statistical rigor causes false positives that erode trust. Builds on Phases 1-2 measurement stability.
**Delivers:** GitHub Actions workflow, baseline comparison with configurable tolerances, statistical significance gates, baseline management and staleness tracking
**Avoids:** Pitfalls 4 (CI non-determinism), 5 (baseline drift)

### Phase 4: Performance Dashboards
**Rationale:** Visual trend analysis and regression reporting completes the benchmarking story. Depends on all prior phases producing reliable data.
**Delivers:** HTML dashboard generator with trend charts, baseline comparison visualization, regression annotation, CI artifact integration
**Avoids:** Pitfall 7 (multi-GPU sync errors manifest in dashboard data)

### Phase Ordering Rationale

- Phase 1 must come first — measurement methodology underpins everything else
- Phase 2 before Phase 3 — CI gates need benchmarks to run and baselines to compare against
- Phase 4 last — dashboards consume data from Phases 2-3

### Research Flags

- **Phase 2:** Multi-GPU NCCL benchmarks need hardware topology detection (existing DeviceMesh may help)
- **Phase 3:** CI GPU runner selection (dedicated vs. cloud) affects statistical thresholds
- **Phase 4:** Dashboard hosting strategy (GitHub Pages, internal server, artifact attachment)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Context7-verified for Google Benchmark, NVTX, pytest, plotly |
| Features | HIGH | CUDA benchmarking domain well-understood, existing codebase patterns extend cleanly |
| Architecture | HIGH | Layer integration patterns clear, dependency order confirmed by pitfall analysis |
| Pitfalls | HIGH | All 8 pitfalls verified against NVIDIA best practices and CUDA profiling guides |

**Overall confidence:** HIGH — research aligns with existing codebase patterns.

### Gaps to Address

- **Baseline storage location:** git-committed `scripts/benchmark/baselines/` vs. GitHub artifact storage — decide based on team preferences
- **Regression thresholds per algorithm type:** Memory-bound vs. compute-bound may need different tolerances (suggest 5-10% default, configurable)

## Sources

### Primary (HIGH confidence)
- Google Benchmark official docs — Custom timers, JSON output, CMake integration
- NVIDIA NVTX documentation — nvtx3 header-only C++ API, scoped_marker patterns
- NVIDIA Nsight Systems profiling guide — NVTX timeline visualization

### Secondary (MEDIUM confidence)
- NCCL Tests repository (NVIDIA/nccl-tests) — Collective communication benchmark patterns
- CUTLASS benchmarks — GEMM benchmarking patterns and scaling curves

### Tertiary (LOW confidence)
- Cloud GPU benchmarking variance studies — general patterns only, need validation against specific CI hardware

---
*Research completed: 2026-04-26*
*Ready for roadmap: yes*
