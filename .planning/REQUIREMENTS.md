# Requirements: Nova CUDA Library — v1.7 Benchmarking & Testing

**Defined:** 2026-04-26
**Core Value:** A reliable, high-performance CUDA compute library that can be trusted in production environments, with comprehensive algorithms for scientific computing, image processing, and emerging workloads.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Benchmark Infrastructure Foundation (BENCH)

- [ ] **BENCH-01**: Developer can run benchmark suite with `python scripts/benchmark/run_benchmarks.py --all`
- [ ] **BENCH-02**: Benchmarks use CUDA events for accurate wall-clock timing with proper synchronization
- [ ] **BENCH-03**: Benchmarks include warmup iterations before measurement to stabilize GPU clocks
- [ ] **BENCH-04**: NVTX annotation framework provides scoped range guards with compile-time toggle
- [ ] **BENCH-05**: NVTX annotations can be disabled without affecting benchmark timing accuracy

### Comprehensive Benchmark Suite (SUITE)

- [ ] **SUITE-01**: Developer can benchmark reduce operations (float, double) with configurable input sizes
- [ ] **SUITE-02**: Developer can benchmark scan operations (inclusive, exclusive) with configurable input sizes
- [ ] **SUITE-03**: Developer can benchmark sort operations with configurable input sizes
- [ ] **SUITE-04**: Developer can benchmark FFT operations (forward, inverse) with configurable input sizes
- [ ] **SUITE-05**: Developer can benchmark matmul operations with configurable input sizes and batch dimensions
- [ ] **SUITE-06**: Benchmark results include throughput metrics (GB/s, items/sec) and latency (ms)
- [ ] **SUITE-07**: Developer can benchmark memory operations (H2D, D2H, D2D) with throughput metrics
- [ ] **SUITE-08**: Multi-GPU benchmarks measure NCCL collective operations (all-reduce, broadcast, all-gather)
- [ ] **SUITE-09**: Benchmarks support parameterized input sizes across meaningful ranges (small to production-scale)

### CI Regression Testing (CI)

- [ ] **CI-01**: Benchmark results are exported as machine-readable JSON files
- [ ] **CI-02**: Baseline JSON files are committed to `scripts/benchmark/baselines/` with version metadata
- [ ] **CI-03**: Python harness compares results against baselines with configurable tolerance thresholds
- [ ] **CI-04**: Regression detection uses statistical significance testing (Welch's t-test) to reduce false positives
- [ ] **CI-05**: GitHub Actions workflow runs benchmarks on PR and fails if regression exceeds threshold
- [ ] **CI-06**: CI workflow stores baseline freshness metadata and alerts when baselines are stale
- [ ] **CI-07**: Regression check failures include clear output showing which benchmark regressed, by how much, and the baseline value

### Performance Dashboards (DASH)

- [ ] **DASH-01**: Python script generates HTML dashboard from benchmark JSON results
- [ ] **DASH-02**: Dashboard displays benchmark results in tabular format with time, throughput, and iterations
- [ ] **DASH-03**: Dashboard shows trend charts comparing current results against baselines
- [ ] **DASH-04**: Dashboard highlights regressions in red, improvements in green, and stable results in neutral color
- [ ] **DASH-05**: Dashboard includes hardware context (GPU model, driver version, CUDA version)
- [ ] **DASH-06**: Dashboard generates self-contained HTML that can be served statically or attached as CI artifact

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Distributed Training Benchmarks

- **DIST-01**: Developer can benchmark tensor parallelism operations with scaling curves across GPU counts
- **DIST-02**: Developer can benchmark pipeline parallelism schedules (1F1B, interleaved) with throughput metrics

### Advanced Profiling

- **PROF-01**: Developer can launch Nsight Systems from benchmark tooling for interactive GPU analysis
- **PROF-02**: Benchmark reports include L2 cache hit rates and memory efficiency metrics
- **PROF-03**: Developer can trigger automated alert routing (Slack/email) on regression detection

### Memory Profiling

- **MEMP-01**: Developer can detect memory leaks by comparing allocation count before and after benchmark runs
- **MEMP-02**: Developer can measure memory pool fragmentation metrics across benchmark iterations

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time dashboard streaming | Continuous GPU polling adds overhead; periodic JSON + static HTML is sufficient |
| Automatic kernel tuning | Takes hours to run, changes kernel behavior unexpectedly |
| Cross-vendor validation (AMD/Intel) | Different architectures yield different metrics; focus on NVIDIA |
| ML-based anomaly detection | False positives in noisy GPU workloads; statistical tolerance bands are more reliable |
| Per-instruction SASS profiling | Prohibitive overhead; offer as opt-in manual tool (Nsight Compute) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BENCH-01 | Phase 29 | Pending |
| BENCH-02 | Phase 29 | Pending |
| BENCH-03 | Phase 29 | Pending |
| BENCH-04 | Phase 29 | Pending |
| BENCH-05 | Phase 29 | Pending |
| SUITE-01 | Phase 30 | Pending |
| SUITE-02 | Phase 30 | Pending |
| SUITE-03 | Phase 30 | Pending |
| SUITE-04 | Phase 30 | Pending |
| SUITE-05 | Phase 30 | Pending |
| SUITE-06 | Phase 30 | Pending |
| SUITE-07 | Phase 30 | Pending |
| SUITE-08 | Phase 30 | Pending |
| SUITE-09 | Phase 30 | Pending |
| CI-01 | Phase 31 | Pending |
| CI-02 | Phase 31 | Pending |
| CI-03 | Phase 31 | Pending |
| CI-04 | Phase 31 | Pending |
| CI-05 | Phase 31 | Pending |
| CI-06 | Phase 31 | Pending |
| CI-07 | Phase 31 | Pending |
| DASH-01 | Phase 32 | Pending |
| DASH-02 | Phase 32 | Pending |
| DASH-03 | Phase 32 | Pending |
| DASH-04 | Phase 32 | Pending |
| DASH-05 | Phase 32 | Pending |
| DASH-06 | Phase 32 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-26*
*Last updated: 2026-04-26 after requirements definition*
