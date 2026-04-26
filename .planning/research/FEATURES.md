# Feature Research

**Domain:** CUDA/C++ Benchmarking and Performance Testing Infrastructure
**Researched:** 2026-04-26
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Basic throughput measurement** | Core benchmark requirement | LOW | Already implemented in `cuda::benchmark::Benchmark` |
| **Latency measurement** | Kernel timing is fundamental | LOW | Already implemented via cudaEvent timing |
| **Statistical aggregation** | Single runs are noisy | LOW | Mean, stddev, min, max already implemented |
| **Regression detection** | Catch performance degradation | MEDIUM | Tolerance-based comparison exists; needs baseline storage |
| **Warm-up iterations** | GPU clocks need stabilization | LOW | Already implemented |
| **CI integration** | Automated performance validation | MEDIUM | Requires JSON baseline storage and CI workflow |
| **Memory throughput metrics** | GPU memory bandwidth is critical | LOW | `compute_throughput_gbps` helper exists |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Multi-GPU scaling curves** | Validate distributed training scalability | HIGH | Requires NCCL integration, node coordination |
| **Automated regression alerting** | Proactive performance monitoring | MEDIUM | GitHub Actions with threshold-based gates |
| **NVTX-instrumented kernels** | Timeline visualization of kernel execution | MEDIUM | NVIDIA Nsight Systems integration |
| **Memory leak detection** | Catch allocation bugs in benchmarks | MEDIUM | Track allocations per benchmark run |
| **Pool fragmentation metrics** | Understand memory allocator behavior | MEDIUM | Requires memory pool instrumentation |
| **HTML trend reports** | Visual performance history | MEDIUM | Chart.js or similar for browser-based dashboards |
| **Weak/strong scaling benchmarks** | Validate parallel efficiency | HIGH | Algorithmic scalability measurement |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Real-time dashboard streaming** | "Would be cool" appeal | Continuous GPU state polling adds overhead; complexity explosion | Periodic JSON exports + static HTML |
| **Automatic kernel tuning** | Optimize launch parameters | Takes hours to run; changes kernel behavior | Document optimal configs separately |
| **Cross-vendor validation** | AMD/Intel compatibility | Different architectures, different metrics, massive scope | Focus on NVIDIA, document limitations |
| **Machine learning-based anomaly detection** | "AI will find everything" | False positives in noisy GPU workloads | Statistical tolerance bands are sufficient |
| **Per-instruction profiling** | SASS-level granularity | Prohibitive overhead; requires NSight Compute | Offer as opt-in manual tool |

## Feature Dependencies

```
Comprehensive Benchmark Suite
    ├──requires──> Statistical Aggregation (exists)
    ├──requires──> Throughput Calculation (exists)
    └──requires──> CI Integration Pipeline

Performance Regression Testing
    ├──requires──> Baseline Storage (JSON files)
    ├──requires──> Regression Detection (exists)
    └──requires──> CI Integration Pipeline

Memory Profiling & Validation
    ├──requires──> Memory Metrics (exists)
    ├──requires──> Allocation Tracking
    └──requires──> Fragmentation Analysis

Distributed Training Benchmarks
    ├──requires──> NCCL Integration (exists in codebase)
    ├──requires──> Multi-GPU Context Management
    └──requires──> Scaling Curve Framework

Continuous Profiling Hooks
    ├──requires──> NVTX Integration
    ├──requires──> Profiler Infrastructure (exists)
    └──requires──> NSight CLI Integration

Performance Dashboards
    ├──requires──> JSON Export (partial - exists)
    ├──requires──> Baseline Comparison
    └──requires──> HTML Generation Framework
```

### Dependency Notes

- **Baseline Storage requires CI Integration:** Without automated CI runs, baselines become stale and useless.
- **Distributed Benchmarks enhance Comprehensive Suite:** Multi-GPU tests are a superset of single-GPU benchmarking.
- **NVTX hooks conflict with pure throughput measurement:** NVTX adds overhead; separate measurement modes needed.
- **Memory leak detection conflicts with pool fragmentation:** Different memory tracking strategies; don't combine in same run.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] **Baseline storage system** — JSON files for persisting benchmark results across commits. Essential for regression testing. Storage format: `{ "commit": "...", "results": [...] }`.
- [ ] **CI integration with regression gates** — GitHub Actions workflow that runs benchmarks, compares to stored baselines, and fails PR if tolerance exceeded. Threshold: 10% default, configurable per-benchmark.
- [ ] **Algorithmic benchmarks (reduce, scan, sort)** — These expose scaling behavior clearly. Throughput + latency per problem size.
- [ ] **JSON export for all benchmark results** — Machine-readable output enabling trend analysis and dashboard generation.

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **Memory profiling suite** — Leak detection (allocation count mismatch), pool utilization metrics, fragmentation analysis. Trigger: Memory bugs appearing in CI.
- [ ] **Multi-GPU benchmark harness** — NCCL-based collective benchmarks (all-reduce, broadcast, all-gather) with scaling curves. Trigger: Distributed training needs.
- [ ] **HTML report generator** — Static pages with Chart.js trend charts, baseline comparison, regression annotations. Trigger: Team grows beyond 3 people.
- [ ] **NVTX annotation framework** — Lightweight range markers for kernel-level timeline in Nsight Systems. Trigger: Profiling sessions needed.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Pipeline parallelism benchmarks** — Micro-benchmarks for 1F1B schedule, interleaved schedules. Defer: Requires full distributed context.
- [ ] **Tensor parallelism benchmarks** — All-reduce + all-gather patterns at scale. Defer: Hardware-dependent validation.
- [ ] **Automatic alert routing** — Slack/Teams notifications on regressions. Defer: After baseline storage validated.
- [ ] **Interactive Nsight integration** — Launch Nsight Systems from benchmark tooling. Defer: UI complexity.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Baseline storage system | HIGH | LOW | P1 |
| CI regression gates | HIGH | MEDIUM | P1 |
| Algorithmic benchmarks (reduce/scan/sort) | HIGH | LOW | P1 |
| JSON export for results | HIGH | LOW | P1 |
| Memory leak detection | MEDIUM | MEDIUM | P2 |
| Memory pool metrics | MEDIUM | MEDIUM | P2 |
| Multi-GPU NCCL benchmarks | MEDIUM | HIGH | P2 |
| HTML trend reports | MEDIUM | MEDIUM | P2 |
| NVTX annotation framework | MEDIUM | MEDIUM | P2 |
| FFT benchmarks | MEDIUM | LOW | P2 |
| Matmul benchmarks | MEDIUM | LOW | P2 |
| Collective scaling curves | LOW | HIGH | P3 |
| Pipeline parallelism suite | LOW | HIGH | P3 |
| Tensor parallelism suite | LOW | HIGH | P3 |
| Alert routing integration | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Google Benchmark | NVIDIA Nsight Compute | cuBLAS/cuFFT Benchmarks | Our Approach |
|---------|------------------|----------------------|-------------------------|--------------|
| **Basic microbenchmarking** | Yes - CPU only | CLI profiling | Reference implementations | Extend existing `cuda::benchmark::Benchmark` |
| **Kernel throughput** | Via manual timing | Metrics collection | Throughput in docs | First-class citizen with GB/s metrics |
| **Statistical aggregation** | Built-in with reporters | Via ncu-rep parsing | Manual | Mean/stddev/min/max built-in |
| **Baseline comparison** | Custom reporters needed | Via `ncu --diff` | None | JSON-based with tolerance thresholds |
| **CI integration** | CMake/CTest | GitHub Actions samples | None | Native GitHub Actions workflow |
| **Multi-GPU support** | No | Via mpirun/NCCL | No | NCCL-aware benchmark harness |
| **HTML reports** | Via custom reporters | Nsight Systems GUI | None | Static site generator |
| **Memory leak detection** | AddressSanitizer | Via replay | None | Integration with existing memory pool |

## Sources

- [Google Benchmark Library](https://github.com/google/benchmark) - CPU microbenchmarking standard
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/) - GPU profiling guide
- [NVIDIA NVTX Documentation](https://docs.nvidia.com/nsight-compute/) - Timeline annotation
- [CUTLASS Benchmarks](https://github.com/NVIDIA/cutlass) - GEMM benchmarking patterns
- [cuBLAS Benchmarks](https://docs.nvidia.com/cuda/cublas/) - BLAS performance reference
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests) - Collective communication benchmarks

---
*Feature research for: CUDA/C++ Benchmarking and Performance Testing Infrastructure*
*Researched: 2026-04-26*
