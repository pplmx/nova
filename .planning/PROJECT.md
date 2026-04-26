# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Current Milestone: v2.1 New Algorithms

**Previous milestone:** v2.0 Testing & Quality — SHIPPED 2026-04-26

**Status:** ✅ COMPLETE

**Goal:** Add new algorithm capabilities including sparse matrices, graph neural networks, and quantization.

**Target features:**
- Sparse Matrix Support: CSR/CSC formats, sparse matmul, SpMM kernels
- Graph Neural Networks: GNN primitives, message passing, graph attention
- Quantization: INT8/FP16 quantization, QAT support, mixed precision

**Completed:** 2026-04-26

## Core Value

A reliable, high-performance CUDA compute library that can be trusted in production environments, with comprehensive algorithms for scientific computing, image processing, and emerging workloads.

## Requirements

### Validated

- ✓ Five-layer CUDA architecture (memory → device → algo → api) — existing
- ✓ Memory management (Buffer, unique_ptr, MemoryPool) — existing
- ✓ Algorithm wrappers (reduce, scan, sort, histogram) — existing
- ✓ Image processing (blur, sobel, morphology, brightness) — existing
- ✓ Matrix operations (add, mult, ops) — existing
- ✓ Device capability queries and auto block size selection — v1.0
- ✓ Memory pool statistics and fragmentation reporting — v1.0
- ✓ Stream-based async operations with event synchronization — v1.0
- ✓ Signal/image processing via FFT — v1.0
- ✓ Ray tracing primitives (ray-box, ray-sphere, BVH) — v1.0
- ✓ Graph processing (BFS, PageRank) — v1.0
- ✓ Deep learning primitives (matmul, activations, normalization) — v1.0
- ✓ Device mesh detection and peer memory access — v1.1
- ✓ Multi-GPU data parallelism primitives (reduce, broadcast, all-gather, barrier) — v1.1
- ✓ Distributed memory pool across GPU devices — v1.1
- ✓ Multi-GPU matmul with single-GPU fallback — v1.1
- ✓ C++23 standard (CMAKE_CXX_STANDARD 23) — v1.2
- ✓ CUDA 20 standard (CMAKE_CUDA_STANDARD 20) — v1.2
- ✓ CMake 4.0+ minimum version — v1.2
- ✓ 444 tests passing — v1.2
- ✓ NCCL integration for optimized multi-GPU collectives — v1.3
- ✓ Extended NCCL collectives with unified fallback — v1.3
- ✓ Tensor parallelism for large layer support — v1.3
- ✓ Pipeline parallelism for deep model support — v1.3
- ✓ MPI-based NCCL initialization for inter-node communication — v1.4
- ✓ Topology-aware collective selection across nodes — v1.4
- ✓ Cross-node NCCL communicator management — v1.4
- ✓ Fuzz testing infrastructure with libFuzzer — v2.0
- ✓ Property-based testing with reproducible seeds — v2.0
- ✓ Code coverage reports with lcov/genhtml — v2.0
- ✓ CI integration with coverage gates — v2.0
- ✓ Sparse matrix CSR/CSC formats — v2.1
- ✓ Graph neural network primitives — v2.1
- ✓ INT8/FP16 quantization — v2.1

### Completed (v1.6)

- [x] Distributed batch normalization with cross-GPU sync — Phase 25
- [x] Performance profiling infrastructure — Phase 26
- [x] Kernel fusion for training efficiency — Phase 27
- [x] Memory optimization (compression, accumulation) — Phase 28

### Completed (v1.7)

- [x] Comprehensive benchmark suite with Google Benchmark + Python harness — Phase 29
- [x] Performance regression testing with automated detection — Phase 30
- [x] Continuous profiling hooks (NVTX, CI baseline comparison) — Phase 31
- [x] Performance dashboards (HTML reports, trend charts, regression alerts) — Phase 32

- [x] GPU checkpoint/restart with full state serialization — Phase 21
- [x] Communication error recovery for NCCL/TCP failures — Phase 22
- [x] Memory error detection and ECC error handling — Phase 23
- [x] Job preemption signal handling — Phase 24

### Out of Scope

- Python bindings — separate project
- Real-time video processing pipeline — not in scope

## Context

**Project:** nova CUDA library at `https://github.com/pplmx/nova`
- **Current:** C++23, CUDA 20, CMake 4.0+
- Target architectures: 6.0, 7.0, 8.0, 9.0 (Pascal through Ampere)
- Five-layer architecture with clear separation of concerns
- **444 tests using Google Test v1.14.0**
- **v1.2 shipped:** Toolchain upgrade (C++23, CUDA 20, CMake 4.0)

**Current capabilities:**
- Device mesh detection and peer memory access between GPUs
- Multi-GPU collective operations (all-reduce, broadcast, all-gather, barrier)
- Distributed memory pool spanning multiple GPUs
- Multi-GPU matrix multiply with single-GPU fallback
- All v1.0-v1.5 features: FFT, Ray Tracing, Graph Algorithms, Neural Net Primitives, Async/Streaming, NCCL, Tensor Parallelism, Pipeline Parallelism, Fault Tolerance

**Added in v1.4:**
- MPI-based NCCL bootstrapping for multi-node
- Topology-aware collective algorithm selection
- Hierarchical cross-node communicators

**Added in v1.5:**
- GPU checkpoint/restart with full state serialization
- Communication error recovery with exponential backoff
- Memory error detection and device health monitoring
- Job preemption signal handling (SIGTERM/SIGUSR1)

**Added in v1.6:**
- Distributed batch normalization with NCCL all-reduce
- CUDA event-based kernel profiling infrastructure
- Matmul-bias-activation kernel fusion
- ZSTD/LZ4 checkpoint compression and gradient buffering

**Added in v1.7:**
- Comprehensive benchmark suite with Google Benchmark + Python harness
- Performance regression testing with CI-gated detection and statistical significance
- NVTX profiling annotations and automated baseline comparison
- HTML performance dashboards with Plotly charts and color-coded status

**Added in v1.8:**
- Error message framework with std::error_code integration
- CMake package export with find_package(nova) support
- IDE configuration for clangd and VS Code
- CMakePresets.json with dev/release/ci build presets

## Constraints

- **Tech stack:** C++23, CUDA 20, CMake 4.0+ — current versions
- **Backward compatibility:** Existing API must not break
- **Testing:** All existing tests must pass after upgrade
- **Performance:** New implementations must not regress existing algorithms

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Foundation-first phasing | Quality foundations enable reliable feature work | ✓ v1.0 shipped |
| Streams for async | Native CUDA streams, not abstraction layer | ✓ Implemented |
| FFTW-style API | Familiar interface for signal processing users | ✓ Implemented |
| BVH helpers over full ray tracer | Focus on GPU compute primitives | ✓ Implemented |
| P2P ring-allreduce fallback | No NCCL dependency for v1.1 | ✓ Implemented |
| Row-wise split matmul | Simple, builds on existing infrastructure | ✓ Implemented |
| Device mesh singleton | Lazy initialization, single source of truth | ✓ Implemented |
| C++23 adoption | std::expected, constexpr, ranges for modern patterns | ✓ v1.2 shipped |
| CUDA 20 standard | Next-generation CUDA toolkit for new features | ✓ v1.2 shipped |
| CMake 4.0+ | Modern CMake features and policy support | ✓ v1.2 shipped |
| Optional NCCL with P2P fallback | Preserve single-node without NCCL | ✓ v1.3 shipped |
| TensorParallelMatmul (col/row) | Build on existing DistributedMatmul | ✓ v1.3 shipped |
| 1F1B pipeline scheduler | Classic GPipe-style scheduling | ✓ v1.3 shipped |
| MPI for multi-node init | Standard for cluster NCCL bootstrapping | ✓ v1.4 shipped |
| Checkpoint granularity | Full state (weights + optimizer + RNG) | ✓ v1.5 shipped |
| Error recovery strategy | Detect → isolate → recover → retry | ✓ v1.5 shipped |
| Signal handling | SIGTERM/SIGUSR1 for graceful shutdown | ✓ v1.5 shipped |
| Thread-safety | Mutex protection for signal state | ✓ v1.5 shipped |
| BatchNorm strategy | SyncBatchNorm with NCCL all-reduce | ✓ v1.6 shipped |
| Profiling approach | CUDA events for kernel timing | ✓ v1.6 shipped |
| Benchmark framework | Google Benchmark + Python harness hybrid | ✓ v1.7 shipped |
| Regression strategy | CI-gated threshold comparison against baseline | ✓ v1.7 shipped |
| Dashboard approach | HTML reports with trend charts, JSON data export | ✓ v1.7 shipped |
| Statistical testing | Welch's t-test for significance (scipy) | ✓ v1.7 shipped |
| NVTX approach | Header-only with compile-time toggle | ✓ v1.7 shipped |
| Error framework | std::error_code categories with recovery hints | ✓ v1.8 shipped |
| CMake package | Config-file packages with exported targets | ✓ v1.8 shipped |
| IDE support | .clangd, VS Code settings, compile_commands | ✓ v1.8 shipped |
| Doxygen docs | HTML output with group definitions | ✓ v1.9 shipped |
| Tutorial guides | Markdown docs covering quick start through profiling | ✓ v1.9 shipped |
| Example programs | Runnable examples in examples/ directory | ✓ v1.9 shipped |
| Fuzz testing | libFuzzer-based property fuzzing | ✓ v2.0 shipped |
| Property tests | QuickCheck-style tests with reproducible seeds | ✓ v2.0 shipped |
| Coverage reports | lcov/genhtml with per-module breakdown | ✓ v2.0 shipped |
| CI integration | GitHub Actions with coverage gates | ✓ v2.0 shipped |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state (users, feedback, metrics)

---
*Last updated: 2026-04-26 after v2.0 Testing & Quality complete*
*Last updated: 2026-04-26 for v2.1 New Algorithms*
