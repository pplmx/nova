# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Current Milestone: v2.10 Sparse Solver Acceleration

**Goal:** Accelerate Krylov solver convergence with preconditioners and sparse matrix ordering

**Target features:**
- Jacobi preconditioner (diagonal scaling)
- RCM sparse matrix ordering (reduce fill-in)
- ILU(0)/ILU(k) incomplete factorization
- Integration with existing CG/GMRESGPU/BiCGSTAB solvers

**Status:** Ready to start (Phase 88+)

## Completed (v2.6)

- ✓ FlashAttention-2 integration with IO-aware tiling — Phase 69
- ✓ Paged KV cache with block allocation and LRU eviction — Phase 70
- ✓ Block manager with CPU-GPU sync — Phase 71
- ✓ Sequence manager with continuous batching — Phase 72
- ✓ Sequence parallelism across TP ranks — Phase 73
- ✓ Integration tests with NVTX annotations — Phase 74

## Completed (v2.7)

- ✓ Timeline visualization with Chrome trace export — Phase 75
- ✓ Memory bandwidth measurement (H2D/D2H/D2D) — Phase 75
- ✓ Kernel statistics collection — Phase 75
- ✓ Occupancy analyzer with recommendations — Phase 75
- ✓ Segmented sort — Phase 76
- ✓ SpMV using CSR/CSC formats — Phase 76
- ✓ Sample sort for large datasets — Phase 76
- ✓ Delta-stepping SSSP — Phase 76
- ✓ Memory safety validation framework — Phase 77
- ✓ Test isolation framework — Phase 77
- ✓ Layer-aware error injection — Phase 77
- ✓ Boundary condition tests — Phase 77
- ✓ FP determinism control — Phase 77
- ✓ E2E robustness + profiling — Phase 78
- ✓ Memory safety validation — Phase 78
- ✓ Performance baselines — Phase 78
- ✓ Documentation updates — Phase 78

### Completed (v2.8)

- ✓ Sparse matrix ELL format with row-wise padding — Phase 79
- ✓ Sparse matrix SELL format with configurable slice height — Phase 79
- ✓ CSR to ELL/SELL conversion with automatic padding — Phase 79
- ✓ ELL/SELL SpMV kernels matching CSR baseline — Phase 79
- ✓ Conjugate Gradient (CG) solver for SPD systems — Phase 80
- ✓ Generalized Minimal Residual (GMRES) solver — Phase 80
- ✓ Biconjugate Gradient Stabilized (BiCGSTAB) solver — Phase 80
- ✓ Convergence criteria configuration — Phase 80
- ✓ Device peak FLOP/s queries (FP64/FP32/FP16) — Phase 80
- ✓ Memory bandwidth measurement — Phase 80
- ✓ Arithmetic intensity calculation — Phase 80
- ✓ HYB (Hybrid ELL+COO) format for irregular matrices — Phase 81
- ✓ Performance classification with confidence levels — Phase 81
- ✓ JSON export for Roofline analysis — Phase 81
- ✓ Solver workspace reuse via memory pool — Phase 82
- ✓ Solver diagnostics (timing, convergence rate) — Phase 82
- ✓ E2E integration tests — Phase 82
- ✓ Performance benchmarks — Phase 82
- ✓ NVTX integration with nova_sparse domain — Phase 82
- ✓ Documentation updates — Phase 82

### Completed (v2.9)

- ✓ SparseMatrix<T> with cuda::memory::Buffer<T> — Phase 83
- ✓ cuSPARSE integration for GPU SpMV — Phase 84
- ✓ GPU-accelerated CG solver — Phase 85
- ✓ GPU-accelerated GMRESGPU solver — Phase 85
- ✓ GPU-accelerated BiCGSTAB solver — Phase 85
- ✓ SparseMatrixCSR<T> deprecated with [[deprecated]] — Phase 86
- ✓ ToSparseMatrix() conversion function — Phase 86
- ✓ Comprehensive E2E integration tests — Phase 87

**Next milestone:** TBD

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
- ✓ GPU radix sort (ascending/descending) — v2.3
- ✓ Top-K selection without full sort — v2.3
- ✓ Binary search with warp shuffle — v2.3
- ✓ SVD (full/thin modes) — v2.3
- ✓ Eigenvalue decomposition — v2.3
- ✓ QR, Cholesky factorization — v2.3
- ✓ Monte Carlo with variance reduction — v2.3
- ✓ Numerical integration (trapezoidal/Simpson) — v2.3
- ✓ Root finding (bisection/Newton-Raphson) — v2.3
- ✓ Interpolation (linear/cubic spline) — v2.3
- ✓ FFT convolution — v2.3
- ✓ Haar wavelet transform — v2.3
- ✓ FIR filters — v2.3

### Planned (v2.4)

- [ ] CUDA Graphs for batch workload optimization (10-50x launch overhead reduction)
- [ ] L2 cache persistence for working set optimization
- [ ] Priority stream pool for latency-sensitive operations
- [ ] NVBench GPU microbenchmarking integration
- [ ] Async error tracking and propagation
- [ ] NVTX domain extensions for observability
- [ ] Error injection framework for chaos testing
- [ ] Memory pressure and concurrent stream stress tests

### Completed (v2.4)

- ✓ CUDA Graphs with GraphExecutor and capture/replay — v2.4
- ✓ Memory nodes for device/host/managed memory — v2.4
- ✓ Algorithm wrappers for reduce/scan/sort — v2.4
- ✓ L2 cache persistence with RAII control — v2.4
- ✓ Priority stream pool (low/normal/high) — v2.4
- ✓ NVBench integration headers — v2.4
- ✓ NVTX domain extensions per layer — v2.4
- ✓ Async error tracker for deferred errors — v2.4
- ✓ Health metrics dashboard (JSON/CSV) — v2.4
- ✓ Error injection framework for chaos testing — v2.4
- ✓ Memory pressure stress tests — v2.4
- ✓ Concurrent stream stress tests — v2.4
- ✓ Production hardening guide (PRODUCTION.md) — v2.4

### Completed (v2.5)

- ✓ Per-operation timeout tracking with configurable deadlines — v2.5
- ✓ Watchdog timer system for detecting stalled operations — v2.5
- ✓ Deadline propagation across async operation chains — v2.5
- ✓ Timeout callback/notification system — v2.5
- ✓ Exponential backoff with configurable base delay — v2.5
- ✓ Jitter implementation (full/decorrelated) — v2.5
- ✓ Circuit breaker pattern with threshold configuration — v2.5
- ✓ Retry policy composition and chaining — v2.5
- ✓ Reduced precision mode (FP64→FP32→FP16 fallback) — v2.5
- ✓ Fallback algorithm registry with priority ordering — v2.5
- ✓ Quality-aware degradation with threshold configuration — v2.5
- ✓ Degradation event logging and metrics — v2.5

### Completed (v2.6)

- ✓ Attention backend enum (Standard/FlashAttention/PagedAttention) — Phase 69
- ✓ FlashAttention forward with IO-aware tiling — Phase 69
- ✓ Stable softmax with max subtraction — Phase 69
- ✓ Backward pass with deterministic dropout — Phase 69
- ✓ Dynamic workspace allocation — Phase 69
- ✓ Block-based KV cache allocator (16/32/64 tokens) — Phase 70
- ✓ LRU eviction on memory pressure — Phase 70
- ✓ Prefix caching with hash lookup — Phase 70
- ✓ KVCacheStats tracking — Phase 70
- ✓ BlockManager with block table mapping — Phase 71
- ✓ append_tokens with atomic block allocation — Phase 71
- ✓ CPU-GPU block table synchronization — Phase 71
- ✓ Paged attention forward with bounds validation — Phase 71
- ✓ SequenceManager for lifecycle management — Phase 72
- ✓ Continuous batching with iteration-level scheduling — Phase 72
- ✓ GQA/MQA support — Phase 72
- ✓ SequenceParallelAttention with TP integration — Phase 73
- ✓ Ring sequence parallelism for long sequences — Phase 73
- ✓ InferenceGraphExecutor with CUDA Graph capture — Phase 74
- ✓ NVTX domain for inference phases — Phase 74
- ✓ Integration tests covering all 18 requirements — Phase 74

### Completed (v2.9)

- ✓ SparseMatrix<T> with cuda::memory::Buffer<T> — Phase 83
- ✓ cuSPARSE integration for GPU SpMV — Phase 84
- ✓ GPU-accelerated ConjugateGradient solver — Phase 85
- ✓ GPU-accelerated GMRESGPU solver — Phase 85
- ✓ GPU-accelerated BiCGSTAB solver — Phase 85
- ✓ SparseMatrixCSR<T> deprecated with [[deprecated]] — Phase 86
- ✓ ToSparseMatrix() conversion from legacy CSR — Phase 86

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

**Added in v2.3:**
- GPU sorting algorithms (CUB-based radix sort, top-K, binary search)
- Linear algebra extras (cuSOLVER SVD, EVD, QR, Cholesky)
- Numerical methods (Monte Carlo, integration, root finding, interpolation)
- Signal processing (FFT convolution, Haar wavelet, FIR filters)

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
| CUB for sorting | Production-quality GPU sorting primitives | ✓ v2.3 shipped |
| cuSOLVER for linalg | SVD, EVD, factorization via NVIDIA library | ✓ v2.3 shipped |
| std::rand for Monte Carlo | Simple PRNG for numerical methods | ✓ v2.3 shipped |
| cuFFT for signal | Fast convolution via FFT | ✓ v2.3 shipped |

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
*Last updated: 2026-05-01 after v2.10 Sparse Solver Acceleration started*
*v2.10: Preconditioners, Matrix Ordering, Solver Acceleration*
