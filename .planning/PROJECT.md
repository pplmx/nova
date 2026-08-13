# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Current Milestone: v2.25 MicroTrainer: End-to-End Gradient Training Convergence On The Tensor-Parallel Stack

**Status:** In Progress (opened 2026-08-13, RIL Round 25 — P2 implemented)

**Milestone v2.25.** Turns the v2.24 verified building blocks into a reusable training loop
and proves the parallel stack actually *learns* (TASK-027 / DEC-011). P2 (CHG-014 cc35094,
TASK-029) added `training::MicroTrainer` (`training.h/.cpp`): owns the TensorParallelMLP,
one AdamW per weight tensor (gate/up/down — moments stay private per weight), and the device
scratch; `train_step` chains forward → device `cross_entropy_logits_backward` → MLP backward
→ per-shard AdamW and returns the mean CE loss (host scalar); `evaluate` adds top-1 accuracy.

**Bug found & fixed** (`issue-v24-ce-loss-running-sum`): `cross_entropy_loss` filled
`log_probs` while accumulating `sum_exp`, so the softmax denominator for a target at class c
used the partial running sum over c' ≤ c — under-reporting loss for early targets (dev 2.18
vs host fp64 3.91 on identical logits; exposed by the trajectory parity; masked in v2.24 by
relative-only assertions and correct-gradient device CE-backward). Full-class sum now first.

Verified (EV-019): single-GPU MicroTrainer converges (loss descends 60 steps, accuracy rises
above chance 1/8); multi-GPU K=10 sharded MicroTrainer == host fp64 full-weight trajectory
(assembled shards ≤ 2e-2 AND loss curve ≤ 0.1) **GREEN on 2 & 4 GPUs**; cross-suite
**40/40 on 2 GPUs**; single-GPU neural regression **45/45**. The stack now demonstrably
learns, sharded. Host note: use `CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.24 End-to-End Training Step On The Tensor-Parallel Stack

**Status:** Complete (2026-08-13, RIL Round 24)

**Milestone v2.24.** Makes the v2.21-23 forward+backward layers actually *trainable* on the
parallel path (TASK-023 / DEC-010). The stack computed gradients but nothing could train it:
`loss_functions.cu` was host-side with a forward-only scalar (its three `__global__` kernels
were dead code — no `dlogits` seed existed), the `weight_` shards were private with no
optimizer-step surface, and no loop chained forward → loss → backward → optimizer. P1
(TASK-024) added the contracts and pinned them RED (EV-017). P2 (CHG-013, TASK-025)
implemented: device `cross_entropy_logits_backward` kernel (`dlogits = (softmax - onehot)/B`,
max-subtract softmax recomputed per thread); per-layer `step(AdamWOptimizer&,
grad_weight_full, step_no)` extracting the rank shard grad (strided column slice via
`cudaMemcpy2DAsync` / contiguous row block) and applying AdamW in place to the private
`weight_` shard (per-rank optimizer instance = no cross-rank state); `TensorParallelMLP::step`
chains gate/up/down; read-back via `copy_weight_shard`/`copy_weights`.

Key insight: the meshed gradient layout is exact (col grad-shard = column block of
`X^T dY_full`; row grad-shard = row block) and AdamW is per-element — so stepping each rank
shard tiles to the full-weight update, giving a clean K-step parity. Verified (EV-018):
`MultiGpu_TrainingStep_MatchesHostFullWeight` (K=3 AdamW steps per rank, shards assembled ==
host fp64 full-weight reference) GREEN on **2 & 4 GPUs**; single-GPU CE-logits backward +
col/row/MLP step vs host fp64 4/4; isolated NCCL cross-suite **39/39 on 2 & 4 GPUs**;
single-GPU neural regression 37/37; full-suite baseline **1426/0**. The parallel stack is now
trainable end-to-end. Host note: use `CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.23 Tensor-Parallel Layer Backward Passes

**Status:** Complete (2026-08-12, RIL Round 23)

**Milestone v2.23.** Adds backward (gradient) passes to the v2.21 weight-managed layers
(TASK-022), closing the gap that kept the parallel stack inference-only. P1 shipped 6
analytic reference-parity RED tests. P2 (7855ab3) implemented `backward()`: `dW = X^T dY`,
`dX = dY W^T` (realized via new transpose/elementwise_add/silu_and_mul_backward kernels
onto the verified non-transposed matmul convention; per-instance stream-bound handle);
the column layer's grad-input is the **AllReduce** of the per-rank `dY W^T` — the new
never-run collective, `NcclResult` checked (replicated input ⇒ summed gradient); the row
layer's grad-input is local (matches the sharded-input layout); the gated MLP chains
down→silu_and_mul→gate/up with the grad-inputs summed and AllReduced; grad weights
written as col/row shard slices into the caller's full buffers.

Verified on real multi-GPU: single-GPU backward 3/3 + activation helpers 3/3; multi-GPU
backward 3/3 on 2 & 4 GPUs; isolated distributed cross-suite 19/19 on 2 & 4 GPUs;
single-GPU regression 49/49; full-suite baseline 1422/0. (A test-side strided-block
extraction bug in the row grad-input reference was fixed — the implementation was
correct.) Decision: DEC-009. The stack is now forward+backward capable; a
loss→backward→optimizer end-to-end training integration is the natural next milestone.
Host note: use `CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.22 Ring Sequence Parallelism On Real Multi-GPU

**Status:** Complete (2026-08-12, RIL Round 22)

**Milestone v2.22.** Implements `RingSequenceParallelism::ring_attention` for real
(TASK-014 / issue-v19-ring-parallel-noop), replacing the DEC-004 fail-fast disposition.
P1 shipped `MultiGpu_RingAttention_MatchesFullSequenceReference` RED (the old path
threw "send_recv_kv undefined"). P2 (TASK-016) implemented the ring: each of the P-1
steps sends the local KV block clockwise to next_rank while receiving prev_rank's block
(NCCL P2P send/recv, wrapped in `ncclGroupStart/End` — **ungrouped P2P deadlocked the
ring, a real finding**, EV-012), accumulating every block with online-softmax kernels
(`src/cuda/distributed/ring_attention.cu`). Per-rank output now equals standard scaled
dot-product attention over the FULL KV sequence across all ranks — GREEN on 2 & 4 GPUs
(EV-012), isolated cross-suite 16/16 on both, single-GPU seq-parallel 33/33, full-suite
baseline 1416/0 (EV-013). The obsolete NotImplemented pin was replaced by
`RejectsMissingHiddenDim` (config.hidden_dim is now required by the multi-GPU ring
path). Decisions: DEC-008. Host note: use `CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.21 Weight-Managed TensorParallelLayers

**Status:** Complete (2026-08-12, RIL Round 21)

**Milestone v2.21.** Implemented the previously-disposed (DEC-006) neural parallel-training
layers for real with **weight-sharded Megatron semantics** (TASK-010 /
issue-v20-tp-layers-stubs). P1 wrote reference-parity tests against the v2.20 stubs (RED:
every functional forward threw). P2 replaced the stubs with weight-managed layers: each
layer owns its rank's shard of the weight on device; `set_weight()` uploads the full
weight and slices the rank shard via `NcclContext::rank_of_device` (the v2.20 verified
membership convention — raw device indices stay wrong for non-default groups); a
per-instance stream-bound cuBLAS handle keeps the thread-per-rank GEMMs safe (the shared
handle is not thread-safe); `ColumnParallelLayer::forward` is a local shard matmul
(sharded output, no comm), `RowParallelLayer::forward` adds one block-wise AllReduce
(replicated output, `NcclResult` checked), `TensorParallelMLP` composes gate/up
column-parallel projections + the new `silu_and_mul` gated activation + a row-parallel
down projection; non-divisible in/out feature dims reject at construction (the v2.19
finding that the pre-v2.20 shard math was broken is now pinned by parity tests). Green on
real multi-GPU (2 & 4 GPUs, `CUDA_VISIBLE_DEVICES=2,3+`): column shard concat ==
single-GPU full-weight reference, row AllReduce replicated == reference, gated MLP ==
single-GPU full-weight reference — 15/15 isolated NCCL cross-suite, TP+layer+sequence
multi-GPU green (EV-010/EV-011). Single-GPU (tp<=1) path 6/6; full-suite baseline EXIT=0.
Note: the NCCL multi-GPU suites still hang when interleaved with `cudaDeviceReset` suites
in one process (pre-existing issue-v19-shared-nccl-context-reset — run the cross-suite via
the isolated curated filter). Decision: DEC-007 (milestone direction + new explicit
in/out-features API + diff-size exception). API change is safe: zero callers in the tree.
Host note: GPUs 0/1 externally loaded — use `CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.20 TensorParallelMatmul Production Hardening

**Status:** Complete (2026-08-12, RIL Round 20)

**Milestone v2.20.** Closed the cpp-review BLOCK on the v2.19 TP rewrite and disposed the
last non-functional neural parallel-training surface. P1 hardened `TensorParallelMatmul`
onto the verified NCCL conventions: rank is resolved through `ctx_.rank_of_device()`
(validated membership — no more raw-device-index rank, which breaks on non-default NCCL
groups and could overrun buffers), the column AllReduce's `NcclResult` is now checked and
failures throw `NcclException` (no more silent-garbage on a dead comm), plus a RowParallel
non-divisible test and corrected header docs — 20/20 TP+NCCL suites, acceptance 5/5 at
4 GPUs (2137cea, EV-009). P2 disposed the `TensorParallelLayers` skeletons
(ColumnParallelLayer / RowParallelLayer / TensorParallelMLP) which were never implemented
— their `forward()` passed `B = nullptr` into cuBLAS and the MLP forward was empty; they
now fail fast with explicit errors and a real weight-managed implementation is tracked as
TASK-010 (DEC-006). `issue-v19-shared-nccl-context-reset` investigated (driver-level
context-loss detection is infeasible — `cuCtxGetCurrent` returns the same address after
reset, EV-008); disposition is the documented curated multi-GPU workflow + standard
baseline without `NCCL_TESTS_AVAILABLE`. Decisions: DEC-005 (milestone direction),
DEC-006 (layers disposition). Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.19 Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism)

**Status:** Complete (2026-08-12, RIL Round 19)

**Milestone v2.19.** Extended the R15-R18 "never-run multi-GPU surface hides real bugs"
thread to the neural parallel-training layer. P1 proved `TensorParallelMatmul`'s
multi-GPU paths broken — ColumnParallel sliced B/C as contiguous (column-major) offsets
against the row-major matmul and AllReduce-summed disjoint output slices, RowParallel
ignored the rank index, and non-divisible `n` was silently dropped (4/4 RED, HYP-002
validated, c5f302a→2006da8). P2 rewrote the partition math on the verified NCCL layer:
zero-padded strided column blocks + a single block-wise AllReduce (honoring the documented
"compute A @ B_part then AllReduce, output identical" contract), rank-correct RowParallel,
explicit shape rejection, and per-instance stream-bound cuBLAS handles (the shared handle
is not thread-safe) — 8/8 acceptance GREEN on 2, 4, and 8 GPUs; TP+NCCL suites 19/19
(EV-005 refutes HYP-002). P3 drove `SequenceParallelAttention` (gather_kv / scatter_output
/ all_reduce_sequence) with real `NcclContext::current_comm()` on each rank — 4/4 GREEN on
2/4 GPUs, verifying the collectives were actually correct — and made
`RingSequenceParallelism`'s multi-GPU no-op fail fast (DEC-004). Decisions: DEC-003
(milestone direction), DEC-004 (ring no-op disposition). Host note: GPUs 0/1 are under
external load — use `CUDA_VISIBLE_DEVICES=2,3+`.

## Previous Milestone: v2.18 MeshBarrier On The Verified Layer + Distributed Robustness

**Status:** Complete (2026-08-12, RIL Round 18)

**Milestone v2.18.** Closed out the distributed-ops convergence thread: `MeshBarrier` — the
last high-level distributed op still on the legacy per-instance host event-poll — was proven
to have no cross-rank arrival semantics (per-rank rendezvous tests RED against it,
HYP-001/EV-001) and converged onto the verified `NcclBarrier` layer (5/5 multi-GPU barrier
tests green; CHG-002 0a12a84). Then hardened the collective harness: a bounded 120s
thread-per-rank barrier turns a dead rank into a diagnosed `RankBarrierTimeout` (previously
an unkillable >120s hang), and `NcclContext` now learns when the error layer aborts a
communicator (`mark_comm_aborted`) so dead comms fail fast and recover via destroy+reinit
instead of poisoning later tests (CHG-003 7914b34) — closing the R16 review HIGH-B and
`issue-v17-dist-ops-harness-flake` (DEC-002). Full suite 1461/1423/0; 2-GPU cross-suite
49/49; stress stable. Decisions: `DEC-001` (v2.18), `DEC-002` (flake disposition).

## Previous Milestones: v2.16/v2.17 Distributed Multi-GPU

**v2.17 "Distributed Ops On Real Multi-GPU" (Complete, Round 17).** The high-level distributed
API's multi-GPU paths were never exercised; proven broken (P1), converged onto the verified
NCCL layer (P2, 4/4 green), and the SyncBatchNorm multi-GPU gradient path
verified/corrected vs a global-batch host reference (P3), exposing a latent single-GPU mean
double-centering bug. Full suite 1456/1423/0. Decision `decision-v17-dist-ops-real-multigpu`.

## Completed (v2.16 Distributed Multi-GPU Verification)

**Goal:** Turn the last un-verified distributed functionality into running, asserted tests - the
RIL skip-audit showed every skipped test block so far hid at least one real bug, and the NCCL
multi-GPU suite was the largest remaining never-run block.

**Delivered:**

- NCCL multi-GPU collectives (all-reduce, broadcast, barrier, all-gather, reduce-scatter, groups)
  run and pass for real on multi-GPU (Round 15: 14/14 green, fix bfb9e80)
- Distributed memory pool / mesh multi-GPU paths verified on 2 GPUs
- `DistributedMatmul::matmul_multi_gpu` — the real row-split + NCCL all-gather path — implemented
  and tested thread-per-rank (Round 16), replacing the last unconditional distributed skip.
  Supersedes the "multi-process required" assumption (`decision-matmul-thread-per-rank`).
- Full-suite baseline at close: 1451 ran / 1423 pass / 0 fail / EXIT=0.

## Completed (v2.14 Documentation Quality)

**Goal:** Comprehensive documentation sweep

**Shipped:**

- API Documentation: 30 headers documented with Doxygen comments — Phase 107
- Code Comments: 5 source files commented (memory, device, algo layers) — Phase 108
- Error/Log Messages: Structured logging infrastructure (5 levels) — Phase 109
- README/docs: README.md, CHANGELOG.md, docs/ARCHITECTURE.md, SPARSE.md, QUANTIZATION.md — Phase 110

## Previous Milestone: v2.13 Transformer Optimization (SHIPPED 2026-05-06)

**Goal:** FP8 support, CUDA kernels, production calibration, and QAT integration

**Status:** Complete — archived at `.planning/milestones/v2.13-*`

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

### Completed (v2.10)

- ✓ Preconditioner base interface — Phase 88
- ✓ JacobiPreconditioner with weighted diagonal scaling — Phase 88
- ✓ Unit tests for Jacobi preconditioner — Phase 88
- ✓ RCM (Reverse Cuthill-McKee) matrix ordering — Phase 89
- ✓ Unit tests for RCM ordering — Phase 89
- ✓ ILU(0) preconditioner via cuSPARSE — Phase 90
- ✓ Unit tests for ILU preconditioner — Phase 90
- ✓ CG solver with preconditioner support — Phase 91
- ✓ E2E convergence tests — Phase 91
- ✓ Performance benchmarks — Phase 92

### Completed (v2.11)

- ✓ NVBlox metrics integration header — Phase 93
- ✓ Kernel-level profiling hooks — Phase 93
- ✓ Custom metric aggregators (AI, FLOP/s, BW) — Phase 93
- ✓ KernelFusionAnalyzer with 10+ patterns — Phase 94
- ✓ Fusion profitability model (100us threshold) — Phase 94
- ✓ Fusion recommendation engine with confidence — Phase 94
- ✓ RooflineModel with device peaks — Phase 95
- ✓ BandwidthUtilizationTracker — Phase 95
- ✓ CacheAnalyzer with CUPTI fallback — Phase 95
- ✓ DashboardExporter with JSON/CSV — Phase 96
- ✓ FlameGraphGenerator from NVTX — Phase 96
- ✓ NVTX performance domains — Phase 93/96
- ✓ Integration tests — Phase 97
- ✓ PERFORMANCE_TOOLING.md documentation — Phase 97

**Next milestone:** v2.12 Advanced Quantization

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

### Completed (v2.11)

- ✓ NVBlox integration for kernel-level analysis — Phase 93
- ✓ Kernel fusion analysis with opportunity detection — Phase 94
- ✓ Memory bandwidth optimization (roofline, utilization) — Phase 95
- ✓ Dashboard & visualization with fusion impact — Phase 96
- ✓ Integration tests and documentation — Phase 97

### Completed (v2.12)

- ✓ FP8 types (E4M3, E5M2) with IEEE 754-like semantics — Phase 98
- ✓ CUDA quantization kernels with vectorization — Phase 99
- ✓ Production calibration (histogram, minmax, MSE, per-channel) — Phase 100
- ✓ QAT patterns (FakeQuantize, STE gradients) and AMP manager — Phase 101
- ✓ Benchmark suite and accuracy comparison tools — Phase 102

### Completed (v2.14)

- ✓ API Documentation: Doxygen coverage for 30 headers — Phase 107
- ✓ Code Comments: Inline comments in 5 source files — Phase 108
- ✓ Error/Log Messages: Structured logging with 5 levels — Phase 109
- ✓ README/docs: User/developer documentation guides — Phase 110

### Validated (v2.14)

- ✓ Doxygen @brief, @param, @return documentation — v2.14
- ✓ @defgroup module organization (error, sparse, quantize, gnn) — v2.14
- ✓ @deprecated tags with migration guidance — v2.14
- ✓ Inline comments explaining algorithm rationale — v2.14
- ✓ Thread-safety documentation in concurrent code — v2.14
- ✓ Structured logging (ERROR/WARN/INFO/DEBUG/TRACE) — v2.14
- ✓ Compile-time log level filtering — v2.14
- ✓ Updated README.md with v2.x features — v2.14
- ✓ CHANGELOG.md with v1.0-v2.14 milestones — v2.14
- ✓ docs/ARCHITECTURE.md five-layer guide — v2.14
- ✓ docs/SPARSE.md operations guide — v2.14
- ✓ docs/QUANTIZATION.md INT8/FP8 guide — v2.14

---

*Last updated: 2026-05-09 after v2.15 Test Quality Assurance started*
*v2.15: In progress - fixing 89 failing tests*
