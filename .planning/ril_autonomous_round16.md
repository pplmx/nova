# RIL — Round 16 (2026-08-11) — milestone v2.16 "Distributed Multi-GPU Verification"

## Round focus

Phase 3 of milestone v2.16: **make the distributed matmul multi-GPU path a real
test**. After Round 15 enabled the NCCL collective suite (14/14 on 2 GPUs), the
last unverified distributed functionality was `DistributedMatmul`'s row-split +
all-gather path — which was **not implemented at all**: `matmul_async()` always
delegated to `matmul_single_gpu` regardless of device count, and the
`MultiGpu_RequiresMultiProcess` test was an unconditional, documented skip.

### Observation (evidence)

- `src/cuda/distributed/matmul.cu` `matmul_async()`: unconditional
  `matmul_single_gpu(A, B, C, m, n, k, options)` — the multi-GPU
  row-partition + all-gather behavior documented in `matmul.h` had no
  implementation to test.
- `tests/distributed/distributed_matmul_test.cu` `MultiGpu_RequiresMultiProcess`:
  `GTEST_SKIP()` in all cases, deferring to "multi-process execution (e.g., NCCL)".
- Decision `decision-dist-matmul-multiprocess` (R15) recorded that multi-GPU
  matmul "intentionally" requires a multi-process execution model and to
  "convert later with a process-per-rank harness".

### Root cause

The "multi-process required" assumption is **false**: Round 15 proved that
genuine multi-device NCCL collectives run correctly **thread-per-rank** in a
single process — one thread per device pinned with `cudaSetDevice`, each
posting its rank's collective on its own communicator — for the whole NCCL
collective suite (all-reduce, broadcast, all-gather, reduce-scatter, barrier,
groups), and cpp-reviewer validated that concurrency design. The same pattern
applies to distributed matmul: each rank thread computes its row partition of
C locally (cuBLAS), then an NCCL all-gather reconstructs the full result on
every device. No OS-level process spawning (fork/exec, `ncclGetUniqueId`
bootstrap) is needed; thread-per-rank avoids CUDA-fork hazards entirely.

### Fix (this round)

- **`matmul.h`**: declared `DistributedMatmul::matmul_multi_gpu(A, B, C, m, n, k,
  options)` returning `cuda::nccl::NcclResult` (never throws — matches the NCCL
  collective contract so a failing rank surfaces safely from a spawned thread).
- **`matmul.cu`**: implemented `matmul_multi_gpu`:
  - Row partition `rows_per_gpu = ceil(m / n_dev)`; rank r owns rows
    `[r*rows_per_gpu, min((r+1)*rows_per_gpu, m))`. Each rank sends exactly
    `rows_per_gpu*n` elements, so `ncclAllGather`'s equal-send-count
    requirement holds (padding past `local_m` is carried but ignored — the
    caller reads only the first `m*n` of the gathered buffer).
  - Seeds the local dense block with the rank's rows of the caller's C so the
    `beta*C` accumulation term matches single-GPU semantics.
  - NCCL all-gather via `NcclAllGather` (routes through the current device's
    communicator/stream, per R15's `current_comm()` design).
  - Rejects transposed operands explicitly (they don't compose with row slicing)
    rather than emit a silently wrong result. `n_dev <= 1` falls back to
    `matmul_single_gpu`, preserving `needs_multi_gpu()==false` semantics.
  - `get_cublas_handle_for_device` made thread-safe (mutex + double-checked
    append): the multi-GPU path now calls it concurrently from one thread per
    rank, and the old unlocked static vector was a data race.
- **cpp-review pass (before commit)** exposed two HIGH items, both fixed:
  - *Empty/negative last slice*: `m < n_dev` alone allowed `m=5, n_dev=4` →
    `local_m = -1` → `size_t` underflow in the seed copy + OOB `A` offset.
    Guard now requires `(n_dev-1)*rows_per_gpu < m` so every rank owns >= 1 row;
    rejected as `ncclInvalidArgument` **before** the collective. Regression test
    `MultiGpu_RejectsBadShapes` added.
  - *cuBLAS handle keyed on rank instead of device*: `get_cublas_handle_for_device`
    is keyed on the CUDA device index and binds the handle to that device's
    context; passing `rank` was correct only for default `{0..n-1}` groups. Now
    keyed on the current device, consistent with `rank_of_device()` handling of
    custom groups.
  - (The pre-collective early-return poisoning of the shared NcclContext is
    documented in the header contract + `.cu` comments; a collective
    readiness/fault exchange is tracked as a task in milestone v2.17.)
- **`distributed_matmul_test.cu`**: added a `run_per_rank` helper (same pattern
  as the NCCL suite — RankBarrier + one thread per device + first-exception
  rethrow after all joins). Replaced the unconditional skip with three real
  multi-GPU tests that skip only on genuine env conditions (<2 GPUs, or NCCL
  not available / not initialized):
  - `MultiGpu_RowSplitAllGather` — random A/B, `m = 6*device_count - 1`
    (never divisible by device_count, so the last rank gets a short partition
    and the padded all-gather path is exercised), asserts every rank's gathered
    C == single-GPU reference.
  - `MultiGpu_AlphaBeta` — same with `alpha=2, beta=0.5`, verifying the seeded
    beta-accumulation path.
  - `MultiGpu_RejectsBadShapes` — regression: `m smaller than the group, and
    transposed operands, return `ncclInvalidArgument` before the collective on
    every rank (guards are rank-independent, so no rank hangs).

### Verification

- `CUDA_VISIBLE_DEVICES=0,1 NCCL_TESTS_AVAILABLE=1 DistributedMatmulTest.*` →
  **14/14 pass** (11 single-GPU/edge + 3 real multi-GPU), run 5+ times stable.
  `MultiGpu_RowSplitAllGather`, `MultiGpu_AlphaBeta`, `MultiGpu_RejectsBadShapes`,
  `MultiGpu_RowPartition` all green on 2 GPUs.
- NCCL collective suite unchanged: `NcclCollectivesTest.*` **14/14 pass** on 2
  GPUs (no regression from touching `nccl_context` include paths / matmul).
- Phase 2 confirmation: `DistributedMemoryPoolTest.{AutoAllocateSelectsDevice,
  DeallocateFromCorrectDevice}` pass on 2 GPUs.
- Without `NCCL_TESTS_AVAILABLE` (single-GPU env): the new multi-GPU tests
  **skip cleanly** (genuine env gate) — no unconditional skips remain in the
  matmul suite. Distributed/mesh subset 61 pass on one GPU.
- Full-suite regression (CUDA_VISIBLE_DEVICES=2): **1451 ran / 1423 pass /
  0 fail / 28 skip / 1 disabled, EXIT=0** (the +2 over the R15 baseline of 1449
  is the net of 3 real multi-GPU tests replacing the 1 unconditional skip; the
  guard regression test env-skips on single-GPU). No regressions.

### Known skips remaining (verified reasons)

- 14 NCCL tests: skip without `NCCL_TESTS_AVAILABLE` (real when set).
- 8 MPI tests: build-flag off; no MPI headers on host (env-bound).
- 2 distributed pool tests: skip on <2 GPUs (env-bound), verified on 2 GPUs.
- 1 PeerCopy test: needs 2 GPUs (env-bound).
- 3 distributed matmul multi-GPU tests (`MultiGpu_RowSplitAllGather`,
  `MultiGpu_AlphaBeta`, `MultiGpu_RejectsBadShapes`): skip without NCCL/2 GPUs
  (env-bound) — now real when the env provides them.

## Graph delta
nodes:
  - id: change-r16-matmul-multigpu
    type: change
    commit: a1f438d
  - id: task-dist-matmul-multiprocess
    type: task
    status: active → resolved
  - id: issue-dist-matmul-multigpu-unverified
    type: issue
    status: active → resolved
  - id: decision-dist-matmul-multiprocess
    type: decision
    status: active → superseded (by decision-matmul-thread-per-rank)
  - id: decision-matmul-thread-per-rank
    type: decision
  - id: ev-r16-matmul-mgpu
    type: evidence
    confidence: 1.0
  - id: ev-r16-fullsuite-baseline
    type: evidence
    confidence: 1.0
edges:
  - from: change-r16-matmul-multigpu
    type: implements
    to: task-dist-matmul-multiprocess
  - from: change-r16-matmul-multigpu
    type: resolves
    to: issue-dist-matmul-multigpu-unverified
  - from: decision-matmul-thread-per-rank
    type: supersedes
    to: decision-dist-matmul-multiprocess
  - from: decision-matmul-thread-per-rank
    type: governs
    to: task-dist-matmul-multiprocess
