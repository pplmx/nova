# RIL — Round 19 (2026-08-12) — milestone v2.19 "Parallel Training On Real Multi-GPU"

## Round focus

**Open milestone v2.19** and execute **P1 (TASK-004)**: the neural parallel-training
layer is the largest remaining never-run multi-GPU surface. `TensorParallelMatmul`'s
multi-GPU execution paths (`matmul` / `matmul_async`, Column/RowParallel) had no real
multi-GPU test (only buffer-size formulas in tensor_parallel_test.cpp), and the
sequence-parallel tests pass `comm = nullptr` so they never drive a real NCCL
collective. Following the R15-R18 pattern — every never-run multi-GPU surface hides
real bugs — P1 builds real per-rank multi-GPU tests; P2 converges the TP matmul math;
P3 drives SequenceParallelAttention with real communicators.

## Observation (evidence / model)

- v2.18 closed with zero active tasks; distributed-ops convergence (v2.15→v2.18) done.
- Static read of `src/cuda/neural/tensor_parallel_matmul.cpp` showed strong bug smells:
  ColumnParallel slices B/C as contiguous blocks (column-major offsets) against the
  row-major `cuda::neural::matmul`, `recv_buffer` is dead code, every rank redundantly
  computes ALL slices with the full B, `local_n = n/tp` silently drops remainders,
  and RowParallel never reads the rank index.
- `RingSequenceParallelism::ring_attention`'s multi-GPU branch is an empty no-op;
  `send_recv_kv` declared but never defined.

## Model (graph delta this round)

- `DEC-003` (decision-v19-parallel-training) — milestone direction: P1 real multi-GPU
  TP tests → P2 converge TP matmul math on verified NCCL layer → P3 SequenceParallel
  real comm; governs `comp-neural-parallel`. Alternatives rejected: MPI (env-bound),
  inference-layer multi-GPU (depends on this), pure scale-up (no new verified surface).
- `HYP-002` — ColumnParallel contiguous-slice partition math yields wrong output on
  real multi-GPU (confidence 0.85).
- `TASK-004/005/006` — P1 tests / P2 fix / P3 sequence-parallel, plus `TASK-007`
  (ring-parallel no-op follow-up).

## Execute — Phase 1 (TASK-004)

Added `tests/neural/tensor_parallel_multigpu_test.cpp` (RED bug-finding tests, shipped
`DISABLED_MultiGpu_*`): per-rank `run_per_rank` ColumnParallel (sync + async),
non-divisible-n reject contract, and RowParallel, each asserted against a single-GPU
reference. Committed `2006da8` (`CHG-004`).

## Verify — Phase 1 (evidence)

2x A100: all 4 FAIL RED (`EV-004` validates `HYP-002`) — ColumnParallel sync+async
differ from the reference (rank 0 element 0: -0.82 vs -0.46), `n % tp != 0` silently
drops columns (no throw), RowParallel ignores the rank index (rank 0 block untouched).

## Execute — Phase 2 (TASK-005)

Rewrote `tensor_parallel_matmul.cpp` + header (`CHG-005`, pending):
- ColumnParallel: zero the full-width C, compute each rank's strided column block via
  cuBLAS with narrowed output width and full leading dims (`ldb = ldc = n`), then a
  single block-wise AllReduce sums the padded blocks into the full result — matching
  the documented "compute A @ B_part then AllReduce, output identical" contract with
  the row-major layout.
- RowParallel: each rank computes its contiguous row block `A[rank*local_m..] @ B` into
  `C[0, local_m*n)` (the rank index was previously ignored entirely).
- Non-divisible shapes throw `std::invalid_argument` (fail-fast, cf. MeshBarrier subset).
- Per-instance lazily-created cuBLAS handle bound to the caller's stream — the shared
  `get_cublas_handle()` is not thread-safe and cannot be stream-bound per rank.
  Class became non-copyable/non-movable (NcclAllReduce base is non-movable).
- `matmul()` sync path uses `cudaDeviceSynchronize()` (covers default-stream fallback).
- ColumnParallel without a live NCCL context now throws instead of returning a partial
  result.

## Verify — Phase 2

`MultiGpu_*` renamed acceptance tests GREEN **4/4 on 2, 4, and 8 GPUs**; TP + NCCL
suites together **19/19** on free GPUs; `EV-005` refutes `HYP-002`. SequenceParallel
real-comm tests (see P3) **4/4 GREEN on first try** — the sequence-parallel collectives
were actually correct, refuting the "always broken" generalization.

## Execute — Phase 3 (TASK-006)

Added `tests/distributed/sequence_parallel_multigpu_test.cu` (`CHG-006`, pending):
`SequenceParallelAttention` with a real `NcclContext::current_comm()` on each rank,
asserting all_reduce_sequence (global sum), gather_kv (rank-slot all-gather), and
scatter_output (ReduceScatter `sp × replicated block`; slice-copy variant). **4/4 GREEN
on 2 and 4 GPUs.**

## Environment discovery (important)

`/tmp/cores` showed the full (non-NCCL) suite SIGSEGVs in `cuStreamDestroy` (libcuda)
after long interleaving — the stream-destroy-after-`cudaDeviceReset` hazard flagged in
`decision-heap-singletons-exit`; unrelated suites call `cudaDeviceReset()` and a shared
singleton stream outlives the reset context. This is pre-existing and order-dependent
(also reproduced at the parent commit — see verification). Also: GPUs 0/1 are now under
external `/workspace/llm` load (~80 GB), so all multi-GPU runs must use
`CUDA_VISIBLE_DEVICES=2,3+` (memory saved separately).

## Learn — Milestone v2.19 close

- `TASK-004/005/006/007` resolved — `CHG-004` 2006da8, `CHG-005` 837a966, `CHG-006`
  2a93f52, `DEC-004` 650de28. `issue-v19-tp-multigpu-unverified` and
  `issue-v19-ring-parallel-noop` resolved; `issue-v19-shared-nccl-context-reset` open
  (tracked — fix is a future hardening thread: detect context resets vs the shared
  singleton's communicators/streams).
- `HYP-002` validated by `EV-004` (RED) then refuted by `EV-005` (GREEN after the fix).
- **Milestone v2.19 closed**: 3/3 phases green on the targeted (NCCL-enabled, curated)
  configurations. Note: the full-suite baseline in this environment is unstable for
  pre-existing reasons (libcuda `cuStreamDestroy` SIGSEGV after interleaved
  `cudaDeviceReset` suites, tracked as `issue-v19-shared-nccl-context-reset`; GPUs 0/1
  externally loaded — use `CUDA_VISIBLE_DEVICES=2,3+`).
