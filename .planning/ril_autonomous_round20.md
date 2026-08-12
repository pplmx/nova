# RIL — Round 20 (2026-08-12) — milestone v2.20 "TensorParallelMatmul Production Hardening"

## Round focus

**Open milestone v2.20** as a hardening follow-up to the v2.19 TP rewrite. The cpp-review
of the v2.19 P2 diff returned a **BLOCK** on two HIGH findings that still applied to the
committed code, plus a couple of MEDIUMs. P1 closes those; P2 dispositions the
non-functional `TensorParallelLayers` skeletons; P3 (context-reset detection) was
investigated and determined infeasible in this environment (recorded, not shipped).

## Observation / Model

- `HYP-002`-era review: TP matmul derived rank from the raw device index (wrong on
  non-default NCCL groups; no membership bounds → possible OOB) and discarded
  `all_reduce_async`'s `NcclResult` (a failed collective silently returned garbage).
- `TensorParallelLayers` (ColumnParallelLayer / RowParallelLayer / TensorParallelMLP,
  "shipped" v1.3): zero tests/usages; `forward()` passed `B = nullptr` into the
  cuBLAS-backed TP matmul (null crash) and MLP forward was an empty body — the same
  incomplete-surface pattern as `RingSequenceParallelism`.
- `issue-v19-shared-nccl-context-reset`: full/curated runs that interleave
  `cudaDeviceReset`-heavy suites with the shared NCCL singleton SEGV in libcuda
  `cuStreamDestroy`. The 2-suite repro (SequenceParallelTest then SP-multigpu) was
  intermittent.

## Execute — Phase 1 (TASK-008, `CHG-007` 2137cea)

- **HIGH-1**: rank now resolved via `ctx_.rank_of_device(device)` (maps through the
  group's device_ids; throws `NcclException` when the active device is not in the group)
  instead of the raw `cudaGetDevice` index — matching the verified distributed matmul.
- **HIGH-2**: the column AllReduce is now checked: a failed `NcclResult` throws
  `NcclException` (matching `distributed/reduce.cu`) instead of returning success with
  garbage.
- **MEDIUM-6**: added `MultiGpu_TensorParallelRowParallelNonDivisible` (m % tp reject).
- Corrected the RowParallel header docs (split along m, A replicated, C partitioned).

## Verify — Phase 1

TP+NCCL suites **20/20 GREEN** at 2 GPUs; TP acceptance **5/5 at 4 GPUs** (EV-009).

## Execute — Phase 2 (TASK-009, `CHG-008`)

`TensorParallelLayers` stubs now **fail fast**: every `forward()` throws an explicit
"not implemented: no weight storage" error (no more null-B cuBLAS crash / empty MLP
output), class docs point at `TensorParallelMatmul` as the supported primitive. New
`TensorParallelLayersTest` pins the reject contract for all three layers; real
weight-managed implementation deferred (`DEC-006` supersedes `DEC-004`, `TASK-010`).

## Verify — Phase 2

3/3 fail-fast GREEN; neural/TP/SP regression smoke green (EV-009).

## Execute — Phase 3 (investigation, not shipped)

Attempted to fix `issue-v19-shared-nccl-context-reset` (context-loss auto-detection on
the shared singleton). A driver probe showed `cuCtxGetCurrent` **returns the same
CUcontext pointer after `cudaDeviceReset`** (driver reuses the primary-context address),
so handle-comparison-based detection is infeasible (EV-008). Robust fixes are
architectural (don't share NCCL resources across a process whose devices get reset; or
reorganize the test suites). Disposition recorded: use the documented curated multi-GPU
workflow + standard baseline without `NCCL_TESTS_AVAILABLE` (matches R15-R18).

## Learn — Milestone v2.20 close

- `TASK-008/009` resolved (`CHG-007` 2137cea, `CHG-008`); `TASK-010` (weight-managed
  layers) deferred as a feature follow-up.
- `issue-v20-tp-layers-stubs` resolved (fail-fast disposition); `issue-v19-shared-nccl-
  context-reset` OPEN with disposition (curated workflow; detection infeasible — EV-008).
- **Milestone v2.20 closed**: 2/2 phases green; verified against the reviewer's BLOCK.
