---
gsd_state_version: 1.0
milestone: v2.21
milestone_name: Weight-Managed TensorParallelLayers
status: Complete
last_updated: "2026-08-12"
last_activity: 2026-08-12 — Round 21: P1-P3 complete, milestone closed
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 0
  completed_plans: 0
---

# Project State

**Project:** Nova CUDA Library Enhancement
**Last Updated:** 2026-08-12

## Current Position

Milestone: v2.21 Weight-Managed TensorParallelLayers
Status: Complete (Round 21)
Last activity: 2026-08-12 — priority: TASK-010 implemented (real weight
shards), closing the DEC-006 fail-fast disposition of the v1.3 layer stubs

## Milestone v2.21 — Weight-Managed TensorParallelLayers

Replace the DEC-006 fail-fast disposition of the v1.3 layer stubs with real
weight-sharded layers (TASK-010 / issue-v20-tp-layers-stubs):

| Phase | Name | Status |
|-------|------|--------|
| 1 | Reference-parity tests: single-GPU full-weight refs (col/row/MLP + silu) and multi-GPU thread-per-rank (col shard concat / row AllReduce / gated MLP parity; shape rejection); RED against the stubs | Complete (Round 21) — 4/4 layer + 2/2 silu single-GPU; 5/5 multi-GPU layer parity on 2 GPUs (EV-010/EV-011) |
| 2 | Weight-managed implementation: per-rank device weight shards (set_weight slices via rank_of_device), col/row/gated-MLP forward, silu_and_mul kernel, per-instance cuBLAS handle, divisibility validation | Complete (Round 21) — 15/15 isolated NCCL cross-suite on 2 & 4 GPUs; full baseline EXIT=0 |
| 3 | RIL close: issue + TASK-010/011/012/013 resolved, docs/round record | Complete (Round 21) |

Decisions: DEC-007 (milestone direction — Megatron weight-shard semantics, new
explicit in/out-features API, >300-line diff exception). NOTE: NCCL multi-GPU
suites still hang when interleaved with `cudaDeviceReset` suites in one process
(pre-existing issue-v19-shared-nccl-context-reset — use the isolated curated
filter for cross-suite runs).

## Milestone v2.20 — TensorParallelMatmul Production Hardening

Close the cpp-review BLOCK on the v2.19 TP rewrite and disposition the last
non-functional neural parallel-training surface:

| Phase | Name | Status |
|-------|------|--------|
| 1 | TP matmul: rank via `rank_of_device` (membership validation) + propagate collective failures + RowParallel non-divisible test | Complete (Round 20) — closes review HIGH-1/HIGH-2/MEDIUM-6; TP+NCCL 20/20, acceptance 5/5 at 4 GPUs (2137cea, EV-009) |
| 2 | TensorParallelLayers stubs fail fast (were null-B cuBLAS crash / empty MLP body) | Complete (Round 20) — 3/3 reject tests; weight-managed implementation deferred (DEC-006, TASK-010) |

`issue-v19-shared-nccl-context-reset` (full-suite/curated crash after interleaved
`cudaDeviceReset`) investigated for P3: driver-level detection infeasible
(`cuCtxGetCurrent` returns the same address after reset, EV-008); disposition =
documented curated multi-GPU workflow + standard baseline without `NCCL_TESTS_AVAILABLE`.

## Milestone v2.19 — Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism)

Extend the R15-R18 thread (every never-run multi-GPU surface hides real bugs) to the
neural parallel-training layer — TensorParallelMatmul's multi-GPU execution was only
tested through buffer-size formulas, and the sequence-parallel tests passed
`comm = nullptr`:

| Phase | Name | Status |
|-------|------|--------|
| 1 | Per-rank TensorParallelMatmul multi-GPU RED tests (evidence) | Complete (Round 19) — 4/4 RED: ColumnParallel wrong output, non-divisible silently dropped, RowParallel ignored rank (2006da8, EV-004/HYP-002) |
| 2 | Correct TP matmul partition math on the verified NCCL layer | Complete (Round 19) — zero-padded strided column blocks + one AllReduce; rank-correct RowParallel; shape rejection; per-instance stream-bound cuBLAS handle (837a966, EV-005) |
| 3 | SequenceParallel real-comm + ring no-op disposition | Complete (Round 19) — gather_kv/scatter_output/all_reduce_sequence verified 4/4 on 2/4 GPUs (2a93f52); RingSequenceParallelism multi-GPU made fail-fast (650de28, DEC-004) |

`HYP-002` (TP ColumnParallel contiguous-slice math broken) validated by RED then refuted
by the converged GREEN acceptance tests. SequenceParallel collectives were verified
correct on first real-comm run — a positive refutation of the "always broken"
generalization. Follow-ups tracked: `issue-v19-shared-nccl-context-reset` (full-suite
NCCL-enabled run can SIGSEV in libcuda cuStreamDestroy after interleaved
cudaDeviceReset suites — R15-R18 never ran the NCCL-enabled full suite; pre-existing,
order/timing-sensitive), `issue-v19-ring-parallel-noop` (closed by DEC-004 fail-fast).

## Milestone v2.18 — MeshBarrier On The Verified Layer + Distributed Robustness

Close out the distributed-ops convergence thread (v2.15→v2.17 converged every
high-level op except `MeshBarrier`) and harden the collective test harness
against the two tracked follow-ups (harness flake, R16 review HIGH-B context
poisoning):

| Phase | Name | Status |
|-------|------|--------|
| 1 | Per-rank MeshBarrier multi-GPU rendezvous tests (evidence) | Complete (Round 18) — RED proof the legacy event-poll had no cross-rank arrival signal (HYP-001/EV-001) |
| 2 | Converge MeshBarrier onto verified NcclBarrier (sync/async/devices) | Complete (Round 18) — 5/5 multi-GPU barrier tests green (0a12a84) |
| 3 | Harness robustness: bounded barrier + self-healing broken-comm state | Complete (Round 18) — fail-fast vs dead comms; diagnostic RankBarrierTimeout vs dead ranks (7914b34) |

## Milestone v2.17 — Distributed Ops On Real Multi-GPU (Complete)

High-level distributed ops' multi-GPU paths proven broken (P1), converged onto the
verified NCCL layer (P2), and the SyncBatchNorm multi-GPU gradient path verified
against a global-batch host reference (P3). Full suite 1456/1423/0.

## Milestone v2.16 — Distributed Multi-GPU Verification (Complete)

NCCL multi-GPU collectives 14/14 real; distributed pool/mesh verified; distributed
matmul real path (row-split + NCCL all-gather) thread-per-rank.

## Milestone History

| Milestone | Status | Date | Requirements |
|-----------|--------|------|--------------|
| v1.0 Production Release | Shipped | 2026-04-24 | 58 |
| v1.1 Multi-GPU Support | Shipped | 2026-04-24 | 13 |
| v1.2 Toolchain Upgrade | Shipped | 2026-04-24 | 9 |
| v1.3 NCCL Integration | Shipped | 2026-04-24 | 26 |
| v1.4 Multi-Node Support | Shipped | 2026-04-24 | 15 |
| v1.5 Fault Tolerance | Shipped | 2026-04-26 | 20 |
| v1.6 Performance & Training | Shipped | 2026-04-26 | 12 |
| v1.7 Benchmarking & Testing | Shipped | 2026-04-26 | 27 |
| v1.8 Developer Experience | Shipped | 2026-04-26 | 16 |
| v1.9 Documentation | Shipped | 2026-04-26 | 12 |
| v2.0 Testing & Quality | Shipped | 2026-04-26 | 12 |
| v2.1 New Algorithms | Shipped | 2026-04-26 | 12 |
| v2.2 Comprehensive Enhancement | Shipped | 2026-04-27 | 18 |
| v2.3 Extended Algorithms | Shipped | 2026-04-28 | 13 |
| v2.4 Production Hardening | Shipped | 2026-04-28 | 15 |
| v2.5 Error Handling & Recovery | Shipped | 2026-04-28 | 12 |
| v2.6 Transformer & Inference Optimization | Shipped | 2026-04-29 | 18 |
| v2.7 Comprehensive Testing & Validation | Shipped | 2026-04-30 | 16 |
| v2.8 Numerical Computing & Performance | Shipped | 2026-05-01 | 20 |
| v2.9 Architecture Refactor | Shipped | 2026-05-01 | 7 |
| v2.10 Sparse Solver Acceleration | Shipped | 2026-05-01 | 11 |
| v2.11 Performance Tooling | Shipped | 2026-05-02 | 14 |
| v2.12 Advanced Quantization | Shipped | 2026-05-03 | 14 |
| v2.13 Transformer Optimization | Shipped | 2026-05-06 | 25 |
| v2.14 Documentation Quality | Shipped | 2026-05-07 | 31 |
| v2.15 Test Quality Assurance | Complete | 2026-05-09 | 5 phases |
| v2.16 Distributed Multi-GPU Verification | Complete | 2026-08-11 | 3 phases |
| v2.17 Distributed Ops On Real Multi-GPU | Complete | 2026-08-12 | 3 phases |
| v2.18 MeshBarrier On The Verified Layer + Distributed Robustness | Complete | 2026-08-12 | 3 phases |
| v2.19 Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism) | Complete | 2026-08-12 | 3 phases |
| v2.20 TensorParallelMatmul Production Hardening | Complete | 2026-08-12 | 2 phases |

---

## State updated: 2026-08-12 — Milestone v2.21 complete (RIL Round 21)

TASK-010 / issue-v20-tp-layers-stubs resolved. The v1.3 TensorParallelLayers
scaffolding (ColumnParallelLayer / RowParallelLayer / TensorParallelMLP) owned
no weight storage and was disposed as fail-fast stubs in v2.20 (DEC-006). This
milestone implements them for real with **weight-sharded Megatron semantics**:
each layer owns its rank's shard of the weight on device; set_weight() uploads
the full weight and slices the rank shard via NcclContext::rank_of_device (the
v2.20 verified convention); ColumnParallelLayer::forward is a local shard
matmul (sharded output, no comm), RowParallelLayer::forward is local matmul +
one block-wise AllReduce (replicated output), TensorParallelMLP is a SiLU-gated
FFN (gate/up column-parallel, down row-parallel). Per-instance stream-bound
cuBLAS handle for thread-per-rank safety; divisibility validated at construction
(backing the v2.19 finding that the pre-v2.20 shard math was broken). Added the
`silu_and_mul` gated activation.

Acceptance (real multi-GPU, CUDA_VISIBLE_DEVICES=2,3): column shard concat ==
single-GPU full-weight reference; row AllReduce replicated == reference; gated
MLP == single-GPU full-weight reference — 15/15 isolated NCCL cross-suite on 2
and 4 GPUs (EV-010/EV-011), TP+layer+sequence multi-GPU green. Single-GPU tp<=1
path: 6/6 (4 layer + 2 silu) reference parity. Full-suite baseline EXIT=0.
NOTE: running the NCCL multi-GPU suites interleaved with cudaDeviceReset suites
(ActivationTest) in one process still hangs/crashes — the pre-existing
issue-v19-shared-nccl-context-reset (order/timing-sensitive); the NCCL cross
-suite must be run via the isolated curated filter, per the documented
disposition. API: layer ctors changed to explicit (ctx, in_features,
out_features); zero external callers (verification grep clean).
