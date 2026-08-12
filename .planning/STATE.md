---
gsd_state_version: 1.0
milestone: v2.19
milestone_name: Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism)
status: Complete
last_updated: "2026-08-12"
last_activity: 2026-08-12 — Round 19: P1-P3 complete, milestone closed
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

Milestone: v2.19 Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism)
Status: Complete (Round 19)
Last activity: 2026-08-12 — priority None: TensorParallelMatmul multi-GPU proven
broken then converged (EV-004/EV-005); SequenceParallel real-comm verified 4/4;
RingSequenceParallelism multi-GPU no-op made fail-fast

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

---

## State updated: 2026-08-12 — Milestone v2.18 complete (RIL Round 18)
3/3 phases green. MeshBarrier — the last unconverged high-level distributed op —
was proven to have no cross-rank arrival semantics (per-instance host event-poll
on empty internal streams) and converged onto the verified NcclBarrier layer;
5/5 per-rank multi-GPU barrier tests green. Harness hardening: a bounded 120s
thread-per-rank barrier turns a dead rank into a diagnosed RankBarrierTimeout
(previously an unkillable >120s hang), and NcclContext now learns when a
communicator is aborted (mark_comm_aborted) so dead comms fail fast and recover
via destroy+reinit instead of poisoning the rest of the suite. Full-suite
baseline 1461/1423/0; 2-GPU cross-suite 49/49 (14 env-skips); 5x+3x 2-GPU stress
stable. Next milestone: TBD. Both v2.17 follow-ups resolved
(issue-v17-meshbarrier-multigpu, issue-v17-dist-ops-harness-flake via DEC-002).
