# RIL — Repository Intelligence Layer (index)

Maintained by the autonomous engineering loop. Full guidance: `.agents/skills/graph-engineering/SKILL.md`
(single source; `.claude/skills` is a whole-directory symlink to `.agents/skills` so Claude Code
discovers it under the skills it scans).
Per-round narratives + graph deltas: `.planning/ril_autonomous_roundN.md` (latest: Round 18).
Milestone thread: v2.15 (Test Quality) closed at Round 14 → **v2.16 "Distributed Multi-GPU
Verification"** closed at Round 16 → **v2.17 "Distributed Ops On Real Multi-GPU"** closed at
Round 17 → **v2.18 "MeshBarrier On The Verified Layer + Distributed Robustness"** opened and
**closed at Round 18**.

## Latest round
- **Round 18 (2026-08-12)** — milestone v2.18 kickoff through **close (P1+P2+P3)**. **P1**
  (`change-v18-p1` c5f302a): per-rank MeshBarrier rendezvous tests FAIL RED — the legacy
  per-instance host event-poll recorded events on empty internal streams that fired
  immediately (no cross-rank arrival signal; `HYP-001`/`EV-001`). **P2** (`change-v18-p2`
  0a12a84): converged `MeshBarrier` onto the verified `NcclBarrier` layer (sync / async /
  synchronize_devices; subset throws; per-rank `NoDeadlock`) — 5/5 multi-GPU barrier tests
  green. **P3** (`change-v18-p3` 7914b34): harness robustness + R16 HIGH-B — bounded 120s
  thread-per-rank barrier (`RankBarrierTimeout` instead of an unkillable hang) and
  `NcclContext::mark_comm_aborted` so dead comms fail fast and recover via destroy+reinit
  instead of poisoning later tests; shared `distributed_test_common.h` single harness home.
  `issue-v17-meshbarrier-multigpu` resolved; `issue-v17-dist-ops-harness-flake` resolved by
  disposition `DEC-002` (both root-cause mechanisms now bounded failures; never re-observed).
  Full suite **1461/1423/0**; 2-GPU cross-suite 49/49 (14 env-skips); 5x+3x stress stable.
  Milestone **closed**. See `ril_autonomous_round18.md`.
- **Round 16 (2026-08-11)** — milestone v2.16 Phase 3: implemented the real distributed matmul
  multi-GPU path `DistributedMatmul::matmul_multi_gpu` (row-split compute + NCCL all-gather) and
  replaced the last unconditional distributed skip with three real multi-GPU tests
  (`MultiGpu_RowSplitAllGather`, `MultiGpu_AlphaBeta`, `MultiGpu_RejectsBadShapes`) driven
  thread-per-rank. Proved the R15 "multi-process required" assumption unnecessary and superseded
  it with `decision-matmul-thread-per-rank`. **14/14 DistributedMatmul on 2 GPUs; full suite
  1451/1423/0 EXIT=0.** See `ril_autonomous_round16.md`.
- **Round 15 (2026-08-11)** — milestone v2.16 kickoff. The 14 `NcclCollectivesTest` cases had
  never run (`NCCL_TESTS_AVAILABLE` was a dead gate); running them for real exposed three bugs:
  `NcclContext` mesh-init never created communicators (nullcomm → invalid argument), sequential
  `ncclCommInitRank` deadlocked (it is a per-rank collective), collectives hardcoded `get_comm(0)`
  (invalid for rank≠0), and `NcclBarrier` passed a host stack int to `ncclAllReduce` (illegal
  memory access). Fixed in `fix(nccl) bfb9e80`; suite rewritten as genuine per-rank multi-GPU
  tests → **14/14 pass on 2 GPUs** (`NCCL_TESTS_AVAILABLE=1`). See
  `ril_autonomous_round15.md`.

## Active tasks (by priority_score; threshold 3.0)
(none — milestone v2.18 closed)

## Resolved (v2.18)
- `task-v18a-meshbarrier-tests` (TASK-001) RESOLVED by `change-v18-p1` (c5f302a) — per-rank
  MeshBarrier rendezvous/sync-devices tests prove the legacy event-poll had no cross-rank
  arrival semantics (RED evidence HYP-001/EV-001).
- `task-v18b-meshbarrier-converge` (TASK-002) RESOLVED by `change-v18-p2` (0a12a84) —
  MeshBarrier converged onto verified NcclBarrier; 5/5 multi-GPU barrier tests green.
- `task-v18c-harness-robustness` (TASK-003) RESOLVED by `change-v18-p3` (7914b34) — bounded
  barrier + self-healing broken-comm state; harness flake + R16 HIGH-B closed.
- `issue-v17-meshbarrier-multigpu` RESOLVED; `issue-v17-dist-ops-harness-flake` RESOLVED by
  disposition `DEC-002` (both mechanisms neutralized; stall never re-observed).

## Resolved (v2.17)
- `task-v17c-syncbn-multigpu-backward` RESOLVED by `change-v17-p3` (a8334d1) — SyncBatchNorm
  multi-GPU gradient path corrected vs global-batch host reference.
- `task-v17b-dist-ops-nccl-converge` RESOLVED by `change-v17-p2` (5c77d01) — high-level ops
  converged onto the verified NCCL layer; 4/4 multi-GPU ops tests green (was 4/4 fail in P1).
- `task-v17a-dist-ops-harness` RESOLVED by `change-v17-p1` (4b2d10a) — thread-per-rank harness
  + 4 ready `DISABLED_` multi-GPU regression tests; all 4 legacy ops multi-GPU paths confirmed
  broken (2x A100). `task-v17d` (redundant annotation) abandoned.

## Candidate follow-up (below threshold / env-bound)
- MPI enablement — `NOVA_ENABLE_MPI=ON` needs MPI headers/dev absent on this host (env-bound).

## Recent decisions
- `decision-v17-dist-ops-real-multigpu` (R17) — milestone v2.17: the high-level distributed ops'
  multi-GPU paths (10 "Requires single GPU" skips) are the next never-run surface, and they back
  the SyncBatchNorm multi-GPU training path. P1 harness (+evidence) → P2 converge on the verified
  NCCL layer (or fix legacy per evidence) → P3 SyncBatchNorm gradient verify.
- `decision-matmul-thread-per-rank` (R16) — the distributed matmul multi-GPU path does NOT require
  separate OS processes: genuine multi-device NCCL collectives run correctly thread-per-rank in one
  process (proven by the R15 NCCL suite). `matmul_multi_gpu` uses row-split + NcclAllGather,
  avoiding fork/exec and CUDA-fork hazards. Supersedes `decision-dist-matmul-multiprocess`.
- `decision-nccl-current-device-routing` (R15) — NCCL collectives resolve their communicator via
  the currently-active device (`NcclContext::current_comm()`), never hardcoded rank 0's; each
  rank owns its device/comm/stream. Supersedes the old "use device 0's communicator" assumption.
- `decision-dist-matmul-multiprocess` (R15) — distributed matmul's multi-GPU path is intentionally
  a multi-process execution model; the single-process suite documents the limitation rather than
  faking single-GPU semantics. Convert later with a process-per-rank harness.
- `decision-heap-singletons-exit` (R12/R13) — heap-allocate never-destroyed singletons whose
  destructors call CUDA driver APIs; supersedes the old "Meyer's singleton" for
  NcclContext/MeshStreams.
- `decision-test-error-hygiene` (R13) — negative-path and `cudaGetLastError`-asserting tests must
  drain sticky CUDA error state so tests cannot poison each other.

## Resolved
- `task-dist-matmul-multiprocess` RESOLVED by `change-r16-matmul-multigpu` — 
  `issue-dist-matmul-multigpu-unverified` fixed; `matmul_multi_gpu` (row-split + NcclAllGather)
  implemented and tested thread-per-rank on 2 GPUs; milestone v2.16 closed.
- `task-r15-nccl-collectives` RESOLVED by `change-r15-nccl-context` + `change-r15-nccl-tests`
  (fix(nccl) bfb9e80) — NCCL multi-GPU suite 14/14 green; exposed/fixed: `issue-nccl-mesh-comm-init`,
  `issue-nccl-init-deadlock`, `issue-nccl-comm-routing`, `issue-nccl-barrier-host-ptr`.
- `task-skip-audit` RESOLVED (R14) — all stale/non-env hard skips replaced or their hidden bugs fixed.
- `task-cusparse-exit-dtor` RESOLVED by `change-r14-cusparse`.
- `task-scheduler-noncontig-mode` RESOLVED by `change-r14-static-batching`.
- `issue-segsort-poison` RESOLVED by `change-r13-poisoners`.
- `issue-meshstreams-exit-crash` RESOLVED by `change-r13-meshstreams`.
- `issue-backward-dinput-nan` RESOLVED by `change-r13-backward-nan`.
- `hypothesis-thrust-plan-cache` `refuted` by `evidence-r13-bisection`.

## Cross-round register
`.planning/codebase/CONCERNS.md` — keep audit dates current; mark stale items, do not delete.
