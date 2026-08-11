# RIL — Repository Intelligence Layer (index)

Maintained by the autonomous engineering loop. Full guidance: `.agents/skills/graph-engineering/SKILL.md`
(single source; `.claude/skills` is a whole-directory symlink to `.agents/skills` so Claude Code
discovers it under the skills it scans).
Per-round narratives + graph deltas: `.planning/ril_autonomous_roundN.md` (latest: Round 15).
Milestone thread: v2.15 (Test Quality) closed at Round 14 → **v2.16 "Distributed Multi-GPU
Verification"** opened at Round 15.

## Latest round
- **Round 15 (2026-08-11)** — milestone v2.16 kickoff. The 14 `NcclCollectivesTest` cases had
  never run (`NCCL_TESTS_AVAILABLE` was a dead gate); running them for real exposed three bugs:
  `NcclContext` mesh-init never created communicators (nullcomm → invalid argument), sequential
  `ncclCommInitRank` deadlocked (it is a per-rank collective), collectives hardcoded `get_comm(0)`
  (invalid for rank≠0), and `NcclBarrier` passed a host stack int to `ncclAllReduce` (illegal
  memory access). Fixed in `fix(nccl) bfb9e80`; suite rewritten as genuine per-rank multi-GPU
  tests → **14/14 pass on 2 GPUs** (`NCCL_TESTS_AVAILABLE=1`). See
  `ril_autonomous_round15.md`.

## Active tasks (by priority_score; threshold 3.0)
| score | task | status |
|-------|------|--------|
| — | (none above threshold this round) | — |

## Candidate follow-up (below threshold / env-bound)
- `task-dist-matmul-multiprocess` — implement a real multi-process execution harness so
  `DistributedMatmulTest.MultiGpu_RequiresMultiProcess` becomes a real test (currently an
  unconditional, *documented* skip). category: core feature; effort: high.
- MPI enablement — `NOVA_ENABLE_MPI=ON` needs MPI headers/dev absent on this host (env-bound).

## Recent decisions
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
