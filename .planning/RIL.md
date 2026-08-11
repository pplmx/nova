# RIL — Repository Intelligence Layer (index)

Maintained by the autonomous engineering loop. Full guidance: `.claude/skills/graph-engineering/SKILL.md`
(loadable skill; `.agents/skills/graph-engineering/SKILL.md` is a symlink alias for the agents.md standard).
Per-round narratives + graph deltas: `.planning/ril_autonomous_roundN.md` (latest: Round 13).

## Latest round
- **Round 13 (2026-08-10)** — full suite 1411 pass/4 fail + exit SIGSEGV → **1419 pass / 0 fail / EXIT=0**.
  Fixed: SegmentedSort-process poisoners (UB test patterns), MeshStreams exit-teardown crash,
  SyncBatchNorm backward all-NaN gradients (from un-skipped tests). See
  `ril_autonomous_round13.md`.

## Active tasks (by priority_score; threshold 3.0)
| score | task | status |
|-------|------|--------|
| ~4.5 | `task-scheduler-noncontig-mode` — `scheduler_edge_test.cpp:264` skips "non-continuous batching not implemented"; verify whether `step()` leaves pending requests stuck and either implement or make the skip honest. Category: core feature. | active |
| ~3.2 | `task-cusparse-exit-dtor` — `nova::sparse::detail::CusparseContext::~CusparseContext` runs `cusparseDestroy` at process exit (latent exit-crash class like NcclContext/MeshStreams). No crash observed — act on evidence only. Category: stability. | active |
| low | `task-skip-audit` — audit remaining `GTEST_SKIP`s (MPI flag, NCCL multi-GPU, block_manager_edge OOM, BiCGSTAB convergence) for guards that silently skip on a normal GPU. | active |

## Recent decisions
- `decision-heap-singletons-exit` (R12/R13) — heap-allocate never-destroyed singletons whose
  destructors call CUDA driver APIs; supersedes the old "Meyer's singleton" for
  NcclContext/MeshStreams.
- `decision-test-error-hygiene` (R13) — negative-path and `cudaGetLastError`-asserting tests must
  drain sticky CUDA error state so tests cannot poison each other.
- `decision-keep-round-docs` (R13) — RIL stored as markdown round docs + graph-delta blocks, not a
  graph DB.

## Resolved (round 13)
- `issue-segsort-poison` RESOLVED by `change-r13-poisoners` (UB test patterns faulting the driver).
- `issue-meshstreams-exit-crash` RESOLVED by `change-r13-meshstreams` (fix(distributed) f9493d8).
- `issue-backward-dinput-nan` RESOLVED by `change-r13-backward-nan` (fix(neural) 4b6c2c1).
- `hypothesis-thrust-plan-cache` `refuted` by `evidence-r13-bisection`.

## Cross-round register
`.planning/codebase/CONCERNS.md` — keep audit dates current; mark stale items, do not delete.
