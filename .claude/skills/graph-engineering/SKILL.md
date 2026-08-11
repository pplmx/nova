---
name: graph-engineering
description: Maintain this repo's repository-intelligence layer (RIL) - a typed engineering graph of components, issues, hypotheses, evidence, decisions, changes and tasks, with lifecycle, scoring and cross-session loading. Use for autonomous/continuous engineering rounds on this repo (the /goal loop: OBSERVE -> MODEL -> EVALUATE -> SELECT -> EXECUTE -> VERIFY -> LEARN), when recording a research/fix outcome or next-task into the graph, or when asked to continue prior autonomous work.
---

# Graph Engineering (RIL)

Codifies the engineering-graph methodology this repo's autonomous rounds use. It exists because the `/goal` directive is session-scoped and ephemeral; this skill gives the repo a durable, canonical reference. Ground it in what the repo actually stores (markdown round docs), not in an idealized graph DB.

## When to use
- Starting or continuing an autonomous engineering round on this repo.
- Recording findings, root causes, decisions, or scored next-tasks (the "LEARN" step, or a `docs(ril)` commit).
- Cross-session reload of prior work (active tasks + recent decisions + repo intel).

## Schema — canonical vocabulary
Nodes carry `id/type/status/created_at/updated_at/confidence`:

- `component` — module/service/file entity
- `issue` — identified problem (bug / risk / debt)
- `hypothesis` — unverified root cause (confidence 0–1)
- `evidence` — concrete observation supporting/refuting a hypothesis; must cite a source (commit hash, test name, file:line)
- `decision` — made decision; record rationale + alternatives_rejected
- `change` — actual code change; tie to a commit sha
- `task` — next action; carries `priority_score`

Edges are directional and semantic (no bare "relates to"):

- `depends_on` — task→task, component→component
- `causes` — issue→issue (mark root-cause vs symptom)
- `blocks` — task→task
- `validates` / `refutes` — evidence→hypothesis
- `resolves` — change→issue
- `supersedes` — decision→decision (records evolution, never overwrite)

**Rule:** a hypothesis is not treated as fact in EVALUATE until it has a `validates`/`refutes` edge.

Node id convention: `<type>-<short-kebab-slug>`, e.g. `issue-segsort-poison`, `hypothesis-thrust-plan-cache`, `change-r13-segsort`.

## Storage — as actually practiced in this repo
The RIL lives in markdown, not a graph database. Locations:

- `.planning/ril_autonomous_roundN.md` — one doc per autonomous round: narrative + a **graph-delta** block (below).
- `.planning/RIL.md` — the index: active tasks ranked by `priority_score`, recent decisions, latest-round pointer.
- `.planning/codebase/CONCERNS.md` — cross-round issue register; mark items STALE, never delete.
- `.planning/STATE.md`, `ARCHITECTURE.md`, `PROJECT.md`, ... — existing planning docs; keep in sync when closing items.

Graph-delta convention — append this compact, machine-scannable block to each round doc (existing docs may predate it; apply going forward):

```yaml
## Graph delta
nodes:
  - id: issue-segsort-poison
    type: issue
    status: resolved
    confidence: 0.95
  - id: change-r13-backward-nan
    type: change
    status: active
edges:
  - from: change-r13-backward-nan
    type: resolves
    to: issue-backward-nan
  - from: evidence-<slug>
    type: refutes
    to: hypothesis-thrust-plan-cache
```

## Lifecycle
- Statuses: `active` / `stale` / `resolved` / `superseded` / `abandoned`.
- A hypothesis/task untouched for N rounds (default 10) → `stale`; keep for audit, skip in EVALUATE unless new evidence re-activates it.
- Decisions are never deleted; replace by adding a new decision with a `supersedes` edge.
- Concurrency (only when multiple agents touch the repo): optimistic-lock node/edge writes, set `owner` on `in_progress` tasks, evidence is append-only. Single-instance fast path: skip the locking machinery.

## Scoring
`priority_score = category_weight × severity × confidence × (1/√effort) × unlock_factor`

- category_weight: correctness/security 10 · stability/critical bug 8 · core feature 6 · performance 5 · test-quality 4 · maintainability 3 · DX 2 · docs 1
- severity = blast radius × trigger probability; confidence = how much the root-cause claim is backed by `validates` evidence; effort = implementation cost (prevents gaming high weights with trivial items); unlock_factor = downstream tasks enabled.
- Switch focus to a new task only when it scores ≥1.5× the current task; record the switch as a decision.

## Round workflow
1. **OBSERVE** — git status/diff/log, issues/TODO/FIXME, tests/CI build state, docs, recent changes. Understand component/data-flow relationships, not just isolated TODOs. Reproduce suspected failures with minimal `--gtest_filter` prefixes before theorizing.
2. **MODEL** — load active tasks (top-K by score) + their directly-linked components/issues/hypotheses + recent decisions. Expand along edges only as needed.
3. **EVALUATE** — score candidates; pick the highest-scoring `active` task above threshold (default 3.0).
4. **SELECT** — one focus line at a time.
5. **EXECUTE** — repo-local, git-revertible changes only. Lock the task (`in_progress` + owner) if multi-instance.
6. **VERIFY** — build/lint/tests; on failure fix the root cause and re-verify. Hard prohibitions: do NOT delete/skip tests, lower assertions or thresholds, comment out failing cases, or change quality bars to manufacture a pass. If a fix cannot be made reliable, roll it back and mark the hypothesis `refuted` with evidence.
7. **LEARN** — write nodes/edges + the graph delta; commit messages reference the task/issue ids.

**Stop conditions** (all must hold): no active task with score > threshold · high-severity issues resolved or documented-with-decision · last VERIFY green · two consecutive deep-dive rounds produced no above-threshold task.

## Repo intel (learned; keep current — mostly from Rounds 9–13)
- **Device-poisoning class**: UB in tests (reads through freed device pointers, leaving invalidated CUDA stream captures, sticky errors from deliberately-failed giant allocs) faults the CUDA driver so later thrust/cub dispatch fails with `cudaErrorInvalidDevice`, or the process SIGSEGVs at exit. Round 13 fixed the three known poisoners + the exit crash.
- **Exit-teardown class**: static singletons whose destructors call CUDA driver APIs crash inside libcuda at exit. `NcclContext` (R12) and `MeshStreams` (R13) are fixed; `nova::sparse::detail::CusparseContext::~CusparseContext` still runs `cusparseDestroy` at exit (latent, no observed crash yet — fix only on evidence).
- **Sticky-error hygiene**: tests asserting `cudaGetLastError()==cudaSuccess` must drain the error in the fixture's SetUp so the assertion reflects only that test; negative-path tests must drain after forcing an error (else they poison later tests).
- **Skips can hide real bugs**: `SyncBatchNormTest`'s GPU tests always skipped because they queried `DeviceMesh::device_count()` without `initialize()` and ran before any mesh test — un-skipping exposed a genuine all-NaN backward gradient bug. Before trusting any `GTEST_SKIP`, verify its guard.
- **Known-skip inventory** (verify current state before relying on it): MPI tests (build flag off), NCCL multi-GPU tests, `block_manager_edge` OOM guards, `scheduler_edge` "non-continuous batching not implemented" (half-implemented mode), sparse BiCGSTAB convergence skip.
- **Quiet-GPU convention**: the suite runs on an idle GPU. `cudaMemGetInfo`-based assertions assume no concurrent allocation churn from other processes.
- **Correct RIL narrative**: Round 12's "thrust plan-cache staleness" explanation for the SegmentedSort failure was wrong (Thrust 2.0.8 has no host plan cache); the real cause was the test-UB poisoners (see Round 13). Cite Round 13, not 12.

## Verification discipline
- Evidence before assertions: name the root cause, fix it, then verify with targeted reproducers AND the full suite.
- Report failures faithfully — never weaken an assertion to get green.
