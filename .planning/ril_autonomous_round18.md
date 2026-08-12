# RIL — Round 18 (2026-08-12) — milestone v2.18 "MeshBarrier On The Verified Layer + Distributed Robustness"

## Round focus

**Open milestone v2.18** and execute **P1 (TASK-001)**: build a real thread-per-rank
multi-GPU `MeshBarrier` test (the `run_per_rank` harness already in this file) that
drives `synchronize()` and `synchronize_devices()` concurrently from one thread per
device — the last unconverged high-level distributed op (v2.17 deliberately deferred
it as `issue-v17-meshbarrier-multigpu`). Expect RED: the legacy per-instance host
event-poll path (barrier.cu) has never been driven from real per-rank threads.

## Observation (evidence / model)

- v2.17 converged DistributedReduce/AllGather/Broadcast onto the verified NCCL layer
  but `src/cuda/distributed/barrier.cu` is untouched (Jul 23), still the legacy
  per-instance host event-poll: `barriering_` atomic flag, `cudaEventQuery` polling
  loop with 10us sleeps, cross-thread `cudaSetDevice` swaps, and `synchronize_devices`
  creating/destroying local events per call.
- Its multi-GPU path is effectively **never-run**: of the four MeshBarrierTest cases,
  three `GTEST_SKIP` on multi-GPU (SingleGpuBarrier/AsyncBarrier/IsBarriering) and
  `NoDeadlock` runs the event-poll loop from a **single host thread** only.
- The R15→R17 pattern holds: every never-run multi-GPU surface has hidden real bugs.

## Model (graph delta this round)

- `DEC-001` (decision-v18-meshbarrier-nccl) — milestone direction (P1 test harness →
  P2 converge on NcclBarrier → P3 harness flake/context-poisoning robustness).
  `governs` comp-dist-ops.
- `TASK-001` (P1, 9.0) — per-rank MeshBarrier test harness; `addresses`
  issue-v17-meshbarrier-multigpu.
- `TASK-002` (P2, 9.0) — converge onto NcclBarrier; `depends_on` TASK-001.
- `TASK-003` (P3, 12.0) — harness robustness (flake + R16 HIGH-B); `depends_on` TASK-002.

## Select

`TASK-001` (P1) first — unlocks P2/P3.

## Execute — Phase 1 (TASK-001)

Added two **per-rank thread-per-rank multi-GPU MeshBarrier tests** to
`tests/distributed/distributed_ops_test.cu`, asserting the *rendezvous semantics*
a barrier must have: a rank must not be released until every rank has reached the
barrier. Each rank sets `reached[d]=true` (deliberately staggered — rank d sleeps
d*15ms) immediately before entering; after the barrier call returns each rank
asserts every `reached[]` flag is set, then marks `released[d]`:

- `MultiGpu_BarrierRendezvous` — `MeshBarrier::synchronize()` driven per-rank.
- `MultiGpu_SynchronizeDevicesRendezvous` — `synchronize_devices()` over the full
  mesh, same check.

Committed as `DISABLED_MultiGpu_*` in the P1 checkpoint (P2 acceptance tests,
same convention as R17 P1).

## Verify — Phase 1 (evidence)

On 2x A100 both new tests **FAIL RED deterministically** (rank 0 released before
rank 1 reached), confirming the milestone hypothesis (`HYP-001` / `EV-001`): the
per-instance host event-poll recorded events on empty internal streams that fired
immediately — no cross-rank arrival signal.

## Execute — Phase 2 (TASK-002)

Converged `MeshBarrier` onto the verified `NcclBarrier` layer (`CHG-002`, 0a12a84),
preserving the public API + single-GPU fallback:

- `synchronize()` — blocking per-rank NcclBarrier (completes only when every rank
  entered); single-GPU fallback `cudaDeviceSynchronize`.
- `synchronize_async(stream)` — NCCL barrier ordered on the caller's stream
  (pre-v2.18 ignored it); `barriering_` resets on success.
- `synchronize_devices()` — full NCCL group only; a proper subset (incl. single
  device on multi-GPU) throws NcclException (no sub-communicators in NcclContext).
- Dropped the per-instance events_/streams_ (constructor no longer creates them);
  `NoDeadlock` made genuinely per-rank on multi-GPU.

## Verify — Phase 2

2x A100: `MultiGpu_BarrierRendezvous` / `MultiGpu_SynchronizeDevicesRendezvous` /
`MultiGpu_SynchronizeAsyncRendezvous` / `MultiGpu_SynchronizeSubsetThrows` +
per-rank `NoDeadlock` → **5/5 green** (was RED at ~1s before; now 15–40ms).
Single-GPU MeshBarrier fallbacks 4/4. Cross-suite 43/43, full suite 1460/1423/0.
cpp-review: **approve** (MEDIUM #1/#2/#3 addressed: subset-throws contract,
async flag reset + new async test).

## Execute — Phase 3 (TASK-003)

Harness robustness + R16 HIGH-B context poisoning (`CHG-003`, 7914b34):

- `NcclContext::mark_comm_aborted()` — the error layer (safe_nccl_call /
  safe_stream_wait) now flags a communicator it aborts on the owning context:
  the comm is nulled (destroy() never double-frees an aborted NCCL comm = UB),
  `broken_` set, `has_nccl()` false → later collectives fail fast instead of
  silently reusing a dead comm and hanging. `destroy()` + fresh `initialize()`
  is the documented recovery. Threaded `&ctx_` through all six collectives.
- Abort-only flagging (never on immediate synchronous caller errors, so
  deliberate negative tests cannot poison the shared singleton).
- New shared `distributed_test_common.h` — single home of the thread-per-rank
  harness (RankBarrier + run_per_rank), replacing four copies; barrier now
  BOUNDED (120s) → a dead rank is a diagnosed `RankBarrierTimeout`, not an
  unkillable >120s scale-hang.
- Distributed suites' readiness gate self-heals a broken singleton.

## Verify — Phase 3

- `AbortedCommPoisoningFailFast` (2x A100): abort → has_nccl false/broken true →
  collective fails fast → destroy+reinit recovers — **1/1**.
- 2-GPU cross-suite 49/49-ish (14 env-skips), **5x + 3x consecutive distributed
  ops stress runs stable (10/10 each)**; the original one-off stall never
  reproduced and its two mechanisms are now bounded failures.
- Full suite **1461/1423/0**. cpp-review: **approve** (M1 self-healing gate +
  L1 faithful abort test addressed).

## Learn — Milestone v2.18 close (graph)

- `TASK-001/002/003` resolved (CHG-001 c5f302a, CHG-002 0a12a84, CHG-003 7914b34).
- `issue-v17-meshbarrier-multigpu` resolved; `HYP-001` validated by `EV-001`,
  post-convergence evidence `EV-002`.
- `issue-v17-dist-ops-harness-flake` resolved by disposition `DEC-002` — the
  flake's two root-cause mechanisms (dead-rank hang, dead-comm poisoning) are
  both now bounded, diagnosed failures; the stall itself never re-observed.
- **Milestone v2.18 closed**: 3/3 phases green, no active task above threshold.
