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

## Execute — P1 (TASK-001, milestone v2.18)

Added two **per-rank thread-per-rank multi-GPU MeshBarrier tests** to
`tests/distributed/distributed_ops_test.cu`, asserting the *rendezvous semantics*
a barrier must have: a rank must not be released until every rank has reached the
barrier. Each rank sets `reached[d]=true` (deliberately staggered — rank d sleeps
d*15ms) immediately before entering; after the barrier call returns each rank
asserts every `reached[]` flag is set, then marks `released[d]`:

- `MultiGpu_BarrierRendezvous` — `MeshBarrier::synchronize()` driven per-rank.
- `MultiGpu_SynchronizeDevicesRendezvous` — `synchronize_devices()` over the full
  mesh, same check.

Both gated on the same `multigpu_nccl_ready()` predicate as the data-op tests so
they stay valid after P2 converges onto NcclBarrier (which requires NCCL). In the
P1 checkpoint they are committed as `DISABLED_MultiGpu_*` (the P2 acceptance
tests, same convention as R17 P1).

## Verify — P1 (evidence)

On 2x A100 (`CUDA_VISIBLE_DEVICES=0,1`, `NCCL_TESTS_AVAILABLE=1`), both new tests
**FAIL RED deterministically** (re-run twice, ~1s / 15ms), confirming the milestone
hypothesis:

| test | observed | root cause |
|------|----------|------------|
| `MultiGpu_BarrierRendezvous` | rank 0 released before rank 1 reached | `synchronize()` records an event per device on internally-created (empty) streams and host-polls — those events fire immediately, so the "barrier" returns as soon as its *own* thread's events complete, with no cross-rank arrival signal (`HYP-001` / `EV-001`) |
| `MultiGpu_SynchronizeDevicesRendezvous` | rank 0 (synchronize_devices) released before rank 1 reached | same premature release |

Checkpoint regression on 2 GPUs: **6 passed / 10 env-skipped / 2 disabled, EXIT=0**
(no behavior change; harness-only). Single-GPU unaffected.

## Learn — P1 (graph)

- `HYP-001` — legacy MeshBarrier multi-GPU path does not enforce cross-rank arrival
  (events on empty streams fire immediately).
- `EV-001` (confidence 1.0) — the two RED failures above; `validates` HYP-001.
- `change-v18-p1` (this round) implements `TASK-001`.
- P2 (`TASK-002`) now unambiguous: route `MeshBarrier::synchronize()` /
  `synchronize_devices()` (and the async variant) onto `NcclBarrier` with
  `current_comm` routing, keep the single-GPU fallback, flip the two disabled
  tests on. `synchronize_async` must also stop ignoring its stream.
