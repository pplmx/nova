# RIL — Round 14 (2026-08-11) — implement non-continuous (static) batching

## Task: task-scheduler-noncontig-mode (was ~4.5, active) — RESOLVED

## Observation
`SchedulerConfig.enable_continuous_batching=false` was a **silent no-op**:
- `add_request()` queues every sequence into `pending_requests_` (Scheduler::add_request).
- In non-continuous mode `step()` did nothing and `get_batch()` returned `active_batch_`
  without recomposition → `get_batch()` always empty, requests stuck in pending forever.
- The only coverage (`SchedulerEdgeTest.NonContinuousBatchingMode`) was an **unconditional
  GTEST_SKIP** ("step() does not move pending requests to active batch"), i.e. a self-admitted
  gap.

This is a real correctness bug for any consumer that sets the flag (empty batch = no progress),
in the same "skip hides a real bug" class as Round 13's SyncBatchNorm discovery.

## Decision
Implement static-batching **cohort** semantics (decision-static-batching):
- `step()`/`get_batch()` prune finished/evicted sequences, then compose a fresh cohort from
  `pending_requests_` **only when the active batch has fully drained**.
- Slots freed by mid-generation completions are NOT refilled (that is continuous-mode
  behaviour) — new requests wait for the next batch.
- Refactored the shared prune loop into `Scheduler::prune_completed()` (no behaviour change to
  the continuous path).
- Alternatives rejected: (a) simply calling `recompose_batch()` in the non-continuous branch —
  that would make the flag meaningless (identical to continuous, just gated);
  (b) leaving the skip — masks a broken feature flag.

## Verification
- New test `SchedulerEdgeTest.NonContinuousBatchingMode` (replaces the skip): first cohort capped
  at `max_batch_size`, no mid-cohort refill after a completion, fresh batch after the cohort
  drains. Runs and passes on GPU 2.
- `SchedulerEdgeTest.*:SchedulerTest.*:IntegrationTest.*` → 46/46 pass.

## Also in this round: task-cusparse-exit-dtor (was ~3.2, active) — RESOLVED
`nova::sparse::detail::CusparseContext::get()` was a function-local static whose destructor calls
`cusparseDestroy` at process exit — the same latent exit-crash class that SIGSEGV'd inside libcuda
for NcclContext (R12) and MeshStreams (R13). No crash was observed (sparse suites exited 0), but
the fix is the established, tiny, consistent one: heap-allocate the singleton and never destroy it
(fix(sparse) 1b817df). Prevents a rare 1GB-core class rather than waiting for evidence.

## Final full-suite verification (GPU 2)
- **1420 pass / 0 fail / 29 skip / 1 disabled, EXIT=0** (round-13 baseline 1419/0/30 skip).

## Graph delta
nodes:
  - id: task-scheduler-noncontig-mode
    type: task
    status: resolved
  - id: change-r14-static-batching
    type: change
    status: active
  - id: task-cusparse-exit-dtor
    type: task
    status: resolved
  - id: change-r14-cusparse
    type: change
    status: active
edges:
  - from: change-r14-static-batching
    type: resolves
    to: task-scheduler-noncontig-mode
  - from: change-r14-cusparse
    type: resolves
    to: task-cusparse-exit-dtor
