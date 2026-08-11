# RIL — Round 17 (2026-08-11) — milestone v2.17 "Distributed Ops On Real Multi-GPU"

## Round focus

**Open milestone v2.17** and execute **Phase 1**: build the thread-per-rank multi-GPU
harness for the high-level distributed ops (`DistributedReduce`, `DistributedAllGather`,
`DistributedBroadcast`, `MeshBarrier`) and convert their "Requires single GPU" test
skips into genuine multi-GPU tests — reproducing the R15/R16 pattern that keeps
converting the repo's largest remaining never-run surfaces into running, asserted code.

## Observation (evidence)

- `tests/distributed/distributed_ops_test.cu` has **10 tests that skip with
  `GTEST_SKIP() << "Requires single GPU"` whenever `device_count() > 1`**, and the file's
  header comment itself defers the work: "Multi-GPU tests require a proper collective test
  harness (all GPUs call the operation simultaneously)." Exactly the deferral that v2.16
  resolved for distributed matmul.
- These are **not dead APIs**: `src/cuda/neural/sync_batch_norm.cu` calls
  `distributed::DistributedReduce::all_reduce_async` at **3 sites** when
  `device_count() > 1` (forward -running stats, backward gamma/beta all-reduce), so a real
  training path depends on an untested multi-GPU implementation.
- The legacy multi-GPU implementations are structurally suspect:
  - `DistributedReduce::all_reduce` (`src/cuda/distributed/reduce.cu`) coordinates via a
    CPU host-staging loop that `cudaSetDevice`-swaps across **all** GPUs while reading the
    caller's `send_data`/`recv_data` pointer (valid only on the calling device) — broken
    when driven per-rank.
  - `DistributedReduce::all_reduce_async` is `// Delegate to sync version for simplicity`
    — a blocking host-staged op that **ignores its `stream` argument**.
- RIL R16 review HIGH-B (context poisoning) is tracked under this milestone: a rank that
  exits a collective early leaves healthy ranks blocking in `safe_nccl_call` and then
  `ncclCommAbort`s the process-wide communicators, poisoning `NcclContext` for later users.

## Root cause

The "requires a proper collective test harness" deferral was copied forward from the
matmul/infra era; v2.16 proved the harness exists (thread-per-rank in one process, exactly
the NCCL suite's pattern) and that the NCCL layer is now verified. The legacy ops were
written before that layer was real and never re-verified. The multi-GPU correctness of the
high-level ops — and of the SyncBatchNorm training path that consumes them — is unknown.

## Model (graph delta this round)

- `comp-dist-ops` — high-level distributed ops component.
- `issue-v17-dist-ops-multigpu-untested` (sev 4) — multi-GPU paths never exercised;
  `located_in` comp-dist-ops.
- `decision-v17-dist-ops-real-multigpu` — milestone direction (P1 harness+evidence, P2
  converge on verified NCCL layer or fix legacy per evidence, P3 SyncBatchNorm gradient
  verify; HIGH-B readiness-exchange in scope).
- Tasks (all addresses the issue; P2 depends on P1, P3 depends on P2):
  - `task-v17a-dist-ops-harness` (21.60) — P1 harness + real multi-GPU tests.
  - `task-v17b-dist-ops-nccl-converge` (20.82) — P2 converge/fix per P1 evidence.
  - `task-v17c-syncbn-multigpu-backward` (14.31) — P3 SyncBatchNorm gradient verify.
- Milestone v2.16 closed at Round 16 (R16 close-out commits a1f438d, b57184d).

## Select

`task-v17a-dist-ops-harness` (P1) is the round 17 focus.

## Execute / Verify / Learn (P1)

(filled at round close)
