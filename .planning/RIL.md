# RIL — Repository Intelligence Layer (index)

Maintained by the autonomous engineering loop. Full guidance: `.agents/skills/graph-engineering/SKILL.md`
(single source; `.claude/skills` is a whole-directory symlink to `.agents/skills` so Claude Code
discovers it under the skills it scans).
Per-round narratives + graph deltas: `.planning/ril_autonomous_roundN.md` (latest: Round 24).
Milestone thread: v2.15 (Test Quality) closed at Round 14 → **v2.16 "Distributed Multi-GPU
Verification"** closed at Round 16 → **v2.17 "Distributed Ops On Real Multi-GPU"** closed at
Round 17 → **v2.18 "MeshBarrier On The Verified Layer + Distributed Robustness"** closed at
Round 18 → **v2.19 "Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism)"**
closed at Round 19 → **v2.20 "TensorParallelMatmul Production Hardening"** opened and
**closed at Round 20** → v2.21/v2.22/v2.23 closed at Rounds 21/22/23 → **v2.24
"End-to-End Training Step On The Tensor-Parallel Stack"** opened at Round 24 (P1 RED).

## Latest round
- **Round 27 (2026-08-13)** — milestone **v2.27 "Device-Native Optimizer Kernels (AdamW +
  Gradient Norm/Clip)"** opened (DEC-013, TASK-036/037/038). The optimizer stack
  (`optimizers.cpp`) is host-side: `AdamWOptimizer::step` D2H→host-loop→H2D full copies
  (9 per MicroTrainer step), `compute_gradient_norm`/`clip_gradients` the same. The v2.25
  cpp-review M2 flagged this as the binding training-perf constraint, deferred by
  DEC-010/011/012. Milestone moves it to device kernels behind the same API (parity is
  exact-element). See `ril_autonomous_round27.md`.
- **Round 26 (2026-08-13)** — milestone **v2.26 "Tensor-Parallel Multi-Head Attention +
  Mini-Transformer Training"** opened through **close (P1+P2+P3)** (DEC-012,
  TASK-032/033/034, CHG-015 bc02049). Added `TensorParallelMultiHeadAttention` (QKV
  column-parallel → collective-free multi-head SDPA → output row-parallel; host-side
  `sdpa_forward/backward`; backward per v2.23 with grad-input AllReduce; `step()` one
  AdamW per weight tensor) — resolving the incomplete single-GPU MultiHeadAttention shell
  (issue-v24-mha-incomplete). Mini-transformer (attention + verified MLP) trains sharded.
  Test-reference fixes (position-major key/v stride; per-head softmax probs) came out of
  reference parity. Verified (EV-020): single-GPU 3/3 (host fp64 fwd/bwd + convergence
  loss 106.6→1.35, acc 0.06→0.56); multi-GPU 3/3 on 2 & 4 GPUs incl. K=12
  mini-transformer shard==single-GPU; cross-suite 43/43 on 2 & 4 GPUs; single-GPU neural
  43/43. Milestone **closed**. See `ril_autonomous_round26.md`.
- **Round 25 (2026-08-13)** — milestone **v2.25 "MicroTrainer: End-to-End Gradient
  Training Convergence On The Tensor-Parallel Stack"** opened (DEC-011,
  TASK-028/029/030) and P2 implemented (CHG-014 cc35094). Added
  `training::MicroTrainer` — owns the MLP + one AdamW per weight tensor +
  device scratch; `train_step` chains forward → device CE-logits backward →
  MLP backward → per-shard AdamW and returns the host mean CE. **Found & fixed
  `issue-v24-ce-loss-running-sum`**: `cross_entropy_loss` used a running-sum
  softmax denominator (target at class c got the partial sum c'≤c), under-
  reporting loss for early targets — masked in v2.24 by relative-only loss
  assertions; exposed by the trajectory parity (dev 2.18 vs host fp64 3.91 on
  identical logits). Verified (EV-019): single-GPU MicroTrainer converges; K=10
  multi-GPU sharded run == host fp64 full-weight trajectory (weights + loss
  curve) GREEN on 2 & 4 GPUs; cross-suite 40/40 on 2 GPUs; single-GPU neural
  45/45. The stack now demonstrably learns. See `ril_autonomous_round25.md`.
- **Round 24 (2026-08-13)** — milestone **v2.24 "End-to-End Training Step On The
  Tensor-Parallel Stack"** opened through **close (P1+P2+P3)** (`DEC-010`,
  `TASK-024/025/026`, `CHG-013` 3053189). The v2.21-23 layers were forward+backward
  capable but untrainable: `loss_functions.cu` was host-side with a forward-only scalar
  (its device kernels were dead code), the `weight_` shards had no optimizer-step surface,
  and no loop chained forward → loss → backward → optimizer. **P1** pinned 5 RED tests. **P2**
  implemented the device `cross_entropy_logits_backward` kernel (dlogits=(softmax-onehot)/B)
  and per-layer `step(AdamWOptimizer&, grad_weight_full, step_no)` applying AdamW in place
  to the private rank shard (strided column slice / contiguous row block extraction for the
  shard grad; one optimizer per weight tensor after the cpp-review HIGH — a shared
  instance's per-element `m/v` would corrupt up/down moments). **P3 verified**: K-step
  multi-GPU shard training == host fp64 full-weight reference GREEN on **2 & 4 GPUs**;
  isolated NCCL cross-suite **39/39 on 2 & 4 GPUs**; single-GPU neural regression 37/37;
  full-suite baseline 1426/0 (`EV-017/EV-018`). The stack is now trainable end-to-end.
  See `ril_autonomous_round24.md`.
- **Round 20 (2026-08-12)** — milestone v2.20 kickoff through **close (P1+P2)**. **P1**
  (`CHG-007` 2137cea): closed the cpp-review BLOCK on the v2.19 TP rewrite — rank resolved
  via `ctx_.rank_of_device()` (membership-validated, no raw-index rank), the column
  AllReduce `NcclResult` is checked and failures throw `NcclException`, RowParallel
  non-divisible test added, header docs corrected — TP+NCCL 20/20, acceptance 5/5 at
  4 GPUs (`EV-009`). **P2** (`CHG-008`): disposed the non-functional `TensorParallelLayers`
  skeletons (were null-B cuBLAS crashes / empty MLP forward) — now fail fast, 3/3 reject
  tests (`EV-009`), real implementation deferred (`DEC-006` supersedes `DEC-004`;
  `TASK-010`). P3 investigation: `issue-v19-shared-nccl-context-reset` auto-detection is
  infeasible (`cuCtxGetCurrent` reuses the address after reset, `EV-008`) — disposition is
  the documented curated multi-GPU workflow. Milestone **closed (2/2 phases)**. See
  `ril_autonomous_round20.md`.
- **Round 19 (2026-08-12)** — milestone v2.19 kickoff through **close (P1+P2+P3)**. **P1**
  (`CHG-004` 2006da8): per-rank TensorParallelMatmul multi-GPU tests FAIL RED (4/4) —
  ColumnParallel contiguous-slice offsets against the row-major matmul produce wrong output,
  `n % tp != 0` silently drops columns, RowParallel ignores the rank index (`HYP-002`/
  `EV-004`). **P2** (`CHG-005` 837a966): corrected the partition math on the verified NCCL
  layer — zero-padded strided column blocks computed via cuBLAS (`ldb=ldc=n`) + one
  block-wise AllReduce, rank-correct RowParallel, shape rejection, per-instance stream-bound
  cuBLAS handle (shared handle is not thread-safe), RAII stream guard — `MultiGpu_*`
  acceptance 8/8 GREEN on 2/4/8 GPUs, TP+NCCL suites 19/19 (`EV-005` refutes `HYP-002`).
  **P3** (`CHG-006` 2a93f52): `SequenceParallelAttention` driven with a real
  `NcclContext::current_comm()` per rank — all_reduce_sequence / gather_kv / scatter_output
  (ReduceScatter + slice-copy) 4/4 GREEN on 2/4 GPUs (collectives were already correct);
  `RingSequenceParallelism` multi-GPU no-op made fail-fast (650de28, `DEC-004`, `TASK-007`).
  New issues: `issue-v19-shared-nccl-context-reset` (full NCCL-enabled suite run can SEGV in
  libcuda `cuStreamDestroy` after interleaved `cudaDeviceReset` suites — pre-existing,
  order/timing-sensitive; R15-R18 never ran the NCCL-enabled full suite),
  `issue-v19-ring-parallel-noop` (closed by DEC-004). Host note: GPUs 0/1 externally loaded;
  use `CUDA_VISIBLE_DEVICES=2,3+`. Milestone **closed (3/3 phases)**. See
  `ril_autonomous_round19.md`.
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
- `TASK-010` (8.4, core-feature) — implement weight-managed ColumnParallelLayer /
  RowParallelLayer / TensorParallelMLP forward (replaces the DEC-006 fail-fast disposition).

## Resolved (v2.20)
- `TASK-008` (P1) RESOLVED by `CHG-007` (2137cea) — TP matmul closes the review BLOCK:
  rank via `rank_of_device()` (membership validation), collective `NcclResult` propagated,
  RowParallel non-divisible test, header docs corrected; TP+NCCL 20/20.
- `TASK-009` (P2) RESOLVED by `CHG-008` — `TensorParallelLayers` stubs fail fast (null-B
  cuBLAS crash / empty MLP forward); 3/3 reject tests.
- `issue-v20-tp-layers-stubs` RESOLVED (fail-fast disposition, DEC-006).
- `issue-v19-shared-nccl-context-reset` OPEN with disposition: curated multi-GPU workflow;
  auto-detection infeasible (`EV-008`).

## Resolved (v2.19)
- `TASK-004` (P1) RESOLVED by `CHG-004` (2006da8) — per-rank TensorParallelMatmul RED tests;
  `HYP-002` validated by `EV-004`.
- `TASK-005` (P2) RESOLVED by `CHG-005` (837a966) — TP matmul partition math corrected on the
  verified NCCL layer; acceptance 8/8 GREEN 2/4/8 GPUs; `EV-005` refutes `HYP-002`.
- `TASK-006` (P3) RESOLVED by `CHG-006` (2a93f52) — SequenceParallelAttention real-comm
  tests 4/4 GREEN on 2/4 GPUs.
- `TASK-007` RESOLVED by disposition `DEC-004` (650de28) — RingSequenceParallelism multi-GPU
  now fails fast (was a silent no-op: `send_recv_kv` declared-never-defined).
- `issue-v19-tp-multigpu-unverified` RESOLVED; `issue-v19-ring-parallel-noop` RESOLVED by
  `DEC-004`. `issue-v19-shared-nccl-context-reset` OPEN (tracked follow-up).

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
