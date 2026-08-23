# RIL — Round 30 (2026-08-23) — milestone v2.30 "Device-Native Cross-Entropy Loss Reduction"

## Round focus

**Open milestone v2.30** — the next installments of the host-round-trip removal
family (v2.27 optimizer → v2.29 SDPA/LN → **v2.30 the loss scalar**). The v2.29
close left one big D2H copy on the training hot path: `MicroTrainer::train_step`
and `TransformerTrainer::train_step`/`evaluate` copy the **whole m×hidden
logits buffer** to host every step (`logits_->copy_to`) and run the host
`cross_entropy_loss` loop (O(m×C) exp+log per step) just to produce a scalar
mean loss. `evaluate` additionally ran a host argmax loop for accuracy. The
actual forward/backward/optimizer are all on device; only this reporting
scalar still round-trips. v2.30 computes that scalar on-device and reads **one
float** back (TASK-047, DEC-016).

Also: environment note — since v2.29 closed, ALL 8 GPUs became free again
(external load released); the v2.28 "~20 memory-heavy tests OOM with ~5.8GB
free/GPU" full-suite gap can now be re-tested.

## Observation / Model

- **M1 (binding)**: the loss round-trip is the last whole-buffer host copy on
  the training path. Per step: `d_logits_->copy_to(h_logits, m*hidden)` (a
  blocking D2H of the full logits) + host `cross_entropy_loss` O(m×C) exp/log +
  `evaluate`'s separate host argmax loop. Grows with batch×seq×hidden like the
  rest of the stack — the family v2.27/v2.29 already removed everywhere else.
- **Convention (mirrors v2.27/v2.29)**: keep the public host `cross_entropy_loss`
  API (still used by host callers + host reference paths/tests); add
  `cross_entropy_loss_device` (device logits/targets → scalar in a [1] buffer,
  caller copies ONE float back) + `accuracy_device` (device top-1 count, [1])
  so `train_step` AND `evaluate` stay fully on device. Verify by device-vs-host
  fp64 parity + unchanged K-step shard==single-GPU trajectory parity.
- **Parity tolerance convention**: 1e-4 (fp32 reduction-order drift, same as the
  v2.29 kernel-parity convention; the float atomicAdd ordering across row blocks
  only perturbs the final sum at ~ulp level).

## Decisions

- **DEC-016**: milestone direction — device-native cross-entropy-loss reduction
  behind the same public `cross_entropy_loss` API: per-row max-subtracted
  softmax target log-probability, block reduction of the row max/sum, row
  losses atomicAdded into a pre-zeroed one-float scalar → the training drivers
  read one float back. `accuracy_device` does the same for evaluate()'s top-1.
  Verified by device-vs-host-fp64 parity + unchanged 2/4-GPU K-step trajectory
  parity. Rejected: GPU-side keep-host-loop (still a blocking m×hidden D2H
  every step — the point is removing the copy, not just the arithmetic);
  fusing loss into cross_entropy_logits_backward (couples the seed-gradient and
  reporting paths; loss must also work for evaluate()); CUDA-graph-capturing
  the trainer (bigger launch-overhead task, deferred — the copy dominates at
  this size); changing the public cross_entropy_loss signature (public host API
  must stay).

## Milestone definition (opened)

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: `cross_entropy_loss_device` vs in-test fp64 reference (batch=6/classes=10 mean; single-row edge; sum reduction; target-distinction sanity); RED against throw-stub | Complete (Round 30) — 4/4 RED (EV-028); GREEN with P2 (EV-029) |
| 2 | Implement: `loss_functions.cu` gains `ce_loss_rows_kernel` (per-row block max/exp-sum reductions, scaled atomicAdd into pre-zeroed scalar) + `accuracy_rows_kernel` (argmax count); route MicroTrainer + TransformerTrainer `train_step`/`evaluate` through them, drop the full `logits_->copy_to` + host loop | Complete (Round 30) — CHG-022 (pending commit); 4/4 loss contracts GREEN + `DeviceAccuracyMatchesHostArgmax` GREEN (EV-029/EV-031) |
| 3 | Verify: device-loss parity GREEN; neural single-GPU regression GREEN (97/97 incl. untrusted suites); multi-GPU cross-suite + deep-block K-step trajectory parity GREEN on 2 & 4 GPUs (5/5 on 2, 19/19 on 4) (EV-030); cpp-reviewer pass; RIL close | In progress (Round 30) |

**Graph additions (Round 30, opening):**
- `DEC-016` (decision, milestone direction); `TASK-047` (umbrella) /
  `TASK-048` (P1) / `TASK-049` (P2) / `TASK-050` (P3);
  `issue-v30-ce-loss-host-roundtrip` (the addressed debt); `EV-028` (P1 RED) /
  `EV-029` (P2 + GREEN) / `EV-030` (multi-GPU 2&4) / `EV-031` (accuracy parity).

## P1 RED (TASK-048, EV-028)

- Added the contract surface with a provisional throw-stub to
  `loss_functions.h/.cu` (`cross_entropy_loss_device`: device logits/targets →
  scalar in `loss_out[1]`), and 4 new `LossFunctionsTest.DeviceCrossEntropyLoss*`
  tests in `loss_functions_test.cpp` vs a host fp64 reference:
  - `MatchesFp64Reference` (batch=6, C=10, mean, ~N(0,2) logits);
  - `SingleRow` (batch=1, C=8, fixed logits);
  - `SumReduction` (reduction_mean=false, raw summed loss);
  - `DistinguishesTargets` (max-class target → small loss, min-class → large).
- RED run (EV-028): 4/4 fail for the right reason — all throw
  "cross_entropy_loss_device not implemented (v2.30 P1 stub)". 84/84 neural
  single-GPU baseline GREEN before this change.

## P2 implementation (TASK-049, EV-029)

- `loss_functions.cu`: `ce_block_reduce_max/sum` (lane-collapse helper pair with
  the v2.29 attention_kernels entry-`__syncthreads` convention — load-bearing
  against the cross-call shared race the v2.29 reviewer found); `ce_loss_rows_kernel`
  (one block per row: block-reduce the row max, then the row exp-sum, thread 0
  `atomicAdd`s `-log_softmax(target) * (1/batch | 1)` into the pre-zeroed scalar);
  `accuracy_rows_kernel` (per-row argmax == target → `atomicAdd` the count).
  Wrappers `cross_entropy_loss_device` / `accuracy_device` zero the scalar with
  `cudaMemsetAsync` (same stream, ordered before the kernel), launch, check
  `cudaGetLastError()`. No per-call device allocation (the initial row_loss
  Buffer design was dropped in favor of atomics — the v2.29 allocation-debt
  convention).
- `MicroTrainer` (training.cpp) + `TransformerTrainer` (training_transformer.cpp):
  `train_step`/`evaluate` route the reported loss through
  `cross_entropy_loss_device` (one float read back); `evaluate` uses
  `accuracy_device` for top-1. The full `logits_->copy_to(host)` + host
  `cross_entropy_loss` loop + host argmax loop are deleted
  (issue-v30-ce-loss-host-roundtrip resolved). The only remaining host copies on
  the training path are the targets upload and the single scalar readback.
- GREEN (EV-029/EV-031): the 4 P1 loss contracts 4/4 (max_abs ~1e-7 vs fp64);
  new `DeviceAccuracyMatchesHostArgmax` 1/1 (host argmax count == device count,
  incl. tie and all-equal rows); MicroTrainer still descends + accuracy rises.

## P3 verify (TASK-050, EV-030)

- Device-loss parity GREEN: the 4 P1 loss contracts + DeviceAccuracyMatchesHostArgmax
  all pass (EV-029/EV-031); MicroTrainer loss-descent + accuracy-rise unchanged.
- Neural single-GPU regression GREEN: 97/97 (LayerNormTrainable 8, AttentionKernels
  4, TransformerBlock, TensorParallelAttention, TensorParallelLayers, MicroTrainer,
  Optimizers 16, loss incl. 5 device contracts, activation/softmax/matmul/LN).
- Multi-GPU cross-suite + deep-block: 5/5 on 2 GPUs and 19/19 on 4 GPUs (EV-030) —
  MicroTrainer shard==host-fp64 full-weight trajectory + TP attention fwd/bwd +
  mini-transformer + DeepBlockMultiGpuPinnedTest UNCHANGED through the device loss.
  Residual multi-GPU suites 24/24 on 2 GPUs.
- Full-suite baseline (all 8 GPUs free, ctest -j8): 1603/1618 passed (99%);
  15 failures are -j8 device-0 OOM contention in memory-heavy inference/quant
  suites (BlockManagerCudaGraph, LongPrompt, ChunkedPrefill, BeamSpeculative, one
  FP8 benchmark) — each sampled failure passes in isolation (EV-032); pre-existing
  env limitation much reduced from the v2.28 ~20-OOM gap.
- cpp-reviewer (EV-033): **APPROVE** — no CRITICAL/HIGH. MEDIUM M1 (accuracy
  count as float) addressed: exact-for-batch<2^24 documented at the API;
  MEDIUM M2 (coverage gaps) addressed: `DeviceAccuracyMatchesHostArgmax` +
  `DeviceCrossEntropyLossLargeClassCount` (C=1024 — exercises the
  multi-partial grid-stride reductions; all earlier cases were < block).
  LOW L1 (unchecked `cudaMemsetAsync` return) fixed with `CUDA_CHECK` in both
  wrappers; L3 (harmless bounds guard) kept as defensive symmetry with
  `accuracy_rows_kernel`; L2 (__expf vs expf, ~ulp inside 1e-4) and L4/L5
  (int overflow at INT_MAX dims; unvalidated targets = same trust boundary as
  the host code replaced) documented as no-change. Post-fix re-verify: neural
  single-GPU 99/99, multi-GPU 5/5 (2) + 19/19 (4). Milestone **closed**.

## Graph status while open (Round 30)

Active tasks: TASK-018 (pre-existing, 1.50 < 3.0). v2.30 tasks resolved,
issue-v30-ce-loss-host-roundtrip resolved (CHG-022), DEC-016 governs the
milestone.

## LEARN (Round 30 close) — next milestone candidate

With the loss round-trip gone, the training hot path is fully device-native but
still launches ~100+ kernels (and several NCCL AllReduces) per train_step with
host-side step orchestration between each phase — the residual launch-overhead
constraint. DEC-016 explicitly deferred "CUDA-graph-capturing the trainer"
because "the copy dominates at this size"; v2.30 removed the copy, so the graph
capture of a full device train_step (forward → loss → backward → AdamW) is the
natural next performance round — verify by graph-replay == eager K-step
trajectory parity (identical kernels, identical memory addresses: buffers are
already allocated once in ensure_scratch and reused). TASK-018 (NCCL P2P/TP
failure routing, 1.50) remains the only active non-milestone task below the
3.0 threshold. Env: all 8 GPUs free.
