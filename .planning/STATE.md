----
gsd_state_version: 1.0
milestone: v2.33
milestone_name: Multi-GPU Rank-Shard Checkpoint Restore
status: Complete
last_updated: "2026-08-24"
last_activity: 2026-08-24 — Round 33: P1 RED -> P2 shard setters / rank-aware save-load -> cpp-reviewer HIGH/MEDIUM fixes -> P3 verify (2/4/8 GPUs + full-suite) + RIL close, milestone closed
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 0
  completed_plans: 0
---

# Project State

**Project:** Nova CUDA Library Enhancement
**Last Updated:** 2026-08-24

## Current Position

Milestone: v2.33 Multi-GPU Rank-Shard Checkpoint Restore
Status: Complete (Round 33)
Last activity: 2026-08-24 — P1 RED (2 contracts vs the tp>1 guard) -> P2 layer
shard setters + rank-aware NSCK save/load -> cpp-reviewer HIGH (test geometry
on 8 GPUs) + MEDIUM (attention heads%tp) fixed -> P3 verify 2/4/8 GPUs +
1469/0 full-suite + RIL close; ISS-001 resolved

## Milestone v2.33 — Multi-GPU Rank-Shard Checkpoint Restore

v2.32 shipped exact trainer checkpointing but explicitly guarded tp>1: with
TP>1 the copy_* readbacks expose rank shards while set_* takes full weights, so
a rank-local checkpoint couldn't restore — the interruption-resume use case
checkpointing exists for. v2.33 removes the guard: no-slice shard uploaders
(ColumnParallelLayer/RowParallelLayer::set_weight_shard, then
TensorParallelMLP::set_weight_shards + TransformerBlock::set_weight_shards),
rank-aware save_state/load_state sizing (from dims/tp), geometry-validated
count-tagged restore onto an identically-geometried trainer (NSCK v2 now also
stores the TP degree, rejected on mismatch). Verified by byte-exact per-rank
roundtrip + resume-vs-uninterrupted on real 2/4/8-GPU trains
(checkpoint_multigpu_test.cpp); the tp=1 CheckpointTest suite stayed
byte-identical. cpp-reviewer: WARNING->GREEN after fixing a test-geometry HIGH
(heads scaled with device_count) + a latent attention ctor MEDIUM
(num_heads % tp enforced). Rank-identity (a same-topology file from another
rank is not detected) is documented caveat + TASK-063.

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: 2 multi-GPU checkpoint contracts (MicroTrainer rank-local roundtrip; TransformerTrainer per-rank) fail against the tp>1 guard | Complete (Round 33) — 2/2 RED (EV-041) |
| 2 | Implement: layer shard setters + set_weight_shards; rank-aware save/load sizing; drop tp>1 guards; load through shard setters | Complete (Round 33) — CHG-025 (69ae442); 2/2 GREEN on 2 GPUs + tp=1 8/8 (EV-043) |
| 3 | Verify: 2-GPU contracts GREEN; tp=1 CheckpointTest 8/8; neural single-GPU + multi-GPU regressions; cpp-reviewer; RIL close; ISS-001 resolved | Complete (Round 33) — cpp-reviewer WARNING fixed (CHG-026 e6ce721, EV-044); 2/2 on 2/4/8 GPUs; 1469/0 full-suite; close |

## Milestone v2.32 — Model Checkpointing (Weights + Optimizer State)

The host-round-trip family is complete (v2.27/29/30/31) and the train stack is
real (v2.25-28), but nothing persists: a trained MicroTrainer/TransformerTrainer
dies with the process, an interrupted run can't resume, and models can't be
exported (the memory-opt CheckpointCompressor is buffer compression, not model
serialization). v2.32 adds save_state(std::ostream&) / load_state(std::istream&)
to both trainers behind a deterministic binary format (NSCK v1: magic, version,
kind, dims, tensors count+float[], moments count+m[]+v[]), plus
AdamWOptimizer::{momentum_capacity, copy_moments_to, copy_moments_from} so a
restored trainer resumes byte-identical (AdamW reads m/v, so weights alone
diverge at the first resumed step). Verified at tp=1 (shard==full) by byte-exact
roundtrip + fresh-state roundtrip + resumed-vs-uninterrupted trajectory +
geometry/corruption rejection + no-partial-restore on rejected loads; tp>1
rank-shard restore is the named follow-up (TASK-059).

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: AdamW moment export/import + trainer save_state/load_state throw-stubs; CheckpointTest contracts (roundtrip byte-exact, fresh-state, resume-vs-uninterrupted, geometry/corrupt rejection); RED against stubs | Complete (Round 32) — 6/6 RED against stubs (EV-038); 26/26 baseline green before |
| 2 | Implement: AdamW moment API; CheckpointWriter/Reader (src/cuda/neural/checkpoint_io.h); MicroTrainer + TransformerTrainer save_state/load_state routing through copy_*/set_* + moment copies + validation | Complete (Round 32) — CHG-024 (EV-039): CheckpointTest 6/6, neural 113/113, multi-GPU 5/5 |
| 3 | Verify: CheckpointTest GREEN; neural single-GPU regression GREEN; cpp-reviewer; RIL close; tp>1 follow-up task opened | Complete (Round 32) — cpp-reviewer APPROVE + 3 MEDIUM fixed (CHG-025, EV-040); re-verified 8/8 + 115/115 + 5/5; full-suite baseline; RIL close |

## Milestone v2.31 — Device-Native LAMB Optimizer Kernel

The v2.27 optimizer round kernelized AdamW + norm/clip but left LAMB host-side
("LAMB keeps its host-side implementation" per the v2.27 header; DEC-014/015
deferred it). This closes the last optimizer D2H round-trip: every
`LAMBOptimizer::step` copied grads+params D2H, ran an O(n) host loop with
per-element `std::vector` m/v, and copied H2D back (optimizers.cpp). v2.31 adds
a fused `lamb_step_kernel` (bias-corrected m/v, trust-ratio update with the
host-computed phi_1/phi_2 as a scalar arg, per-element clamp) behind
`detail::lamb_step_device`; m/v become device buffers; the round-trip is gone.
Public API unchanged. The Round-30 LEARN candidate (CUDA-graph capture of
train_step) was empirically refuted first (EV-034: the null-stream / per-call
op-stream stack can't be captured without a wide stream-parametric rewrite),
so LAMB was chosen as the family-completion round.

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: detail::lamb_step_device vs host LAMB reference (exact-element over K steps; layer-adaptation on/off); RED against throw-stub | Complete (Round 31) — 2/2 RED (EV-035); 16/16 OptimizersTest baseline GREEN before |
| 2 | Implement: lamb_step_kernel (fused m/v + trust-ratio, rtw scalar arg) + detail::lamb_step_device; LAMBOptimizer m/v -> device Buffers; step() routes through it, D2H/H2D dropped | Complete (Round 31) — CHG-023; exact-element parity GREEN (EV-036); Optimizers+trainer 26/26 |
| 3 | Verify: parity GREEN over K steps; OptimizersTest 18/18; neural single-GPU 101/101; multi-GPU deep-block trajectory unchanged (2/2 on 2 GPUs); compute-sanitizer memcheck 0 / racecheck 0 (EV-037); cpp-reviewer; RIL close | Complete (Round 31) |

Decision: DEC-017 (also records the EV-034 CUDA-graph-capture refutation). Env:
all 8 GPUs free.

## Milestone v2.30 — Device-Native Cross-Entropy Loss Reduction

The v2.29 close left one big D2H copy on the training hot path: every
`train_step`/`evaluate` of `MicroTrainer` and `TransformerTrainer` copied the
whole m×hidden logits buffer back to host (`logits_->copy_to`) and ran the host
`cross_entropy_loss` O(m×C) loop just to produce a scalar mean loss; `evaluate`
also ran a host argmax loop for accuracy. Same D2H family v2.27 (optimizer) and
v2.29 (SDPA/LN) removed. This milestone computes the scalar on-device —
`cross_entropy_loss_device` (per-row max-subtracted softmax, block/warp
reduction, scaled atomicAdd of row losses into a pre-zeroed scalar) +
`accuracy_device` (per-row argmax atomic count) — behind the same public
`cross_entropy_loss` API, and routes both trainers through them, dropping the
round-trips (only the targets upload + single scalar readback remain).

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: cross_entropy_loss_device vs host fp64 reference (batch>1 mean, single-row, sum reduction, target-distinction); RED against throw-stub | Complete (Round 30) — 4/4 RED (EV-028); 84/84 neural baseline GREEN before change |
| 2 | Implement: ce_block_reduce_max/sum (v2.29 entry-barrier convention) + ce_loss_rows_kernel (per-row block max/exp-sum, scaled atomicAdd) + accuracy_rows_kernel; route MicroTrainer + TransformerTrainer train_step/evaluate through them | Complete (Round 30) — CHG-022; 4 loss contracts + DeviceAccuracyMatchesHostArgmax + LargeClassCount(C=1024) GREEN (EV-029/EV-031) |
| 3 | Verify: parity GREEN; neural single-GPU 99/99; multi-GPU cross-suite + deep-block 5/5 on 2 GPUs + 19/19 on 4 GPUs (unchanged; EV-030); full-suite baseline 1603/1618 (99%, 15 -j8 OOM contention pass in isolation, EV-032); cpp-reviewer APPROVE (M1/M2/L1 addressed, EV-033); RIL close | Complete (Round 30/32) |

Decision: DEC-016. Env: ALL 8 GPUs free again (external load released) —
full-suite gap since v2.28 (~20 OOM at ~5.8GB free) now 15 contention-OOMs.

## Milestone v2.29 — Device-Native SDPA + LayerNorm Training Kernels

Round-28 LEARN named "a device-native attention + LN training kernels
performance round on the v2.28 stack" as the next milestone. The v2.28 deep
transformer trains correctly but each train_step round-trips attention through
the host (`sdpa_forward`/`sdpa_backward`: 10 blocking D2H/H2D copies + host
loops over m×local_heads×m×head_dim per block; 30 blocking memcpys for a
3-block trainer — the same D2H/H2D family v2.27 removed from the optimizer,
deferred by the v2.26 header to "a later performance pass"), and the trainable
LayerNorm kernels launch one thread per row (M2). This milestone moves
scaled-dot-product attention forward+backward to device kernels behind the same
`sdpa_forward`/`sdpa_backward` API (a standalone `attention_kernels.cu` detail
module mirroring `optimizers_kernels.cu`), parallelizes the trainable LayerNorm
kernels with block/warp reductions, deletes the host round-trips + dead CPU
implementations, and verifies by GPU-vs-fp64-reference parity plus the existing
2/4-GPU shard==single-GPU K-step trajectory parity (unchanged) (TASK-043).

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: detail::sdpa_forward_device/sdpa_backward_device vs in-test fp64 SDPA reference (seq>1, heads>1, head_dim>1); parallelized trainable LayerNorm fwd/bwd (block-reduction path) vs fp64 reference at hidden∈{512,1024}, rows>1 | Complete (Round 30) — 6/6 RED against throw-stubs (EV-025) |
| 2 | Implement: attention_kernels.cu device SDPA forward + analytic backward (per-row shared-memory softmax, block reductions); wire sdpa_forward/sdpa_backward to device, delete host round-trip + CPU impls; parallelize ln_forward_trainable_kernel/ln_backward_stats_kernel with block/warp reductions and route LayerNorm::forward/backward + free layer_norm_backward through them | Complete (Round 30) — CHG-020 3af650a + CHG-021 fix 6a466db; 6/6 GREEN; max_abs ~3e-8 (SDPA) / ~5e-7 (LN) (EV-026/027) |
| 3 | Verify: kernel parity GREEN; all existing neural single-GPU suites GREEN; neural multi-GPU cross-suite 19/19 on 2 & 4 GPUs (deep-block trajectory unchanged); cpp-reviewer pass; RIL close | Complete (Round 30/31) — neural single-GPU 79/79; multi-GPU + deep-block 19/19 on 2 & 4 GPUs (EV-027); cpp-reviewer HIGH (reduction-buffer race, racecheck-verified) + MEDIUM (int overflow) fixed (CHG-021 6a466db) |

Decision: DEC-015. Host note: use `CUDA_VISIBLE_DEVICES=2,3+`; all 8 GPUs
~76GB used — re-check nvidia-smi before multi-GPU runs.

## Milestone v2.28 — Normalized Transformer Blocks (LayerNorm + Residual) + Deep Multi-Layer Training

Round-27 LEARN named "a deeper/multi-layer transformer" as the next capability.
Today the stack trains exactly ONE model — an unnormalized single block
(attention → MLP, logits = MLP(attn(X))): the `layer_norm` module is a
forward-only orphan (no backward → untrainable, never validated against a
reference), no residual connection exists, and no model stacks more than one
block (TASK-039). This milestone adds the normalization+residual foundation and
proves a deep N-block transformer trains sharded: device LayerNorm forward AND
backward with trainable gamma/beta (collective-free — block activations are
replicated, so per-row norm is identical on every rank; the only comm in the
block stays the output projection's AllReduce), a pre-LN residual
`TransformerBlock` (LN→MHA→+x→LN→MLP→+x, one AdamW per weight tensor incl.
each LN gamma/beta), and an N-block stack verified by single-GPU convergence
and 2/4-GPU shard==single-GPU full-weight parity.

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: device LayerNorm forward vs host-fp64 reference + analytic backward (d_x, d_gamma, d_beta); pre-LN residual TransformerBlock fwd/bwd vs host-fp64 block reference; deep N-block convergence + K-step multi-GPU trajectory parity on 2 & 4 GPUs | Complete (Round 28) — 7/7 device contracts RED against throw-stubs; host-fp64 LN backward self-checked vs central finite differences (EV-022) |
| 2 | Implement: device LayerNorm backward kernels + affine gamma/beta (making the forward-only orphan trainable); TransformerBlock (pre-LN attention + pre-LN MLP, residual adds, one AdamW per weight tensor incl. LN gamma/beta); N-block stack + training loop on the MicroTrainer conventions | Complete (Round 28) — CHG-017 cde8ad7; LayerNormTrainableTest 5/5 + TransformerBlockTest 3/3 GREEN (EV-023) |
| 3 | Verify: LN parity GREEN; single-GPU N-block convergence; multi-GPU K-step N-block shard==single-GPU parity GREEN on 2 & 4 GPUs; cross-suite + regression + full baseline; cpp-reviewer; RIL close | Complete (Round 28) — 2/4-GPU cross-suite 19/19 each; cpp-reviewer: C1 (block seq>1 rows) + L2 (inference null mean/var) fixed w/ regression tests (EV-024); full-suite baseline blocked by env memory (~5.8GB free/GPU, 20 OOM in memory-heavy non-neural suites — not v2.28) (EV-023) |

Decision: DEC-014. Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.27 — Device-Native Optimizer Kernels (AdamW + Gradient Norm/Clip)

## Milestone v2.27 — Device-Native Optimizer Kernels (AdamW + Gradient Norm/Clip)

Move the optimizer stack (AdamW update, gradient L2/Inf norm, gradient
clipping) from host-side D2H→host-loop→H2D into CUDA kernels behind the SAME
public API — AdamW m/v become device buffers and the per-element
bias-corrected update is one fused kernel; norm/clip use reductions. Every
caller (layer step(), MicroTrainer, tests) accelerates without edits; parity is
exact-element (kernel == host formula), so the existing host-fp64 trajectory
tests re-attest (TASK-035).

| Phase | Name | Status |
|-------|------|--------|
| 1 | Parity tests: device AdamW over K steps == host formula (exact-element); device norm L2/Inf == host; device clip == host scale | Complete (Round 27) — 3 device-vs-host parity tests GREEN (EV-021) |
| 2 | Implement device AdamW (m/v device buffers + fused kernel), compute_gradient_norm reduction, clip scale on device; API unchanged | Complete (Round 27) — CHG-016 711d511; optimizers_kernels.cu |
| 3 | Verify: parity GREEN; K-step multi-GPU MicroTrainer + TP training still GREEN on 2 & 4 GPUs; cross-suite + regression; RIL close | Complete (Round 27) — cross-suite 43/43 on 2 & 4 GPUs; single-GPU 46/46; full-suite 1435/0 (EV-021) |

Decision: DEC-013. Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.26 — Tensor-Parallel Multi-Head Attention + Mini-Transformer Training

The DEC-010/011-rejected attention capstone, now that MicroTrainer (v2.25)
makes it trainable. Build attention on the verified v2.21-25 layer conventions:
QKV ColumnParallelLayer (each rank owns contiguous head columns -> per-head
SDPA is collective-free), multi-head SDPA (QK^T/sqrt(d) softmax V + analytic
backward), output RowParallelLayer (one AllReduce). step() takes one AdamW per
weight tensor. Mini-transformer = attention + verified MLP, logits =
MLP(attn(X)), trained via the MicroTrainer chain.

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED tests pin the TP-attention contracts (forward/backward host fp64 reference; mini-transformer convergence) | Complete (Round 26) — contracts pinned during P2 |
| 2 | Implement TensorParallelMultiHeadAttention + sdpa forward/backward + mini-transformer training path | Complete (Round 26) — bc02049; GREEN on single-GPU |
| 3 | Verify: multi-GPU assembled forward/backward + mini-transformer shard==single-GPU on 2 & 4 GPUs; cross-suite + regression | Complete (Round 26) — 3/3 multi-GPU on 2 & 4; cross-suite 43/43; single-GPU 43/43 (EV-020) |

Decision: DEC-012. Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.25 — MicroTrainer: End-to-End Gradient Training Convergence On The Tensor-Parallel Stack

Turn the v2.24 verified building blocks into a reusable training loop and prove
the parallel stack actually learns (TASK-027, DEC-011): a
`training::MicroTrainer` owns the MLP + one AdamW per weight tensor + device
scratch; the acceptance is real convergence — loss descent + accuracy rise on
single-GPU, and a full K-step multi-GPU sharded run reproducing the host fp64
full-weight training trajectory (weights AND loss curve) on 2 & 4 GPUs.

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED tests pin the MicroTrainer contracts (train_step returns mean CE loss and steps shards; loss descent; accuracy rise; multi-GPU trajectory parity) | Complete (Round 25) — contracts pinned during P2 development |
| 2 | Implement training.h/.cpp MicroTrainer (forward → device CE-logits backward → MLP backward → per-shard AdamW); fix cross_entropy_loss running-sum denominator bug | Complete (Round 25) — cc35094; issue-v24-ce-loss-running-sum fixed (dev loss == host fp64) |
| 3 | Verify: loss descent + accuracy single-GPU; multi-GPU K-step trajectory parity GREEN on 2 & 4 GPUs; cross-suite + regression; RIL close | Complete (Round 25) — parity GREEN on 2 & 4 GPUs (EV-019); cross-suite 40/40 on 2 GPUs; single-GPU 45/45 |

Decision: DEC-011. Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.24 — End-to-End Training Step On The Tensor-Parallel Stack

Make the v2.21-23 forward+backward layers actually *trainable* on the parallel
path: a device cross-entropy-with-logits backward seeds the chain (the loss
layer today is host-side with a forward-only scalar and dead device kernels),
and each layer gains `step(AdamWOptimizer&, grad_weight_full, step_no)` that
applies AdamW in place to the private rank shard. Acceptance: K-step multi-GPU
shard training assembled == host fp64 full-weight reference (the meshed
gradient layout makes this exact), on 2 & 4 GPUs (TASK-023).

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED tests pin the training contracts: device cross_entropy_logits_backward vs host fp64 analytic; single-GPU col/row/MLP step vs host AdamW over one analytic step; multi-GPU K-step shard parity vs host fp64 reference on 2 & 4 GPUs | Complete (Round 24) — 5/5 RED against provisional throw-stubs (EV-017) |
| 2 | Implement: cross_entropy_logits_backward device kernel (max-subtract softmax - onehot, /B); per-layer step() extracting the rank shard grad (strided column slice / contiguous row block) and applying AdamW to the private weight_ shard; TensorParallelMLP::step chains | Complete (Round 24) — CHG-013; 4 single-GPU training tests GREEN (EV-018) |
| 3 | Verify: multi-GPU shard-step parity GREEN on 2 & 4 GPUs, single-GPU regression green, full-suite baseline, RIL close | Complete (Round 24) — parity GREEN on 2 & 4 GPUs; cross-suite 39/39 on 2 & 4 GPUs; single-GPU regression 32/32; full-suite baseline 1426/0 |

Decision: DEC-010. Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.23 — Tensor-Parallel Layer Backward Passes

Give the v2.21 weight-managed layers a backward (gradient) path, the gap that
keeps the parallel stack inference-only (TASK-022):

| Phase | Name | Status |
|-------|------|--------|
| 1 | Analytic reference-parity RED tests: single-GPU col/row/MLP backward vs host double-precision gradients; multi-GPU col grad-input AllReduce parity / row grad-slice parity / MLP chain parity | Complete (Round 23) — 6/6 RED against unimplemented backward stubs (EV-015) |
| 2 | Implement backward: transposed GEMMs (dW = X^T dY, dX = dY W^T) on the per-instance stream-bound handle; col-layer grad-input AllReduce (NcclResult checked); silu_and_mul backward kernel; shard-sliced grad_weight writes | Complete (Round 23) — 7855ab3; col AllReduce + row local + MLP chain GREEN on 2 & 4 GPUs (EV-016) |
| 3 | Verify: backward parity GREEN on 2 & 4 GPUs, forward regression green, full-suite baseline, RIL close | Complete (Round 23) — cross-suite 19/19, single-GPU 49/49, full baseline 1422/0 |

Decision: DEC-009. Host note: GPUs 0/1 externally loaded — use
`CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.22 — Ring Sequence Parallelism On Real Multi-GPU

Implement `RingSequenceParallelism::ring_attention` for real (TASK-014 /
issue-v19-ring-parallel-noop, replacing the DEC-004 fail-fast disposition):
the multi-GPU ring path currently throws "not implemented" (send_recv_kv was
declared but never defined; pre-DEC-004 it silently returned garbage). The ring
algorithm iterates P-1 send/recv KV steps over the verified NCCL layer with
online-softmax accumulation; acceptance = each rank's local-query output equals
standard scaled dot-product attention over the FULL KV sequence (all ranks),
computed as a single-GPU/host reference.

| Phase | Name | Status |
|-------|------|--------|
| 1 | Reference-parity RED test: per-rank local Q/K/V, ring_attention == full-sequence attention reference on real multi-GPU (currently throws → RED) | Complete (Round 22) — `MultiGpu_RingAttention_MatchesFullSequenceReference` RED against the DEC-004 throw; fail-fast pin green |
| 2 | Implement ring attention: send_recv_kv (NCCL P2P send/recv on the group comm) + online-softmax accumulate over P-1 remote KV blocks | Complete (Round 22) — P-1 send/recv ring + online softmax kernels (ring_attention.cu); un-grouped NCCL P2P deadlock fixed with ncclGroupStart/End (EV-012); parity GREEN on 2 & 4 GPUs; obsolete NotImplemented pin replaced by `RejectsMissingHiddenDim` |
| 3 | Verify: reference parity GREEN on 2 & 4 GPUs, full-suite baseline, RIL close of issue/TASK-014 | Complete (Round 22) — isolated cross-suite 16/16 on 2 & 4 GPUs; single-GPU seq-parallel 33/33; full-suite baseline 1416/0 (EV-013) |

Decision: DEC-008 (milestone direction). Host note: GPUs 0/1 externally loaded —
use `CUDA_VISIBLE_DEVICES=2,3+`.

## Milestone v2.21 — Weight-Managed TensorParallelLayers

Replace the DEC-006 fail-fast disposition of the v1.3 layer stubs with real
weight-sharded layers (TASK-010 / issue-v20-tp-layers-stubs):

| Phase | Name | Status |
|-------|------|--------|
| 1 | Reference-parity tests: single-GPU full-weight refs (col/row/MLP + silu) and multi-GPU thread-per-rank (col shard concat / row AllReduce / gated MLP parity; shape rejection); RED against the stubs | Complete (Round 21) — 4/4 layer + 2/2 silu single-GPU; 5/5 multi-GPU layer parity on 2 GPUs (EV-010/EV-011) |
| 2 | Weight-managed implementation: per-rank device weight shards (set_weight slices via rank_of_device), col/row/gated-MLP forward, silu_and_mul kernel, per-instance cuBLAS handle, divisibility validation | Complete (Round 21) — 15/15 isolated NCCL cross-suite on 2 & 4 GPUs; full baseline EXIT=0 |
| 3 | RIL close: issue + TASK-010/011/012/013 resolved, docs/round record | Complete (Round 21) |

Decisions: DEC-007 (milestone direction — Megatron weight-shard semantics, new
explicit in/out-features API, >300-line diff exception). NOTE: NCCL multi-GPU
suites still hang when interleaved with `cudaDeviceReset` suites in one process
(pre-existing issue-v19-shared-nccl-context-reset — use the isolated curated
filter for cross-suite runs).

## Milestone v2.20 — TensorParallelMatmul Production Hardening

Close the cpp-review BLOCK on the v2.19 TP rewrite and disposition the last
non-functional neural parallel-training surface:

| Phase | Name | Status |
|-------|------|--------|
| 1 | TP matmul: rank via `rank_of_device` (membership validation) + propagate collective failures + RowParallel non-divisible test | Complete (Round 20) — closes review HIGH-1/HIGH-2/MEDIUM-6; TP+NCCL 20/20, acceptance 5/5 at 4 GPUs (2137cea, EV-009) |
| 2 | TensorParallelLayers stubs fail fast (were null-B cuBLAS crash / empty MLP body) | Complete (Round 20) — 3/3 reject tests; weight-managed implementation deferred (DEC-006, TASK-010) |

`issue-v19-shared-nccl-context-reset` (full-suite/curated crash after interleaved
`cudaDeviceReset`) investigated for P3: driver-level detection infeasible
(`cuCtxGetCurrent` returns the same address after reset, EV-008); disposition =
documented curated multi-GPU workflow + standard baseline without `NCCL_TESTS_AVAILABLE`.

## Milestone v2.19 — Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism)

Extend the R15-R18 thread (every never-run multi-GPU surface hides real bugs) to the
neural parallel-training layer — TensorParallelMatmul's multi-GPU execution was only
tested through buffer-size formulas, and the sequence-parallel tests passed
`comm = nullptr`:

| Phase | Name | Status |
|-------|------|--------|
| 1 | Per-rank TensorParallelMatmul multi-GPU RED tests (evidence) | Complete (Round 19) — 4/4 RED: ColumnParallel wrong output, non-divisible silently dropped, RowParallel ignored rank (2006da8, EV-004/HYP-002) |
| 2 | Correct TP matmul partition math on the verified NCCL layer | Complete (Round 19) — zero-padded strided column blocks + one AllReduce; rank-correct RowParallel; shape rejection; per-instance stream-bound cuBLAS handle (837a966, EV-005) |
| 3 | SequenceParallel real-comm + ring no-op disposition | Complete (Round 19) — gather_kv/scatter_output/all_reduce_sequence verified 4/4 on 2/4 GPUs (2a93f52); RingSequenceParallelism multi-GPU made fail-fast (650de28, DEC-004) |

`HYP-002` (TP ColumnParallel contiguous-slice math broken) validated by RED then refuted
by the converged GREEN acceptance tests. SequenceParallel collectives were verified
correct on first real-comm run — a positive refutation of the "always broken"
generalization. Follow-ups tracked: `issue-v19-shared-nccl-context-reset` (full-suite
NCCL-enabled run can SIGSEV in libcuda cuStreamDestroy after interleaved
cudaDeviceReset suites — R15-R18 never ran the NCCL-enabled full suite; pre-existing,
order/timing-sensitive), `issue-v19-ring-parallel-noop` (closed by DEC-004 fail-fast).

## Milestone v2.18 — MeshBarrier On The Verified Layer + Distributed Robustness

Close out the distributed-ops convergence thread (v2.15→v2.17 converged every
high-level op except `MeshBarrier`) and harden the collective test harness
against the two tracked follow-ups (harness flake, R16 review HIGH-B context
poisoning):

| Phase | Name | Status |
|-------|------|--------|
| 1 | Per-rank MeshBarrier multi-GPU rendezvous tests (evidence) | Complete (Round 18) — RED proof the legacy event-poll had no cross-rank arrival signal (HYP-001/EV-001) |
| 2 | Converge MeshBarrier onto verified NcclBarrier (sync/async/devices) | Complete (Round 18) — 5/5 multi-GPU barrier tests green (0a12a84) |
| 3 | Harness robustness: bounded barrier + self-healing broken-comm state | Complete (Round 18) — fail-fast vs dead comms; diagnostic RankBarrierTimeout vs dead ranks (7914b34) |

## Milestone v2.17 — Distributed Ops On Real Multi-GPU (Complete)

High-level distributed ops' multi-GPU paths proven broken (P1), converged onto the
verified NCCL layer (P2), and the SyncBatchNorm multi-GPU gradient path verified
against a global-batch host reference (P3). Full suite 1456/1423/0.

## Milestone v2.16 — Distributed Multi-GPU Verification (Complete)

NCCL multi-GPU collectives 14/14 real; distributed pool/mesh verified; distributed
matmul real path (row-split + NCCL all-gather) thread-per-rank.

## Milestone History

| Milestone | Status | Date | Requirements |
|-----------|--------|------|--------------|
| v1.0 Production Release | Shipped | 2026-04-24 | 58 |
| v1.1 Multi-GPU Support | Shipped | 2026-04-24 | 13 |
| v1.2 Toolchain Upgrade | Shipped | 2026-04-24 | 9 |
| v1.3 NCCL Integration | Shipped | 2026-04-24 | 26 |
| v1.4 Multi-Node Support | Shipped | 2026-04-24 | 15 |
| v1.5 Fault Tolerance | Shipped | 2026-04-26 | 20 |
| v1.6 Performance & Training | Shipped | 2026-04-26 | 12 |
| v1.7 Benchmarking & Testing | Shipped | 2026-04-26 | 27 |
| v1.8 Developer Experience | Shipped | 2026-04-26 | 16 |
| v1.9 Documentation | Shipped | 2026-04-26 | 12 |
| v2.0 Testing & Quality | Shipped | 2026-04-26 | 12 |
| v2.1 New Algorithms | Shipped | 2026-04-26 | 12 |
| v2.2 Comprehensive Enhancement | Shipped | 2026-04-27 | 18 |
| v2.3 Extended Algorithms | Shipped | 2026-04-28 | 13 |
| v2.4 Production Hardening | Shipped | 2026-04-28 | 15 |
| v2.5 Error Handling & Recovery | Shipped | 2026-04-28 | 12 |
| v2.6 Transformer & Inference Optimization | Shipped | 2026-04-29 | 18 |
| v2.7 Comprehensive Testing & Validation | Shipped | 2026-04-30 | 16 |
| v2.8 Numerical Computing & Performance | Shipped | 2026-05-01 | 20 |
| v2.9 Architecture Refactor | Shipped | 2026-05-01 | 7 |
| v2.10 Sparse Solver Acceleration | Shipped | 2026-05-01 | 11 |
| v2.11 Performance Tooling | Shipped | 2026-05-02 | 14 |
| v2.12 Advanced Quantization | Shipped | 2026-05-03 | 14 |
| v2.13 Transformer Optimization | Shipped | 2026-05-06 | 25 |
| v2.14 Documentation Quality | Shipped | 2026-05-07 | 31 |
| v2.15 Test Quality Assurance | Complete | 2026-05-09 | 5 phases |
| v2.16 Distributed Multi-GPU Verification | Complete | 2026-08-11 | 3 phases |
| v2.17 Distributed Ops On Real Multi-GPU | Complete | 2026-08-12 | 3 phases |
| v2.18 MeshBarrier On The Verified Layer + Distributed Robustness | Complete | 2026-08-12 | 3 phases |
| v2.19 Parallel Training On Real Multi-GPU (Tensor + Sequence Parallelism) | Complete | 2026-08-12 | 3 phases |
| v2.20 TensorParallelMatmul Production Hardening | Complete | 2026-08-12 | 2 phases |
| v2.21 Weight-Managed TensorParallelLayers | Complete | 2026-08-12 | 3 phases |
| v2.22 Ring Sequence Parallelism On Real Multi-GPU | Complete | 2026-08-12 | 3 phases |
| v2.23 Tensor-Parallel Layer Backward Passes | Complete | 2026-08-12 | 3 phases |
| v2.24 End-to-End Training Step On The Tensor-Parallel Stack | Complete | 2026-08-13 | 3 phases |
| v2.25 MicroTrainer: End-to-End Gradient Training Convergence | Complete | 2026-08-13 | 3 phases |
| v2.26 Tensor-Parallel Multi-Head Attention + Mini-Transformer Training | Complete | 2026-08-13 | 3 phases |
| v2.27 Device-Native Optimizer Kernels (AdamW + Gradient Norm/Clip) | Complete | 2026-08-13 | 3 phases |
| v2.28 Normalized Transformer Blocks + Deep Multi-Layer Training | Complete | 2026-08-19 | 3 phases |
| v2.29 Device-Native SDPA + LayerNorm Training Kernels | Complete | 2026-08-19 | 3 phases |
| v2.30 Device-Native Cross-Entropy Loss Reduction | Complete | 2026-08-23 | 3 phases |
| v2.31 Device-Native LAMB Optimizer Kernel | Complete | 2026-08-24 | 3 phases |

---

## State updated: 2026-08-19 — Milestone v2.29 complete (RIL Rounds 30/31)

TASK-043/044/045/046 resolved, DEC-015, issue-v29-sdpa-host-roundtrip +
issue-v29-ln-one-thread-per-row resolved (CHG-020/021). The Round-28 LEARN
performance round is done: multi-head SDPA forward/backward moved from the host
D2H/H2D round-trip to device kernels (`attention_kernels.cu` — per-(position,
head) blocks, shared max-subtracted softmax, block reductions, dV/dK global
atomics across query positions), wired behind the same `sdpa_forward`/
`sdpa_backward` API; the trainable LayerNorm kernels parallelized from one
thread per row to block/warp reductions. Verified: 6 P1 contracts GREEN
(max_abs ~3e-8 SDPA / ~5e-7 LN vs fp64 references), neural single-GPU 79/79,
neural multi-GPU cross-suite + deep-block K-step trajectory parity 19/19 on 2 &
4 GPUs (unchanged by the new kernels). cpp-reviewer caught a real HIGH
(cross-call shared race on the block-reduction buffer, compute-sanitizer
racecheck-verified, fixed with entry barriers) + MEDIUM (signed int overflow in
grids, 64-bit + gridDim.x guard) — both fixed with re-verification (CHG-021,
EV-027). Only TASK-018 (maintainability, 1.50 < 3.0) remains active — stop
conditions hold. RN: use `CUDA_VISIBLE_DEVICES=2,3+`.

## State updated: 2026-08-19 — Milestone v2.28 complete (RIL Round 28)

TASK-039/040/041/042 resolved, DEC-014, issue-v28-layer-norm-untrainable
resolved (CHG-017). The stack can now train a DEEP (N-block, pre-LN, residual)
transformer: device LayerNorm fwd/bwd with trainable gamma/beta
(collective-free — replicated activations, one AdamW per affine tensor), a
pre-LN residual TransformerBlock (LN→MHA→+x→LN→MLP→+x, 11 weight tensors), and
a TransformerTrainer (N blocks + final LN head, 11N+2 per-tensor optimizers).
Verified: LayerNormTrainableTest 6/6 + TransformerBlockTest 6/6 (incl. seq>1
regressions), neural multi-GPU cross-suite 19/19 on 2 & 4 GPUs (deep-transformer
shard==single-GPU parity). cpp-reviewer caught C1 (block sized LayerNorm/
residuals on batch rows instead of batch*seq — latent OOB at seq>1) and L2
(layer_norm_inference null mean/var device-fault) — both fixed with regression
tests (799812e, 81edd9d). Full-suite baseline not reproducible: all 8 GPUs
~76GB used, ~20 memory-heavy non-neural tests OOM (environment, not v2.28).
RN: use `CUDA_VISIBLE_DEVICES=2,3+`; re-check nvidia-smi before multi-GPU runs.

## State updated: 2026-08-12 — Milestone v2.23 complete (RIL Round 23)

TASK-022 / TASK-019/020/021 closed. The v2.21 weight-managed layers now have
backward passes, closing the gap that kept the parallel stack inference-only. Backward
math verified against analytic host references on real multi-GPU: `dW = X^T dY`,
`dX = dY W^T` (realized as the library's verified non-transposed matmul convention
through new transpose/elementwise_add/silu_and_mul_backward kernels). The column layer's
grad-input is the **AllReduce** of the per-rank `dY W^T` (replicated input ⇒ summed
gradient — a real collective never exercised before, now with `NcclResult` checked);
the row layer's grad-input is local (matches the sharded-input layout); the gated MLP
chains down→silu_and_mul→gate/up with the grad-inputs summed then AllReduced.

Acceptance: single-GPU backward 3/3 + activation helpers 3/3; multi-GPU backward 3/3 on
2 & 4 GPUs (EV-016); isolated distributed cross-suite 19/19 on 2 & 4 GPUs; single-GPU
regression 49/49; full-suite baseline 1422/0. A test-side strided-block extraction bug
in the row grad-input reference was fixed (the implementation was correct). Decision:
DEC-009. Next: the stack is now forward+backward capable — a gradient/training
end-to-end integration (loss → backward → optimizer stepping the parallel layers) is the
natural next milestone.

## State updated: 2026-08-12 — Milestone v2.22 complete (RIL Round 22)

TASK-014 / issue-v19-ring-parallel-noop closed. `RingSequenceParallelism::ring_attention`
multi-GPU — the last dispositioned-but-unimplemented parallel surface (DEC-004 fail-fast
throw; pre-DEC-004 a silent garbage no-op) — is now a real ring algorithm: each of the
P-1 steps sends the local KV block clockwise to next_rank while receiving prev_rank's
block (NCCL P2P `ncclSend`/`ncclRecv` on the group comm, wrapped in
`ncclGroupStart/End`), accumulating each block with online-softmax kernels in
`src/cuda/distributed/ring_attention.cu`. `SequenceParallelConfig.hidden_dim` is now
required by the multi-GPU ring path (interpret Q/K/V buffers; unset → fail fast).

Key finding (EV-012): **ungrouped consecutive NCCL P2P ops deadlock the ring** —
without `ncclGroupStart/End` the parity test hung (~100% CPU); the group wrap fixed it.
Acceptance: per-rank local-query output == standard scaled dot-product attention over
the FULL KV sequence (all ranks concatenated, host double-precision reference) — GREEN
on 2 & 4 GPUs; isolated distributed cross-suite 16/16 on 2 & 4 GPUs; single-GPU
seq-parallel 33/33; full-suite baseline 1416/0 (EV-012/EV-013). Votes: DEC-008.
Note: the pre-existing issue-v19-shared-nccl-context-reset still hangs NCCL runs when
`cudaDeviceReset` suites interleave — use the isolated curated filter.

## State updated: 2026-08-12 — Milestone v2.21 complete (RIL Round 21)

TASK-010 / issue-v20-tp-layers-stubs resolved. The v1.3 TensorParallelLayers
scaffolding (ColumnParallelLayer / RowParallelLayer / TensorParallelMLP) owned
no weight storage and was disposed as fail-fast stubs in v2.20 (DEC-006). This
milestone implements them for real with **weight-sharded Megatron semantics**:
each layer owns its rank's shard of the weight on device; set_weight() uploads
the full weight and slices the rank shard via NcclContext::rank_of_device (the
v2.20 verified convention); ColumnParallelLayer::forward is a local shard
matmul (sharded output, no comm), RowParallelLayer::forward is local matmul +
one block-wise AllReduce (replicated output), TensorParallelMLP is a SiLU-gated
FFN (gate/up column-parallel, down row-parallel). Per-instance stream-bound
cuBLAS handle for thread-per-rank safety; divisibility validated at construction
(backing the v2.19 finding that the pre-v2.20 shard math was broken). Added the
`silu_and_mul` gated activation.

Acceptance (real multi-GPU, CUDA_VISIBLE_DEVICES=2,3): column shard concat ==
single-GPU full-weight reference; row AllReduce replicated == reference; gated
MLP == single-GPU full-weight reference — 15/15 isolated NCCL cross-suite on 2
and 4 GPUs (EV-010/EV-011), TP+layer+sequence multi-GPU green. Single-GPU tp<=1
path: 6/6 (4 layer + 2 silu) reference parity. Full-suite baseline EXIT=0.
NOTE: running the NCCL multi-GPU suites interleaved with cudaDeviceReset suites
(ActivationTest) in one process still hangs/crashes — the pre-existing
issue-v19-shared-nccl-context-reset (order/timing-sensitive); the NCCL cross
-suite must be run via the isolated curated filter, per the documented
disposition. API: layer ctors changed to explicit (ctx, in_features,
out_features); zero external callers (verification grep clean).
