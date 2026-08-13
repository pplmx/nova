# RIL — Round 24 (2026-08-13) — milestone v2.24 "End-to-End Training Step On The Tensor-Parallel Stack (loss backward → layer backward → optimizer shard step)"

## Round focus

**Open milestone v2.24** on the gap DEC-009 named as the natural next step:
the v2.23 layers are forward+backward capable, but the stack cannot be
*trained* — `cross_entropy_loss` (host-side, its device kernels are dead code)
has no grad, `weight_` shards are private with no `step()`, and nothing chains
forward → loss → backward → optimizer. This milestone adds the missing seed
gradient + the per-shard optimizer step and verifies a full training loop
matches a host fp64 full-weight reference on real multi-GPU (TASK-023).

## Observation / Model

- Concrete gaps (DEC-009 "loss → backward → optimizer stepping the parallel
  layers", now grounded in code):
  1. **No loss backward.** `loss_functions.cu` has three `__global__` kernels
     (cross_entropy_logits, focal_gradient, contrastive_cosine) that are
     *never launched* — the public `cross_entropy_loss()` is a host-side loop
     that reads host vectors and returns a scalar. Nothing computes
     `dlogits = (softmax - onehot)/B` on device, so a layer backward chain has
     no seed.
  2. **No optimizer step surface on the layers.** `backward()` writes
     full-size grad_weight buffers (rank slices scattered via
     cudaMemcpy2DAsync / row-block copies), but `weight_` is private; no
     method (or accessor) applies AdamW to the rank shard.
  3. **No training driver.** Nothing runs forward → loss-backward →
     layer-backward → optimizer on the parallel path.
- The gradient layout is *exact*: the column-layer grad-shard is the column
  block of `X^T dY_full` (replicated input ⇒ per-rank slice = full block), the
  row-layer grad-shard is the row block of `X_r^T dY` (sharded input ⇒ local
  block = full grad's block). AdamW is a per-element map, so stepping each rank
  shard with the same local gradient tiles to the full-weight update — a clean,
  provable parity (this is the core of the milestone's acceptance over K steps).

## Decisions

- **DEC-010**: milestone direction — device cross-entropy-with-logits backward
  (`cross_entropy_logits_backward`, dlogits = (softmax - onehot)/B) seeding the
  chain; per-layer `step(AdamWOptimizer&, grad_weight_full, step_no)` applying
  AdamW in place to the private rank shard; TensorParallelMLP::step chains
  gate/up/down; acceptance = K-step multi-GPU shard training == host fp64
  full-weight reference. Rejected: end-to-end TensorParallelAttention
  composition (capstone, heavier geometry), device-native fused optimizer
  kernels (perf task, orthogonal), a full autograd/DAG framework (too broad).

## Execute — P1 (TASK-024, RED evidence)

- Header contracts added: `cross_entropy_logits_backward` (device, in
  loss_functions.h); `step(optimizer, grad_weight_full, step_no)` +
  `copy_weight_shard`(host out) on ColumnParallelLayer/RowParallelLayer;
  `TensorParallelMLP::step(...)` + `copy_weights(...)`; the layers header now
  includes optimizers.h (step takes AdamWOptimizer by reference).
- Provisional throw-stubs: each new entry point throws
  `std::runtime_error("... not implemented (TASK-025)")`.
- RED tests (5): device CE-logits backward vs host fp64 analytic; single-GPU
  col/row/MLP step vs host AdamW over one analytic gradient (MLP verified via
  a fresh-input forward against the host-stepped reference); multi-GPU
  `MultiGpu_TrainingStep_MatchesHostFullWeight` — K=3 identical AdamW steps on
  the sharded MLP per rank, shards assembled, compared against a host fp64
  full-weight MLP reference (`host_training_step` analytic chain + HostAdamW).

## Verify — P1 (EV-017)

- BOTH new tests fail RED, all against the provisional stubs: loss backward
  throws (grad output stays 0) on device 0; the three single-GPU layer steps
  throw "step not implemented (TASK-025)"; the multi-GPU training-step parity
  throws `cross_entropy_logits_backward not implemented` on real 2-GPU
  (`CUDA_VISIBLE_DEVICES=2,3`). Existing suites unaffected (build green).
- 5/5 RED pins the contracts as written; P2 replaces the stubs with the device
  kernel + shard-step implementation.

## Execute — Phase 2 (TASK-025) + P3 (TASK-026, EV-018)

- Device `cross_entropy_logits_backward_kernel`: one thread per (b, c),
  max-subtract per-row softmax recomputed redundantly per class (cheap for
  C <= hundreds), `grad_logits[b*C+c] = (softmax_c - onehot(c==targets[b])) /
  B` under reduction_mean (scale 1/B), `CUDA_CHECK(cudaGetLastError())` after
  launch. Replaces the P1 stub.
- `ColumnParallelLayer::step`: extracts the rank's column-slice of the caller's
  full grad_weight via `cudaMemcpy2DAsync` into a shard-sized contiguous buffer
  (tp==1: a plain device-to-device copy), then `AdamWOptimizer::step` on the
  private `weight_` shard in place. Same shape for `RowParallelLayer::step`
  (contiguous row-block copy). `TensorParallelMLP::step` chains
  gate_proj_/up_proj_/down_proj_·step. Read-back helpers `copy_weight_shard` /
  `copy_weights` mirror `set_weight` for the parity assertions.
- Per-rank `AdamWOptimizer` instance is the concurrency unit: `m/v` are
  per-instance host vectors, so each rank steps its own shard's moments with no
  cross-rank state — exactly the per-element independence the shard-step parity
  relies on.

Verify (EV-018): 4 single-GPU training tests GREEN on device 0 (loss backward
vs host fp64 analytic; col/row/MLP step vs host AdamW over one analytic step —
the MLP check compares the stepped shards **directly** at 2e-4, not a
tolerance-masked forward); multi-GPU `MultiGpu_TrainingStep_MatchesHostFullWeight`
GREEN on **2 & 4 GPUs** (K=3 AdamW steps per rank, shards assembled == host fp64
full-weight reference); isolated NCCL cross-suite **39/39 on 2 & 4 GPUs** (TP
multi-GPU 14/14 incl the new training-step test, NCCL collectives 15/15,
seq-parallel 6/6, distributed matmul multi-GPU 4/4); single-GPU neural
regression 37/37 (layers/loss/activations/optimizers/matmul); full-suite
baseline **1426/0 EXIT=0** (captured before the host's external GPU load
appeared; a later full-suite re-run hit 54 device-0 OOMs from externally-owned
GPUs 0/1/2.. — environmental, not a code regression; the multi-GPU suites still
pass on the loaded devices 2-5).

cpp-review disposition (CHG-013): the reviewer's **HIGH** was real and is fixed —
`TensorParallelMLP::step` originally chained ONE AdamWOptimizer through
gate→up→down; its `m/v` per-element moment buffers are keyed by element index,
so up/down reused the gate moment history and the training-step tests passed
only at 2e-2 tolerance (direct weights diverge ~1.4e-2). Fix: `step()` now takes
**one optimizer per weight tensor** (gate/up/down); tests use three instances
and assert the stepped shards against the host advis images at 2e-4, so a
shared optimizer can no longer pass. The reviewer's CRITICAL (heap OOB "when
shard sizes differ") does **not** apply: gate/up shard count `hidden*inter/tp`
and down shard count `(inter/tp)*hidden` are the *same integer product*, so
AdamW's first-call sizing always covers all three; no path indexes past
`m/v`. LOW notes (NULL-stream ordering inside the test's forward/backward
vs the CE kernel; AdamW ignoring its stream param) are safe because every layer
call syncs before returning and each rank thread pins its device with
`cudaSetDevice` — dispositioned as no-change.

## Learn

- The loss file's `__global__` kernels were dead code — a useful signal that
  the "forward-only loss" was never wired onto the device stack. The clean
  seed-gradient contract (dlogits = (softmax - onehot)/B under reduction_mean)
  is what future training loops rely on.
- AdamW moment buffers (`m_data_/v_data_`) are per-optimizer-instance and
  host-side, so a *per-rank* AdamW instance is the right concurrency unit for
  the shard-step (no cross-rank state) — the tiles-to-full-weight argument.
- Stream discipline: the library AdamW step is synchronous (D2H/H2D
  `cudaMemcpy` on the calling thread's current device), so each rank thread
  pinned with `cudaSetDevice` steps exactly its own shard — the thread-per-rank
  harness and the per-element AdamW math line up cleanly.

## Milestone close

- `TASK-023/024/025/026` resolved; DEC-010; EV-017/EV-018; CHG-013. Round
  bumped to 25.
- The parallel stack is now *trainable*: forward → device CE-logits backward →
  layer backward → per-rank AdamW shard step reproduces a host fp64 full-weight
  reference over K steps on real multi-GPU. Next (natural candidates): attention
  block forward+backward integration (the DEC-010-rejected capstone), device-
  native fused optimizer kernels (perf task), or a micro-trainer/end-to-end
  mini-model milestone.
