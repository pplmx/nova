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

## Learn

- The loss file's `__global__` kernels were dead code — a useful signal that
  the "forward-only loss" was never wired onto the device stack. The clean
  seed-gradient contract (dlogits = (softmax - onehot)/B under reduction_mean)
  is what future training loops rely on.
- AdamW moment buffers (`m_data_/v_data_`) are per-optimizer-instance and
  host-side, so a *per-rank* AdamW instance is the right concurrency unit for
  the shard-step (no cross-rank state) — the tiles-to-full-weight argument.

## Milestone status (open)

- P1 RED pinned (EV-017); P2 (TASK-025) = implement; P3 (TASK-026) = verify
  multi-GPU GREEN on 2 & 4 GPUs + regression + full-suite baseline + close.
