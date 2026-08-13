# RIL — Round 25 (2026-08-13) — milestone v2.25 "MicroTrainer: End-to-End Gradient Training Convergence On The Tensor-Parallel Stack"

## Round focus

**Open and close milestone v2.25** on the gap v2.24 left open: the parallel
stack proved its *step* is exact, but never that it *learns*. This milestone
adds a reusable `training::MicroTrainer` driver and proves loss descent,
accuracy rise, and a full-run multi-GPU sharded trajectory == host fp64
full-weight reference (TASK-027).

## Observation / Model

- v2.24 evidence: one training step is weight-exact (K=3 parity), but no
  loss-descent / accuracy / long-run trajectory check existed anywhere on the
  parallel path — "the stack is trainable" was an extrapolation, not a fact.
- Building blocks all verified: per-layer step (v2.24), device
  cross_entropy_logits_backward (v2.24), weight-sharded MLP forward/backward
  (v2.21/2.23), host fp64 analytic MLP forward/backward (v2.23 tests).
- The MLP as a classifier (output [m x hidden] = logits over hidden classes)
  is self-contained: no separate head needed.

## Decisions

- **DEC-011**: milestone direction — `training::MicroTrainer` (owns the MLP +
  one AdamW per weight tensor + device scratch) with acceptance = convergence:
  single-GPU loss descends / accuracy rises; multi-GPU K-step sharded run
  reproduces host fp64 full-weight trajectory (weights AND loss curve) on 2 &
  4 GPUs. Rejected: attention-block composition (capstone, heavier), device
  fused optimizer (perf, orthogonal), autograd framework (too broad).

## Execute — P2 (TASK-029, `CHG-014` cc35094) + bug fix

- `training.h/.cpp`: `MicroTrainer` — set_weight, train_step (forward → device
  CE-logits backward → MLP backward → per-shard AdamW, returns host mean CE),
  evaluate (+ top-1 accuracy), copy_weights, accessors. Scratch grows with the
  batch; per-tensor AdamW instances keep moments private per weight.
- **Bug found & fixed** (`issue-v24-ce-loss-running-sum`): `cross_entropy_loss`
  filled `log_probs` while accumulating `sum_exp`, so the softmax denominator
  for a target at class c used the *partial* running sum over c' <= c —
  systematically under-reporting loss for early targets. Masked in v2.24 by
  relative-only loss assertions (descend / finite) and by the correct-gradient
  device CE-backward (training still worked, weight parity held). Exposed by
  the v2.25 device-vs-host-fp64 trajectory parity: identical logits gave
  dev_loss 2.18 vs host fp64 3.91. Fix: compute the full-class sum first, then
  fill all log_probs with it. After fix: dev == host exactly (3.914220 both).
- `tests/neural/micro_trainer_test.cpp` (single-GPU, tp<=1) and
  `tests/neural/micro_trainer_multigpu_test.cpp` (real thread-per-rank).
  Multi-GPU reference: host fp64 `host_training_step` (analytic forward, CE
  backward, MLP backward, per-weight AdamW) over the same K steps; acceptance =
  assembled shards within 2e-2 AND the loss curve within 0.1 per step.

## Verify (EV-019)

- Single-GPU: loss descends over 60 steps, accuracy rises above chance (1/8);
  train_step returns finite, changing losses (weights move).
- Multi-GPU `MultiGpu_MicroTrainer_MatchesHostFullWeight` (K=10, lr=0.01):
  **GREEN on 2 & 4 GPUs** — assembled gate/up/down shards == host fp64 final
  weights (2e-2) AND the per-step loss curve tracks the reference (0.1).
- Cross-suite (TP multi-GPU + NCCL + seq-parallel + distributed matmul +
  MicroTrainer) **40/40 on 2 GPUs**; single-GPU neural regression **45/45**
  (TensorParallelLayers + Loss + Activations + Optimizers + Matmul +
  MicroTrainer). Full-suite baseline 1426/0 (captured before the host's
  external GPU load; a later loaded-GPU run saw only environmental OOMs).
- Test-side visit: the multi-GPU loss capture must come from one rank (the
  output is replicated) — a shared-buffer capture with a size guard scrambles
  the curve when ranks run at different paces.

## Learn

- Relative-only loss assertions hid a real numeric bug in the *loss function*
  itself; a cross-implementation (device vs host fp64) *absolute* check is what
  exposed the running-sum denominator. Absolute-value assertions against an
  independent reference are worth more than trend checks.
- The MicroTrainer is the first reusable training loop on the stack; it is the
  unit future caps (mini-model, attention-block training) will drive.

## Milestone close

- `TASK-027/028/029/030` resolved; DEC-011; EV-019; CHG-014; issue
  `issue-v24-ce-loss-running-sum` resolved. Round bumped to 26.
- The parallel stack now demonstrably **learns**: a sharded MicroTrainer run
  reproduces the single-GPU/fp64 training trajectory end-to-end. Next
  (natural candidates): attention-block forward+backward integration (the
  DEC-011-rejected capstone, now trainable with MicroTrainer), device-native
  fused optimizer kernels (perf), or a mini-language-model milestone.
