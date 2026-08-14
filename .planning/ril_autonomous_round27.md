# RIL — Round 27 (2026-08-13) — milestone v2.27 "Device-Native Optimizer Kernels (AdamW + Gradient Norm/Clip)"

## Round focus

**Open milestone v2.27** on the binding training-performance constraint the
v2.25 cpp-review flagged (M2): `AdamWOptimizer::step` copies full params+grads
D2H, runs the update in a host loop, copies back — 9 blocking D2H/H2D memcpys
per MicroTrainer step (3 weight tensors × 3 copies); `compute_gradient_norm`
and `clip_gradients` do the same full-buffer round-trip. DEC-010/011/012 each
deferred "device-native fused optimizer kernels" as a perf task orthogonal to
their correctness milestones; now that training converges (v2.25) and forms a
transformer (v2.26), removing the round-trip is the next training improvement
(TASK-035).

## Observation / Model

- The optimizer module (`optimizers.cpp`) is entirely host-side: AdamW
  (params/grads D2H → vector loop → H2D, m/v as host `std::vector`), LAMB,
  `compute_gradient_norm` (L2 sum / Inf max over a D2H copy), `clip_gradients`
  (D2H → scale loop → H2D). Every one pays a blocking full-buffer round-trip.
- The training stack (MicroTrainer v2.25, TP attention v2.26) uses
  `AdamWOptimizer::step` per weight tensor every step. The API surface is
  stable: `step(float* params, const float* grads, size_t n, int step_no,
  cudaStream_t)`. Keeping the signature unchanged means every caller
  (layer `step()`, MicroTrainer, the training tests) accelerates with no edits.
- AdamW is per-element: bias-corrected `lr_t`, `m = b1*m + (1-b1)g`,
  `v = b2*v + (1-b2)g²`, `update = m_hat/(sqrt(v_hat)+eps) + wd*w`,
  `w' = w - lr_t*update`. A single fused kernel with device m/v replaces the
  host loop; parity is exact-element (same float ops), so existing host-fp64
  trajectory parity tests re-attest with no tolerance change.

## Decisions

- **DEC-013**: milestone direction — device kernels behind the same API.
  Rejected: rewriting the optimizer API (callers would fork; the device path
  replaces the host loop in place); fusing AdamW into the layer `step()`
  (couples optimizers to layers); TensorCore/AMX-optimized kernels (premature;
  a plain fused kernel already removes the round-trip).

## Execute — P1 (TASK-036, parity)

- RED/parity tests: device AdamW over K steps == host-formula reference
  (exact-element); device `compute_gradient_norm` L2/Inf == host; device
  `clip_gradients` == host scale. The existing multi-GPU MicroTrainer
  trajectory-parity test (v2.25) already asserts training == host fp64
  reference with the device optimizer — it is the end-to-end attestation.

## Milestone status (open)

- P1 parity pinned (EV-021, once implemented); P2 (TASK-037) = kernels; P3
  (TASK-038) = verify GREEN on 2 & 4 GPUs + cross-suite + regression + close.
