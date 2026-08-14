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

## Execute — P2 (TASK-037, `CHG-016` 711d511)

- `optimizers_kernels.cu` (new, compiled as CUDA): fused `adamw_step_kernel`
  (bias-corrected lr_t, m_hat/v_hat, decoupled-wd update in place on device),
  `grad_scale_kernel` (clip), tree-reduced L2/Inf norm partial kernels
  (512 blocks). `detail::` entry points feed the host API.
- `optimizers.cpp`: `AdamWOptimizer::step` sizes device m/v `Buffer` and calls
  the fused kernel (no D2H/H2D); `compute_gradient_norm` uses the device
  reduction; `clip_gradients` scales on device. API unchanged — every caller
  (layer step(), MicroTrainer, tests) accelerates with zero edits. LAMB stays
  host-side (layer-adaptation needs host layer-norm reads / per-element clamp).
- `optimizers_test.cpp`: 3 device-vs-host parity tests (K=5 AdamW, L2/Inf
  norm, clip) — exact-element match.

## Verify (EV-021)

- OptimizersTest 16/16 (incl. 3 new parity); single-GPU neural regression
  46/46; isolated NCCL cross-suite 43/43 on 2 & 4 GPUs — including the
  MicroTrainer host-fp64 trajectory parity and the K=12 mini-transformer
  parity now running on the DEVICE optimizer (the definitive
  training-correctness attestation). Full-suite baseline 1435/0 EXIT=0.
- cpp-reviewer pass on the kernel code pending (CHG-016).

## Learn

- The device path needed a new `.cu` (the existing `optimizers.cpp` is host
  C++); kernels + `detail::` entry points in the .cu, host glue in the .cpp —
  the module boundary is now host-API / device-kernels.
- The chunked partial reductions are n-agnostic (span ceil-div over 512, empty
  blocks write 0); exact-element AdamW parity proves the fused kernel preserves
  the former float semantics — so all host-fp64 trajectory tests re-attest
  unchanged.

## Milestone close

- `TASK-035/036/037/038` resolved; DEC-013; EV-021; CHG-016 (711d511). Round
  bumped to 28.
- The training stack now runs its optimizer entirely on device (no blocking
  D2H/H2D per step) with exact host-formula parity, sustained through the
  MicroTrainer host-fp64 trajectory and mini-transformer parity tests. The
  device-optimizer conversion is the last deferred perf item from DEC-010/011/
  012; future work is capability (a deeper/multi-layer transformer, more
  optimizer variants on device, LAMB kernelization).
