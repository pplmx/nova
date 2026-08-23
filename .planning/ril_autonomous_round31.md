# RIL — Round 31 (2026-08-23) — milestone v2.31 "Device-Native LAMB Optimizer Kernel"

## Round focus

**Open milestone v2.31** — the family-completion round. Round-30's LEARN named
CUDA-graph capture of the device train_step as the next candidate, but an
empirical probe (EV-034) **refuted that direction**: the training stack launches
kernels on the legacy default (null) stream and creates per-call op streams
inside the layers, and `cudaStreamBeginCapture` cannot capture either pattern
("operation would make the legacy stream depend on a capturing blocking
stream"; "operation not permitted when stream is capturing").
Capture-stream-parametric capture would require a wide API rewrite plus
NCCL-in-capture handling — exactly the refactor DEC-016 rejected.

The strongest remaining family member is **LAMB**: v2.27 kernelized AdamW
behind the same API and the gradient-norm/clip reductions, but
`LAMBOptimizer::step` still round-trips the full params+grads through the
host every call — `cudaMemcpy` grads D2H + params D2H, an O(n) host loop with
per-element `std::vector` m/v, then H2D back (optimizers.cpp:82-160, "LAMB
keeps its host-side implementation" per the v2.27 header; DEC-014/015 deferred
it). Its layer adaptation reads host scalars (`layer_norm_1`/`layer_norm_2`),
which are just kernel args (rtw = phi_1/phi_2); the trust ratio
r = param/update is per-element, clamped to [1/clamp_val, clamp_val]. So LAMB
is kernelizable exactly like adamw_step_device (TASK-051, DEC-017).

## Observation / Model

- **M1 (binding)**: LAMBOptimizer::step is a full-buffer D2H→host-loop→H2D
  every call. Same D2H/H2D family v2.27 removed from AdamW / norm / clip. It is
  the last optimizer in the stack that still round-trips; DEC-014/015 deferred
  it as "orthogonal" while AdamW became device-native.
- **Convention (mirrors v2.27)**: keep the public `LAMBOptimizer::step` API and
  `LAMBConfig`. Add `detail::lamb_step_device` (a fused `lamb_step_kernel`:
  bias-corrected m/v, per-element trust-ratio update with rtw as scalar arg),
  m/v become device Buffers (like AdamW's m_data_/v_data_). The host loop
  stays in the tests as the reference oracle. Verify by exact-element
  device-vs-host parity over K steps (AdamWDeviceMatchesHostFormula style) +
  OptimizersTest regression.
- **The host loop is the oracle**: the device kernel must reproduce it
  bit-for-bit (same formulas, same operation order) — the v2.27 convention.

## Decisions

- **DEC-017**: milestone direction — device-native LAMB kernel behind the same
  public `LAMBOptimizer::step` API: fused per-element bias-corrected moments +
  trust-ratio update (rtw host-scalar arg, per-element r clamp), m/v device
  buffers, dropping the D2H/H2D. Verified by exact-element device-vs-host
  parity and the OptimizersTest regression. Rejected: CUDA-graph capture of
  train_step (EV-034 empirically refuted); stream-parametric rewrite (wide,
  NCCL-risky); TASK-018 (below 3.0, orthogonal maintainability);
  issue-v19 shared-context-reset (driver-undetectable, EV-008); fusing LAMB
  into the layers (couples modules — LAMB stays standalone like AdamW).

## Milestone definition (opened)

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: `detail::lamb_step_device` vs host LAMB reference (exact-element over K steps; layer-adaptation on/off; clamp on/off); RED against throw-stub | In progress (Round 31) |
| 2 | Implement: `optimizers_kernels.cu` gains `lamb_step_kernel` (fused m/v + trust-ratio update); `LAMBOptimizer` m/v → device Buffers; `step()` routes through `detail::lamb_step_device`, D2H/H2D dropped | Pending |
| 3 | Verify: LAMB device-vs-host exact parity GREEN over K steps; OptimizersTest regression GREEN; cpp-reviewer pass; RIL close | Pending |

**Graph additions (Round 31, opening):**
- `DEC-017` (decision, milestone direction; also records the EV-034 CUDA-graph
  refutation); `TASK-051` (umbrella) / `TASK-052` (P1) / `TASK-053` (P2) /
  `TASK-054` (P3); `issue-v31-lamb-host-roundtrip` (the addressed debt);
  `EV-034` (graph-capture feasibility refutation).

## P1 RED (TASK-052)

- Added the contract surface with a provisional throw-stub:
  `detail::lamb_step_device` (device params/grads/m/v -> fused update, rtw +
  clamp as scalars) + 2 new `OptimizersTest.LAMBDevice*` tests in
  `optimizers_test.cpp` vs a `host_lamb` reference (the exact
  `LAMBOptimizer::step` math):
  - `WithLayerAdaptation` (n=256, K=5, rtw=1.2, clamp=10); and
  - `NoLayerAdaptation` (n=128, K=3, rtw=1.0, clamp unused).
- RED run (EV-035): 2/2 fail for the right reason — both throw
  "lamb_step_device not implemented (v2.31 P1 stub)". 16/16 OptimizersTest
  baseline GREEN before this change (the 6 LAMB-related existing tests pass).

## P2 implementation (TASK-053, EV-036)

- `optimizers_kernels.cu`: `lamb_step_kernel` (per-element bias-corrected
  m/v, update = m_hat/(sqrt(v_hat)+eps) + wd*param, per-element trust ratio
  r = clamp(|w|/|update|, 1/clamp_val, clamp_val) when use_layer_adaptation,
  w' = w - lr*r*rtw*update; rtw is the host-computed phi_1/phi_2 layer-ratio
  scalar, a kernel arg) behind `detail::lamb_step_device` (256 threads,
  `cudaGetLastError` after launch — the module convention).
- `LAMBOptimizer` (optimizers.h/.cpp): m/v become device `cuda::memory::Buffer`
  (lazily allocated in step(), regrown if the batch grew; zero_momentum fills
  them), `step()` computes rtw on host (the two scalar layer-norm reads stay
  host-side) and routes the whole per-element update through
  `detail::lamb_step_device`. The full `cudaMemcpy` grads D2H + params D2H +
  O(n) host loop + H2D back are deleted (issue-v31-lamb-host-roundtrip
  resolved). Public API unchanged.
- GREEN (EV-036): both P1 contracts pass exact-element (device == host_lamb
  over 5/3 steps); `LAMBOptimizerLayerAdaptation` / zero-momentum / setters /
  construction still pass; Optimizers + trainer regression 26/26.

## P3 verify (TASK-054)

- LAMB device-vs-host exact-element parity GREEN (`WithLayerAdaptation` 1/1 +
  `NoLayerAdaptation` 1/1, EV-036).
- OptimizersTest regression GREEN (16/16 incl. the 2 new LAMB contracts +
  AdamW device parity + LAMB construction/setters/zero-momentum/layer-adaptation).
- Neural single-GPU regression GREEN: 101/101 (Optimizers 18 incl. the 2 new
  contracts, LayerNormTrainable, AttentionKernels, TransformerBlock,
  TensorParallelAttention/Layers, MicroTrainer, loss incl. device contracts,
  activation/softmax/matmul/LN).
- Multi-GPU deep-block K-step trajectory parity UNCHANGED (2/2 on 2 GPUs —
  MicroTrainer + DeepBlockMultiGpuPinnedTest; LAMB doesn't touch the NCCL path).
- cpp-reviewer: [PENDING — see commit message]. RIL close on review pass.
- Self-review hygiene (post-GREEN): dropped the now-unused `<stdexcept>`
  include from optimizers_kernels.cu after the throw-stub was replaced; 18/18
  OptimizersTest + 36/36 optimizer/layer regression re-verified after the edit.
  The kernel is parity-exact with the host oracle (identical formulas and
  operation order — the v2.27 bit-for-bit convention), reads `params[i]`/
  `m[i]`/`v[i]` before writes, uses size_t index math, and checks the launch
  error; the m/v device buffers are allocated once and only regrown (not
  per-call).
