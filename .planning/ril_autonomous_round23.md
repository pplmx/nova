# RIL — Round 23 (2026-08-12) — milestone v2.23 "Tensor-Parallel Layer Backward Passes"

## Round focus

**Open milestone v2.23** on the forward-only gap: the v2.21 weight-managed
layers (and the whole parallel stack) compute no gradients — only loss and
optimizers exist, so the parallel path is inference-only. This milestone adds
`backward()` to ColumnParallelLayer / RowParallelLayer / TensorParallelMLP with
analytic reference parity on real multi-GPU (TASK-022).

## Observation / Model

- Backward math is clean and independently checkable:
  - ColumnParallelLayer (input replicated, output sharded): `dW = X^T dY`;
    the grad-input is the **AllReduce of the per-rank `dY W_r^T`** (replicated
    input ⇒ gradient must sum over ranks) — a real collective never exercised
    anywhere in the stack (the never-run-surface class this thread keeps
    surfacing bugs in).
  - RowParallelLayer (input sharded, output replicated): `dW = X^T dY`;
    grad-input is local — `dY W_r^T` is exactly rank r's block of the full
    grad (matches the sharded-input layout), no comm.
  - TensorParallelMLP chains: down row-backward (grad-sub sharded) →
    silu_and_mul backward (new kernel: `dgate = dsub .* up .* silu'(gate)`,
    `dup = dsub .* silu(gate)`) → gate/up col-backward, whose grad-inputs sum
    and AllReduce to the full replicated grad-input.
- Reference = host double-precision analytic gradients sliced to the shard
  layout, mirroring the v2.19-22 reference-parity pattern.

## Decisions

- **DEC-009**: milestone direction — layer backward passes with analytic
  reference parity. Rejected: end-to-end TensorParallelAttention composition
  (capstone but heavier geometry/heads; more harness than capability); a full
  autograd graph framework (too broad; layer-math parity is the immediate need).

## Execute — Phase 1 (TASK-019, RED evidence)

- Header: added `backward(input, grad_output, grad_input, grad_weight, batch,
  seq)` to ColumnParallelLayer (grad-input AllReduce contract) and
  RowParallelLayer (local grad-input shard contract); `backward(...)` to
  TensorParallelMLP (full chain); ColumnParallelLayer gains a `reducer_`
  (NcclAllReduce) for the P2 grad-input AllReduce.
- Provisional stubs: each backward throws "not implemented (TASK-020)".
- Tests (6): single-GPU col/row/MLP backward vs host double-precision analytic
  gradients; multi-GPU col grad-input AllReduce parity + assembled grad-weight
  slices, row grad-slice parity, full gated-MLP chain parity (grad-input +
  gate/up/down grad-weight assembly).

## Verify — Phase 1

**6/6 RED** (all throw the provisional stub): 3 single-GPU on device 2, 3
multi-GPU on CUDA_VISIBLE_DEVICES=2,3 (EV-015). Forward suites unaffected.

## Execute — Phase 2 (TASK-020, `CHG-012` 7855ab3)

- New kernels in activations: `silu_and_mul_backward` (dgate = dsub .* up .*
  silu'(gate), dup = dsub .* silu(gate)), `elementwise_add`, `transpose` (+
  unit tests). Transposing W/X turns `dW = X^T dY` / `dX = dY W^T` into the
  library's verified non-transposed matmul convention.
- `ColumnParallelLayer::backward`: dW = X^T dY written as the rank's column-slice
  (strided `cudaMemcpy2DAsync` scatter); grad-input = dY W^T, then **AllReduce**
  across ranks (the new never-run collective; `NcclResult` checked → `NcclException`).
- `RowParallelLayer::backward`: dW = X^T dY (contiguous row-block copy); grad-input
  local = dY W_r^T (block of the full grad, matching the sharded input layout).
- `TensorParallelMLP::backward`: down row-backward → silu_and_mul_backward → gate/up
  column-backward; the two full grad-inputs summed (`elementwise_add`).

## Verify — Phase 2/3 (EV-016)

- Single-GPU backward 3/3 + activation helpers 3/3 GREEN.
- Multi-GPU backward **3/3 GREEN on 2 GPUs and 4 GPUs** — col grad-input AllReduce
  parity, row grad-slice parity, MLP gated-chain parity.
- Isolated distributed cross-suite (matmul + layers fwd/bwd + sequence parallel)
  **19/19 on 2 & 4 GPUs**; single-GPU regression 49/49; full-suite baseline 1422/0.
- **Test-side bug (not impl)**: the first multi-GPU row-backward run "failed" because
  the test compared the rank-local [m x k_local] grad-input against a bogus contiguous
  pointer into the [m x k] reference (the feature block strides by the full row length
  k). Extracting the strided block fixed it; the implementation was already correct.
  Col/MLP passed from the start (their grad-input is the full AllReduced matrix).

## Learn

- The column-layer grad-input AllReduce is a genuinely new collective for the stack;
  its reference parity + `NcclResult` propagation is the thing that future training
  relies on. The row layer needs no comm in backward — a good asymmetry to keep.
- A full [m x k] reference can't be viewed as contiguous blocks along k; sub-views of
  row-major matrices along the feature dimension must be extracted stride-aware.

## Milestone close

- `TASK-019/020/021/022` resolved; DEC-009; EV-015/EV-016; CHG-012. Round bumped.
- The stack is now forward+backward capable. Next milestone (natural candidate):
  end-to-end training integration — loss → backward → optimizer stepping the parallel
  layers, or attention-block integration of forward+backward.
