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

## Milestone state

v2.23 OPENED in STATE.md / PROJECT.md (In Progress). RIL: DEC-009,
TASK-019/020/021/022 active, EV-015, graph consistent (98+ nodes). Next phase:
P2 implementation (TASK-020) — transposed GEMMs + col grad-input AllReduce +
silu_and_mul backward kernel.
