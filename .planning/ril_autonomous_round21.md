# RIL — Round 21 (2026-08-12) — milestone v2.21 "Weight-Managed TensorParallelLayers"

## Round focus

**Open milestone v2.21** by driving `TASK-010` (the only `active` high-priority task,
8.40) to resolution: the v1.3 `TensorParallelLayers` stubs — disposed as fail-fast in
v2.20 (DEC-006, issue-v20-tp-layers-stubs) because they owned no weight storage and
passed `B = nullptr` to cuBLAS — get a real weight-managed implementation with
single-GPU / multi-GPU **reference parity** as the acceptance gate.

## Observation / Model

- `TASK-010` describes exactly a feature: "implement weight-managed
  ColumnParallelLayer/RowParallelLayer/TensorParallelMLP forward with real weight
  buffers and multi-GPU reference parity".
- The verified `TensorParallelMatmul` primitive's semantics (Column → replicated full
  width via AllReduce; Row → batch-split) do **not** match the layers' documented
  semantics (column shards/hidden-per-tp output; row consolidates to full width) and
  cannot compose a coherent MLP — so the layers need true **weight-sharded** semantics,
  reusing the v2.20 conventions (rank via `rank_of_device`, per-instance stream-bound
  cuBLAS handle, checked `NcclResult`).
- Layers have zero callers in the tree (grep-verified) → API redesign is safe.
- Reconfirmed the pre-existing `issue-v19-shared-nccl-context-reset`: the NCCL multi-GPU
  cross-suite hangs at ~100% CPU when `cudaDeviceReset`-heavy suites (ActivationTest)
  run interleaved in the same process; the isolated curated filter is unaffected.

## Decisions

- **DEC-007**: milestone direction — implement the layers with Megatron weight-shard
  semantics (col shard → local matmul, row shard → local + AllReduce, gated MLP), new
  explicit `(ctx, in_features, out_features)` API, diff-size exception (>300-line single
  commit) documented because this replaces a stub surface + adds tests + a kernel.
  Rejected: keeping fail-fast (dead surface); re-exposing the primitive with replicated
  full weights (defeats weight sharding, incoherent MLP); keeping the rigid 3*hidden QKV
  column API.

## Execute — Phase 1 (TASK-011, RED evidence)

- Rewrote `tests/neural/tensor_parallel_layers_test.cpp` as functional single-GPU tests:
  col/row/MLP forward == single-GPU full-weight reference (tp<=1 via uninitialized ctx),
  `set_weight` overwrites the buffer, getters, + `silu_and_mul` unit tests in
  `activations_test.cpp` (host formula + negative-gate).
- Added multi-GPU parity tests to `tensor_parallel_multigpu_test.cpp`: column shard
  concat == reference, column non-divisible out rejects, row AllReduce replicated ==
  reference, row non-divisible in rejects, gated MLP == single-GPU full-weight reference.
- Provisional throw-stub `.cpp` + new header → built: **4/4 layer + 2/2 silu tests RED**
  (all threw "not implemented").

## Execute — Phase 2 (TASK-012, implementation)

- `silu_and_mul` gated-activation kernel/launcher (activations).
- Rewrote `tensor_parallel_layers.{h,cpp}`: per-rank device weight shards
  (`set_weight` slices the full row-major weight by the `rank_of_device` membership
  rank), col/row/gated-MLP forward (col = local shard matmul → sharded output; row =
  local matmul + in-place block AllReduce → replicated output with `NcclResult` checked;
  MLP = gate/up col proj + `silu_and_mul` + down row proj with growable scratch),
  per-instance stream-bound cuBLAS handle, divisibility + has_nccl fail-fast guards.

## Verify — Phase 2 (EV-010 / EV-011)

- Single-GPU: 4/4 layer + 2/2 silu GREEN.
- Real multi-GPU (`CUDA_VISIBLE_DEVICES=2,3`, `NCCL_TESTS_AVAILABLE=1`): 5/5 layer parity
  GREEN (column shard concat, row replicated, gated MLP, both shape rejections).
- Isolated NCCL cross-suite (TensorParallelMultiGpuTest + SequenceParallelMultiGpuTest):
  **15/15 GREEN on 2 GPUs and on 4 GPUs** (tp=4 column/row/MLP parity).
- Full-suite baseline (NCCL unset): **EXIT=0** (multi-GPU suites skip cleanly).
- Regression: pre-existing core-dump path only reproduces when NCCL env is set to a
  non-null value AND `cudaDeviceReset` suites interleave — the documented v19 flake.

## Learn

- Making the layer tests RED against a throw-stub + then green on real hardware pinned
  the exact shard math v2.19 proved the pre-v2.20 code got wrong (contiguous column-major
  slicing); the parity tests are now the regression gate.
- The row-parallel AllReduce is the only real comm in the stack; column-parallel and the
  MLP composition verified the weight-shard + scratch-accounting paths.
- Running NCCL multi-GPU suites interleaved with `cudaDeviceReset` suites re-hangs at
  ~100% CPU — consistent with issue-v19-shared-nccl-context-reset; the isolated curated
  filter is the supported way to run the cross-suite.

## Milestone close

- `TASK-011`/`TASK-012`/`TASK-013` resolved; `TASK-010` resolved; `issue-v20-tp-layers-stubs`
  resolved; DEC-007 recorded; EV-010/EV-011 recorded.
- Next milestone: TBD — natural candidates remain the never-run high-level surfaces
  (gather-output column variant, gradient/backward passes, attention integration).
