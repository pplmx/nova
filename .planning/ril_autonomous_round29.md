# RIL — Round 29 (2026-08-19) — milestone v2.29 "Device-Native Scaled-Dot-Product Attention + LayerNorm Training Kernels"

## Round focus

**Open milestone v2.29** — the performance round Round 28's LEARN named next
("device-native attention + LN training kernels … on the v2.28 stack"). The
v2.28 stack now *trains* a deep normalized transformer correctly (N-block,
LayerNorm, one AdamW per tensor, shard==single-GPU parity on 2 & 4 GPUs), but
every train_step pays a host round-trip for attention: `sdpa_forward` (3 D2H +
host loops over m×local_heads×m×head_dim + 3 H2D) and `sdpa_backward` (4 D2H +
3 H2D) run once per block per step, and the trainable LayerNorm kernels launch
one thread per row. This milestone removes that binding training-latency
constraint by moving SDPA to device kernels and parallelizing the LN kernels —
the exact D2H/H2D-debt family v2.27 already removed from the optimizer
(TASK-043, DEC-015; the v2.26 attention header explicitly deferred SDPA to "a
later performance pass").

## Observation / Model

- **M3 (binding)**: `tensor_parallel_attention.cpp` computes multi-head SDPA
  on the host — `sdpa_forward` copies Q/K/V D2H, runs the CPU loops (m ×
  local_heads rows, each an O(m×head_dim) attention), copies out H2D;
  `sdpa_backward` the same (4 D2H + 3 H2D). `TransformerBlock::forward` calls
  attention.forward once, so a deep trainer runs this round-trip N× per step
  (v2.28: 3 blocks → 3× forward + 3× backward = 30 blocking memcpys per step,
  all on the replicated Q/K/V head shards that never leave the GPU). Grows
  linearly in blocks and quadratically in seq. The file header marks it
  explicitly: "a later performance pass can move it to device kernels".
- **M2**: `ln_forward_trainable_kernel` and `ln_backward_stats_kernel` are
  launched `<<<m, 1>>>` — one thread per row sums the hidden dim serially.
  Fine at hidden≤64 (all current tests), a serialization bottleneck at
  hidden≥512. `ln_backward_input_kernel` is already element-parallel (256).
- **M1**: the free `layer_norm_backward` allocates its per-call stats buffer
  every call (the `LayerNorm` class already caches via `ensure_stats`).
- **Convention (mirrors v2.27)**: keep the public APIs (`sdpa_forward`,
  `sdpa_backward`, `LayerNorm::forward/backward`, free `layer_norm_backward`)
  unchanged; route them through a new `detail::` device-kernel module
  (`attention_kernels.cu`, like `optimizers_kernels.cu`), delete the host
  round-trips + dead CPU implementations, and verify by GPU-vs-fp64-reference
  parity + unchanged training-trajectory parity. Parity tolerance convention is
  already 1e-4 (SDPA) / 1e-4 (LayerNorm) — absorbs fp32 reduction-order
  differences from the new blocked reductions.

## Decisions

- **DEC-015**: milestone direction — device-native SDPA forward+backward
  kernels behind the same `sdpa_forward`/`sdpa_backward` API (removing the
  per-block host round-trip, M3) + parallelized trainable LayerNorm
  forward/backward kernels via block/warp reductions (M2), on/behind the v2.28
  stack; verified by GPU-vs-fp64 parity and unchanged 2/4-GPU
  shard==single-GPU K-step trajectory parity. Rejected: LAMB kernelization
  (orthogonal perf, still deferred as in DEC-014); deeper scale-up first
  (more blocks/longer seq would multiply the host round-trip — the bottleneck
  grows with the workload); fusing SDPA into the TP layer (couples modules —
  SDPA stays a standalone detail:: kernel module like optimizers_kernels);
  keeping the host CPU SDPA as a fallback (dead code — independent fp64
  references live in the tests).

## Milestone definition (opened)

Phases (TASK-043 umbrella, TASK-044/045/046):

- **P1 (RED/parity, TASK-044)** — pin the new contract surface RED against
  throw-stubs: `detail::sdpa_forward_device` / `detail::sdpa_backward_device`
  (new attention_kernels module) vs an in-test fp64 SDPA reference at
  seq>1, heads>1, head_dim>1 (out/dq/dk/dv); parallelized trainable LayerNorm
  forward/backward entries vs the fp64 reference at hidden∈{512,1024}, rows>1
  (exercising the block-reduction path).
- **P2 (implementation, TASK-045)** — `attention_kernels.cu`: device SDPA
  forward + analytic backward kernels (per-row shared-memory softmax,
  block/warp reductions over keys/values, exact port of the host algorithm so
  no reference-copy semantics change), wired into `sdpa_forward`/`sdpa_backward`
  with the host round-trip and CPU impls deleted; parallelize
  `ln_forward_trainable_kernel`/`ln_backward_stats_kernel` with block/warp
  reductions and route `LayerNorm::forward/backward` + free
  `layer_norm_backward` through the block-reduction path (absorbing M1's per-
  call buffer where the class already caches).
- **P3 (verify, TASK-046)** — kernel-level SDPA + parallel-LN parity GREEN;
  all existing neural single-GPU suites GREEN (LayerNormTrainableTest,
  TransformerBlockTest, attention, MicroTrainer); neural multi-GPU cross-suite
  19/19 on 2 & 4 GPUs re-run untouched (deep-block K-step trajectory parity
  unchanged — proves device SDPA doesn't perturb training); cpp-reviewer pass;
  RIL close. Full-suite baseline still not reproducible in this environment
  (env memory).

## Graph additions (Round 29, opening)

- `DEC-015` (decision, milestone direction); `TASK-043/044/045/046` (umbrella,
  P1/P2/P3); `issue-v29-sdpa-host-roundtrip`, `issue-v29-ln-one-thread-per-row`
  (the two addressed debts).
