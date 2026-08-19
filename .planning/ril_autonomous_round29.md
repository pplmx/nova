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

## P3 verify + cpp-reviewer (TASK-046, EV-026/027)

- cpp-reviewer pass on 377209e..3af650a caught **1 HIGH (racecheck-verified,
  compute-sanitizer `--tool racecheck`)**: every block-reduction helper
  (`block_reduce_sum`/`block_reduce_max` in attention_kernels.cu,
  `ln_block_reduce_sum` in layer_norm.cu) both returns `red[0]` after its
  final `__syncthreads()` AND writes `red[wid]` at its start — so a
  *subsequent* call on the same buffer could overwrite `red[0]` while other
  threads still read the previous call's result. Masked natively by warp
  lockstep (25/25 runs green), the reviewer reproduced a numeric failure on
  the large-seq backward under instrumentation. Fixed by adding a load-bearing
  entry `__syncthreads()` at the top of all three helpers. The reviewer also
  confirmed-OK the shuffle masks, shared atomics, global dk/dv atomics, layout,
  and edge cases (m/h < block, m not a multiple of block).
- **MEDIUM** (signed `int` overflow): `m * local_heads` (SDPA grids) and
  `m * h` (LayerNorm element counts/grids) computed in `int` could wrap and
  silently under-cover the grid. Fixed with 64-bit products + a
  `gridDim.x`-max guard (`layer_norm_bwd_grid` helper, `sdpa_*_device` grid
  validation).
- **LOW**: forward shared over-allocation dropped; `-1e30f` sentinel kept
  (parity-exact with the deleted host code); include hygiene (`<string>` added,
  unused `buffer.h`/`<numeric>` dropped).
- Verify after fixes (EV-027): neural single-GPU regression **79/79**; neural
  multi-GPU cross-suite + deep-block **19/19 on 2 GPUs and 19/19 on 4 GPUs**
  (DeepBlockMultiGpuPinnedTest 1/1 on 2 & 4 GPUs — K-step shard==single-GPU
  trajectory parity unchanged through device SDPA + parallel LN). Milestone
  **closed** (CHG-021 fix commit).

## Graph additions (Round 29, opening)

- `DEC-015` (decision, milestone direction); `TASK-043/044/045/046` (umbrella,
  P1/P2/P3); `issue-v29-sdpa-host-roundtrip`, `issue-v29-ln-one-thread-per-row`
  (the two addressed debts).

## P1 RED (TASK-044, EV-025)

- Added the contract surface with provisional throw-stubs: `detail::` device
  entries in `tensor_parallel_attention.h` (`sdpa_forward_device` /
  `sdpa_backward_device`) and `layer_norm.h`
  (`layer_norm_forward_trainable` / `layer_norm_backward_trainable`), a new
  `attention_kernels.cu` module (throw-stubs, wired into CMake, mirroring
  `optimizers_kernels.cu`), new `attention_kernels_test.cpp` (host-fp64 SDPA
  forward/backward references) + 2 parallel-LN tests appended to
  `layer_norm_trainable_test.cpp`.
- RED run (EV-025): 6/6 contract tests fail for the right reason — 4
  `AttentionKernelsTest` (fwd/bwd at m=6 seq>1 heads=3 hd=8, and large m=128
  heads=2 hd=16) + 2 `LayerNormTrainableTest.Parallel*` (fwd at hidden=512,
  bwd at hidden=1024) all throw "not implemented (v2.29 P1 stub)". All 6
  existing LayerNormTrainableTest tests + TensorParallelAttentionTest still
  pass — the public `sdpa_forward`/`sdpa_backward`/`LayerNorm` paths are
  untouched (only the new detail:: entries throw).
- Commit `test(neural) P1 RED for milestone v2.29 ...` (377209e).

## P2 implementation (TASK-045, CHG-020 3af650a)

- `attention_kernels.cu`: device `sdpa_forward_kernel` (per-(position,head)
  block; per-row max over keys chunked by 256-thread blocks with one block
  reduction, then exp-normalized V accumulation into a shared out-accumulator
  with float atomics; m bounded only by the key count) and
  `sdpa_backward_kernel` (3 passes: max, esum + weighted-num, then dscore and
  dV/dK global-atomics across query positions with dQ shared-reduced; P and
  dP are RECOMPUTED in the final pass rather than stored — a shared scratch
  would be raced by concurrent (position, head) blocks, found during P2).
  `sdpa_forward`/`sdpa_backward` route to the detail:: entries; the host
  D2H/H2D round-trips + CPU implementations deleted (issue-v29-
  sdpa-host-roundtrip resolved).
- `layer_norm.cu`: `ln_forward_trainable_kernel` and `ln_backward_stats_kernel`
  switched from one-thread-per-row to one-BLOCK-per-row (kLnBlock=256) with an
  `ln_block_reduce_sum` shuffle+shared reduction for the row mean/variance and
  the backward stats; `LayerNorm::forward/backward`, the free
  `layer_norm_backward`, and the new `detail::layer_norm_forward_trainable` /
  `layer_norm_backward_trainable` all use the block-reduction path
  (issue-v29-ln-one-thread-per-row resolved). Debugging caught two real bugs:
  forward's variance was used UN-reduced (thread 0's partial) making inv ~25x
  too large — fixed by `var = ln_block_reduce_sum(var, red)`, and the same
  latent fault in the backward stats kernel; both confirmed by a temporary
  device printf before the fix.
- GREEN (EV-026): the 6 P1 contracts 6/6 — AttentionKernelsTest 4/4
  (max_abs ~3e-8 vs the fp64 references after the scratch-race fix; before it
  the large backward showed ~0.1 errors from the cross-block psave/dpsave
  race), LayerNormTrainableTest.Parallel\* 2/2 (max_abs ~5e-7); neural
  single-GPU regression 79/79; neural multi-GPU cross-suite 23/23 on 2 GPUs
  and 18/18 on 4 GPUs; DeepBlockMultiGpuPinnedTest
  (MultiGpu_DeepTransformer_MatchesSingleGpu) 1/1 on 2 & 4 GPUs — the K-step
  shard==single-GPU trajectory parity is UNCHANGED through the device SDPA +
  parallel LN (the definitive proof the new kernels don't perturb training).
