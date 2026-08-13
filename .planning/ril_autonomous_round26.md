# RIL — Round 26 (2026-08-13) — milestone v2.26 "Tensor-Parallel Multi-Head Attention + Mini-Transformer Training"

## Round focus

**Open and close milestone v2.26** — the DEC-010/011-rejected attention
capstone, now that MicroTrainer (v2.25) makes it trainable. The parallel stack
could train a plain MLP but had NO attention: the single-GPU MultiHeadAttention
was an incomplete shell (issue-v24-mha-incomplete). This milestone builds
attention on the verified v2.21-25 layer conventions and proves a
mini-transformer (attention + MLP) trains sharded == full on real multi-GPU
(TASK-031).

## Observation / Model

- v2.25 evidence: MicroTrainer trains a tensor-parallel MLP and converges; but
  no attention exists on the parallel path — the stack cannot form a
  transformer.
- Design (DEC-012): Q/K/V as ColumnParallelLayer (each rank owns contiguous
  head columns -> the rank's heads are complete locally, so the per-head SDPA
  needs NO cross-rank communication), output as RowParallelLayer (one
  block-wise AllReduce -> replicated output). This is Megatron-style head
  sharding and reuses every verified primitive:
  - sdpa_forward/backward: new host-side correctness-first SDPA primitives.
  - backward: output row-layer backward -> local head-grads; SDPA backward ->
    dQ/dK/dV (analytic scaled-dot-product chain); QKV col-layer backward ->
    per-rank shard grads + AllReduced replicated-input grad-input, summed to
    the full block grad-input (the v2.23 pattern).
  - step(): one AdamW per weight tensor (q/k/v/o) — the per-tensor-moment
    convention from the v2.24 cpp-review.
- Mini-transformer: logits = MLP(attn(X)) composed on the existing verified
  layers + the new attention block; trained via the MicroTrainer chain.

## Decisions

- **DEC-012**: milestone direction (see above). Rejected: device-native fused
  AdamW (perf, orthogonal), a full mini-language-model (needs attention first
  — attention is the dependency, so it is this milestone).

## Execute (TASK-033, `CHG-015` bc02049)

- `tensor_parallel_attention.h/.cpp`: TensorParallelMultiHeadAttention (QKV
  col-parallel, multi-head SDPA fwd/bwd, output row-parallel), step() (one
  AdamW per weight tensor), copy_weights read-back. sdpa_forward/backward are
  host-side (D2H/H2D) correctness-first primitives — deliberately no dead
  device kernel (the issue-v24-mha-incomplete shell is the anti-pattern).
- Mini-transformer training path (attention + verified MLP).
- Test-reference bugs fixed along the way: the shared key/value stride was
  position-major (heads*hd), not head-major — a genuine layout bug in the
  first implementation caught by reference parity; and the backward test
  references reused softmax probs across heads (fixed to per-head recompute).

## Verify (EV-020)

- Single-GPU 3/3: forward and backward vs host fp64 reference; standalone SDPA
  fwd/bwd; mini-transformer training convergence (loss 106.6 -> 1.35 over 40
  steps, accuracy 0.06 -> 0.56).
- Multi-GPU 3/3 on **2 & 4 GPUs**: assembled attention forward, assembled
  backward weight-grad shards, and `MultiGpu_MiniTransformer_MatchesSingleGpu`
  (K=12, 7 weight tensors per rank, attention+MLP) — assembled shards == the
  single-GPU full-weight reference run.
- Cross-suite **43/43 on 2 & 4 GPUs** (was 40; +3 attention multi-GPU);
  single-GPU neural regression **43/43** (was 40; +3 attention).

## Learn

- Head sharding makes per-head attention collective-free: the only comm in the
  block is the output projection's AllReduce. This is the same "keep comm at
  the boundary" shape the MLP already used (row-layer AllReduce only).
- A second real bug was caught by reference parity (key/value stride); the
  value of an independent host reference over the device output proved itself
  again. The residual test-reference bugs (shared probs across heads, clobbered
  grad_output by a reused forward-out buffer) are the cost of hand-rolled
  references — worth asserting each intermediate against a cleaner stage test.

## Milestone status

- Opened and P2 implemented + verified (EV-020). Tasks TASK-031/032/033/034
  resolved. Pending: cpp-reviewer disposition of CHG-015 + round doc close
  commit.
