# RIL — Round 22 (2026-08-12) — milestone v2.22 "Ring Sequence Parallelism On Real Multi-GPU"

## Round focus

**Open milestone v2.22** on the last dispositioned-but-unimplemented parallel
surface: `RingSequenceParallelism::ring_attention` (TASK-014). v2.21 closed the
weight-managed `TensorParallelLayers` thread (TASK-010); this round opens the
next milestone and starts P1 (RED evidence).

## Observation / Model

- `RingSequenceParallelism`: prev_rank_/next_rank_ wired; `send_recv_kv` declared,
  never defined. multi-GPU `ring_attention` throws (DEC-004 fail-fast); pre-DEC-004
  it was a silent no-op returning garbage (issue-v19-ring-parallel-noop).
- The correct algorithm: each rank owns a local Q/K/V sequence shard; iterate
  P-1 times sending its KV block to next_rank and receiving prev_rank's,
  accumulating partial attention over each received block with **online softmax**
  (carry running max + denominator). Final per-rank output = its local query
  tokens attended over the FULL KV sequence across all ranks.
- Acceptance oracle: standard scaled dot-product attention over the concat of
  all ranks' K/V, computed in double precision on host — unambiguous and
  independent of the ring/collective path.

## Decisions

- **DEC-008**: milestone direction — implement ring attention for real. Rejected:
  keeping fail-fast (leaves the last parallel surface dead); AllGather-then-attend
  (proves nothing about the ring algorithm / defeats its memory purpose); a
  comm-null implementation (would not verify the real P2P path).

## Execute — Phase 1 (TASK-015, RED evidence)

Added `SequenceParallelMultiGpuTest.MultiGpu_RingAttention_MatchesFullSequenceReference`
to `tests/distributed/sequence_parallel_multigpu_test.cu`: per-rank random local
Q/K/V [local_seq x hidden], reference = local queries attended over all ranks'
concatenated K/V at 1/sqrt(hidden) scale, compared per rank.

## Verify — Phase 1

On `CUDA_VISIBLE_DEVICES=2,3` + NCCL: the parity test is **RED** (ring_attention
throws "not implemented"); the existing `MultiGpu_RingAttentionNotImplemented`
fail-fast pin stays **green** (P2 will drop it when the real algorithm lands).

## Execute — Phase 2 (TASK-016, `CHG-010` 9d91be1)

- Added `SequenceParallelConfig.hidden_dim` (required by the multi-GPU ring path to
  interpret Q/K/V buffers; unset → fail fast).
- New `src/cuda/distributed/ring_attention.cu`: `ring_attn_init/block/finalize` —
  online softmax with max-rescale (corr = expf(m - s) applied to running accumulation
  and denominator), one thread per query token.
- `RingSequenceParallelism::ring_attention` multi-GPU: P-1 steps, each sending the
  local block clockwise to next_rank and receiving prev_rank's block (NCCL P2P
  `ncclSend`/`ncclRecv` on the group comm), accumulating every block into the running
  softmax state; `std::swap` buffers; finalize divides by l_i.
- **Deadlock finding (EV-012)**: ungrouped consecutive NCCL P2P ops serialized on the
  communicator and hung the parity test (~100% CPU); wrapping the four ops in
  `ncclGroupStart/End` fixed it. This is a real bug the reference-parity RED test caught.
- Replaced the obsolete `MultiGpu_RingAttentionNotImplemented` pin (its meaning died
  with the throw) with `RejectsMissingHiddenDim`.

## Verify — Phase 2/3 (EV-012 / EV-013)

- `MultiGpu_RingAttention_MatchesFullSequenceReference` **GREEN** on 2 GPUs and on 4
  GPUs (multi-hop ring, P=4) — per-rank local-query output == full-sequence attention.
- Isolated distributed cross-suite (TensorParallelMultiGpuTest + SequenceParallelMultiGpuTest):
  **16/16 on 2 GPUs and 4 GPUs**. Single-GPU seq-parallel regression **33/33**.
- Full-suite baseline (NCCL unset): **1416 pass / 0 fail** (1 pre-existing DISABLED).

## Learn

- The parity RED test turned a subtle NCCL P2P deadlock into a 30-second diagnosis
  (hang → group wrap → green): never run consecutive `ncclSend`/`ncclRecv` outside an
  `ncclGroup` on the same communicator.
- Online-softmax accumulation verified exactly matches a double-precision full softmax
  within 1e-2 — the rescaled-max path is numerically sound for this scale.

## Milestone close

- `TASK-014`/`TASK-015`/`TASK-016`/`TASK-017` resolved; `issue-v19-ring-parallel-noop`
  resolved; DEC-008; EV-012/EV-013; CHG-010. Round bumped.
- Next milestone: TBD — the folded parallel-training stack (TP layers + SP attention +
  ring attention) could now be composed/tested end-to-end, or attention integration into
  the transformer block; a gradient/backward pass would extend to training.
