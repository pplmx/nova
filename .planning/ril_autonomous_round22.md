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

## Milestone state

v2.22 OPENED in STATE.md / PROJECT.md (In Progress). RIL: DEC-008,
TASK-014/015/016/017 active, graph consistent (87 nodes, 79 edges), round=21.
Next phase: P2 implementation (TASK-016) — send_recv_kv + online-softmax ring.
