# RIL — Round 32 (2026-08-24) — milestone v2.33 "Multi-GPU Rank-Shard Checkpoint Restore"

## Round focus

**Open milestone v2.33** — completes the v2.32 persistence milestone on the
multi-GPU path. v2.32 shipped exact trainer checkpointing (save_state /
load_state + AdamW moment export/import, NSCK v1) but **explicitly guarded
tp > 1**: with TP>1 the `copy_*` readbacks expose rank shards while the `set_*`
uploaders take *full* weights and slice this rank's shard (ColumnParallelLayer
slices by rank, RowParallelLayer by input), so `copy_*`→`set_*` is asymmetric
away from tp=1 — and a rank-local checkpoint could not restore. That guard was
documented as the genuine follow-up (TASK-059, 6.0). This round removes it:
rank-shard checkpoint restore on real multi-GPU.

## Observation / Model

- **M1 (binding)**: `save_state`/`load_state` throw on tp > 1 (the v2.32
  guard). A 2+ GPU training run cannot persist or resume — the very
  interruption-resume use case checkpointing exists for. Same core-feature
  family as v2.32 (ISS-001's remaining half).
- **M2 (shard layout)**: each trainer's rank shards are derivable from the
  geometry — MicroTrainer gate/up `hidden×(inter/tp)`, down `(inter/tp)×hidden`;
  TransformerBlock wq/wk/wv `hidden×(qkv/tp)`, wo `(qkv/tp)×hidden`, wg/wu
  `hidden×(inter/tp)`, wd `(inter/tp)×hidden`, LN gamma/beta replicated `hidden`
  (verified against ColumnParallelLayer / RowParallelLayer / TP-attention
  layouts in P2 OBSERVE). `copy_weight*` already returns exactly these;
  the optimizer moment capacity equals the shard size too.
- **M3 (restore primitive)**: a no-slice shard uploader is literally the
  tp=1 branch of the existing `ColumnParallelLayer::set_weight` / row layer —
  extract it as `set_weight_shard`, then `TensorParallelMLP::set_weight_shards`
  (3) and `TransformerBlock::set_weight_shards` (11: 4 replicated LN + 7
  sharded matmul) compose them. `load_state` reads the count-tagged NSCK-v1
  records (count = this trainer's shard counts at tp>1, full at tp=1),
  validates, then applies via the shard setters.
- **M4 (semantics)**: rank-local same-topology restore — a checkpoint written
  on a given topology loads onto an identically-geometried trainer (the
  resume-after-interruption case); the cross-topology "distribute a tp=1
  checkpoint onto N ranks" (full→shard) and the "gather 1 checkpoint" (NCCL
  all-gather on save) features stay out of scope. Count-tagged records reject
  a geometry mismatch (a tp=1 file into a tp>1 trainer, or vice versa), so no
  silent mis-restore.

## Decisions

- **DEC-019**: milestone direction — remove the tp>1 checkpoint guard: rank
  shard setters (`ColumnParallelLayer`/`RowParallelLayer::set_weight_shard`,
  `TensorParallelMLP::set_weight_shards`, `TransformerBlock::set_weight_shards`)
  + rank-aware `save_state`/`load_state` (sizes from dims/tp; load applies via
  the shard setters, geometry-validated). Verified by byte-exact per-rank
  roundtrip + resumed-vs-uninterrupted on real 2-GPU trains (MicroTrainer
  single-trainer-over-ctx style; deep transformer thread-per-rank style),
  and the existing tp=1 CheckpointTest suite still byte-identical at tp=1.
  Rejected: full-weight cross-rank gather via NCCL on save (in-scope only for
  a single-checkpoint-per-cluster feature — separate follow-up); leaving the
  guard in place (the interruption-resume use case is the point of the
  preceding milestone); cross-topology load (distributing tp=1 checkpoints —
  out of scope).

## Milestone definition (opened)

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: 2 multi-GPU checkpoint contracts (MicroTrainer rank-local save-load roundtrip + resumed-vs-uninterrupted; TransformerTrainer per-rank same) fail against the tp>1 guard | In progress (Round 33) |
| 2 | Implement: layer shard setters + TensorParallelMLP/TransformerBlock set_weight_shards; rank-aware save/load sizing; drop the tp>1 guards; route load through shard setters | Pending |
| 3 | Verify: 2-GPU checkpoint contracts GREEN; tp=1 CheckpointTest 8/8 unchanged; neural single-GPU + multi-GPU regressions; cpp-reviewer; RIL close; ISS-001 resolved | Pending |

**Graph additions (Round 33, opening):**
- `DEC-019` (decision, milestone direction); `TASK-060` (P1) / `TASK-061`
  (P2) / `TASK-062` (P3), each depends_on the TASK-059 umbrella (which also
  keeps its address to ISS-001).

## P1 RED (TASK-060)

- Added `CheckpointMultiGpuTest` (tests/neural/checkpoint_multigpu_test.cpp,
  registered in tests/CMakeLists.txt) with 2 contracts over a real 2-GPU NCCL
  context (thread-per-rank via run_per_rank, matching the MicroTrainerMultiGpu
  / DeepBlockMultiGpu style):
  - `MultiGpu_MicroTrainer_RankLocalCheckpoint`: per-rank MicroTrainer, train
    K steps, save_state -> fresh same-rank trainer load_state, rank-shard
    copy_weights byte-identical, resumed losses == uninterrupted holdout.
  - `MultiGpu_TransformerTrainer_RankLocalCheckpoint`: per-rank
    TransformerTrainer, same contract over all 11N+2 tensors (shard-sized
    matmul + replicated LN), resumed deep trajectory matches.
- RED run (EV-041): 2/2 fail for the right reason — both hit the tp>1
  save_state guard ("...v2.32 follow-up (TASK)..."). Harness detail recorded
  as EV-042: the fixture SetUp must `DeviceMesh::instance().initialize()`
  (device_count reads 0 before init), same as MicroTrainerMultiGpuTest.

## P2 implementation (TASK-061, EV-043)

- Shard setters (the no-slice dual of the existing copy_weight_shard /
  copy_weights): `ColumnParallelLayer` / `RowParallelLayer::set_weight_shard`
  (the tp=1 branch of set_weight extracted; self-sizing, rejects nothing —
  copies the rank shard exactly), then `TensorParallelMLP::set_weight_shards`
  (3), `TensorParallelMultiHeadAttention::set_weight_shards` (4), and
  `TransformerBlock::set_weight_shards` (11: the 7 matmul shards + the 4
  replicated LN gamma/beta via the full set_weight — checkpoint LN records are
  full hidden).
- Rank-aware checkpoints: `MicroTrainer::save_state` sized to
  hidden×(inter/tp) / (inter/tp)×hidden; `TransformerTrainer`'s
  `block_tensor_sizes` takes tp (matmul /tp, LN full). The tp>1 guards are
  dropped from both trainers' save/load; `load_state` validates every record
  against THIS trainer's shard sizes (a tp=1 file won't load on a tp>1 trainer
  or vice versa — only the same topology roundtrips — geometry rejected, never
  mis-restored) and applies via the shard setters. Public trainer APIs
  unchanged (the setters compose inside load_state).
- GREEN (EV-043): CheckpointMultiGpuTest **2/2** on 2 real GPUs — per-rank
  save→load→resume byte-exact shards + resumed-vs-uninterrupted on both
  trainers; tp=1 CheckpointTest **8/8** unchanged (shard==full);
  neural single-GPU **115/115**; multi-GPU cross-suite **21/21 on 2 GPUs**
  (existing trajectory/parity suites + the new checkpoint suite), deep-block
  K-step trajectory unchanged.

## P3 verify (TASK-062)

- CheckpointMultiGpuTest GREEN on **4 GPUs** too (2/2 — the rank-shard math
  generalizes to tp=4; inter/4 and qkv/4 divisibility holds); full multi-GPU
  cross-suite on 4 GPUs **7/7** (incl. deep-block trajectory).
- Full-suite in-process baseline: **1469 PASSED / 0 FAILED** (1547 ran, the
  MPI/multi-GPU launcher-gated suites skipped as usual).
- cpp-reviewer (agent a753b2b2ed2f66750): **WARNING — implementation sound,
  one HIGH test-geometry defect** + one MEDIUM latency gap, no CRITICAL.
  - HIGH: the transformer RankLocalCheckpoint test hardcoded heads=4/hd=2
    (qkv=8), so on 8 GPUs local_heads = 4/8 = 0 -> zero-sized QKV ->
    cuBLAS INVALID_VALUE (RED on the 8-GPU test hardware; the suite's ONLY
    hardware-dependent failure). Fixed: heads/inter now scale with
    device_count (heads = 2*tp, inter = 4*tp) — valid on 2/4/8 GPUs, and the
    suite re-verified GREEN on **8 GPUs** (EV-044).
  - MEDIUM (pre-existing latent, surfaced here): the attention ctor only
    validated qkv % tp (via the inner layer ctors) which does NOT imply
    num_heads % tp == 0 — local_heads truncates to 0 silently. Added
    `num_heads % tp_degree() == 0` to the ctor -> wrong topology now throws
    cleanly instead of a cuBLAS INVALID_VALUE at forward.
  - LOW (folded in): the NSCK header now carries a TP-degree field (v2) so a
    wrong-topology load fails up front rather than by size luck; docs state
    the rank-identity caveat (a same-topology file from another rank matches
    every size check — load this rank's own file; per-rank id is the
    follow-up TASK-063). Step-counter caller-managed already documented (v2.32).
- Re-verified after fixes: CheckpointTest 8/8, CheckpointMultiGpuTest 2/2 on
  2/4/8 GPUs, attention/TP/block + neural single-GPU suites green (59/59 in
  the target filter), multi-GPU cross-suite green incl. deep-block trajectory,
  full-suite baseline 1469/0 re-run with the fixes included.
