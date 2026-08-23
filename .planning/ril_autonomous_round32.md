# RIL — Round 31 (2026-08-24) — milestone v2.32 "Model Checkpointing (Weights + Optimizer State)"

## Round focus

**Open milestone v2.32** — the core-feature round. The host-round-trip family is
complete (v2.27 optimizer, v2.29 SDPA/LN, v2.30 loss, v2.31 LAMB — every
`train_step`/`evaluate` hot path is device-native), and the train stack
(v2.25-28) is real: `MicroTrainer` learns, `TransformerTrainer` trains deep
pre-LN blocks, multi-GPU sharded runs reproduce the host fp64 trajectory. But
nothing **persists**: a trained model dies with the process, an interrupted
run cannot be resumed, and there is no export path. The memory-opt
`CheckpointCompressor` (v2.x) compresses device buffers in-flight — it is not
model serialization. The trainers expose only weight *upload* (`set_weight` /
`set_block_weight` / `set_final_ln_weight`, full-weight semantics) and weight
*readback* (`copy_weights` / `copy_block_weights` / `copy_final_ln`, rank-shard
semantics), and the AdamW/LAMB moment buffers are private. So the save path is
a real gap, and the restore path needs optimizer state to be exact.

## Observation / Model

- **M1 (binding)**: no model persistence. `MicroTrainer` /
  `TransformerTrainer` (v2.25/28) have no save/load; long runs can't be
  checkpointed or resumed. Core-feature gap in the training stack.
- **M2 (restore must be exact)**: an AdamW update reads the existing m/v
  (`m_new = beta1*m + (1-beta1)*grad`), so resuming requires the moment
  buffers restored too — weights alone diverge from the uninterrupted
  trajectory at the first resumed step. The step *counter* is caller-side
  (`train_step(..., step_no)`), so the checkpoint needs weights + moments,
  not a step count.
- **M3 (tp semantics)**: with TP>1, `copy_*` exposes the rank shards while
  the `set_*` uploaders take *full* weights and slice this rank's shard — so
  `copy_*`→`set_*` is asymmetric away from tp=1. At tp=1 (shard == full) the
  existing setters roundtrip byte-exactly. Scope v2.32 to the canonical tp=1
  full-state path with per-tensor count validation (a tp-mismatched or
  corrupted file is rejected, not mis-restored); tp>1 rank-shard restore
  (needs shard setters on the attention/MLP pieces) is the named follow-up.

## Decisions

- **DEC-018**: milestone direction — model checkpointing for the trainers:
  `save_state(std::ostream&)` / `load_state(std::istream&)` on `MicroTrainer`
  and `TransformerTrainer` behind a deterministic binary format (NSCK v1:
  magic, version, kind, dims, tensors as count+float[], moments as
  count+m[]+v[]), plus `AdamWOptimizer::{momentum_capacity, copy_moments_to,
  copy_moments_from}` for the optimizer-state path. Verified by (a) byte-exact
  save→load roundtrip of weights and moments, (b) resumed-vs-uninterrupted
  training equality after restore, (c) fresh-state roundtrip (zero moments),
  (d) geometry/corruption/mismatch rejection, at tp=1; tp>1 rank-shard
  restore is a follow-up task. Rejected: cross-rank full-weight gather via
  NCCL (couples the save path to collectives, wider API); CUDA-graph capture
  (refuted, EV-034); fp16/bf16 mixed precision (multi-milestone scope);
  TASK-018 (1.50 < 3.0, unrelated maintainability).

## Milestone definition (opened)

| Phase | Name | Status |
|-------|------|--------|
| 1 | RED/parity: AdamW moment export/import API + MicroTrainer/TransformerTrainer save_state/load_state (throw-stubs); CheckpointTest contracts (roundtrip byte-exact, fresh-state, resume-vs-uninterrupted, geometry/corrupt rejection); RED against stubs | In progress (Round 32) |
| 2 | Implement: AdamW moment API; CheckpointWriter/Reader (header-only, src/cuda/neural/checkpoint_io.h); MicroTrainer + TransformerTrainer save_state/load_state routing through copy_*/set_* + moment copies + validation | Pending |
| 3 | Verify: CheckpointTest GREEN across the suite; neural single-GPU regression GREEN; cpp-reviewer pass; RIL close; tp>1 follow-up task opened | Pending |

**Graph additions (Round 32, opening):**
- `DEC-018` (decision, milestone direction); `comp-neural-training` (training-
  drivers component, part_of comp-neural-parallel); `ISS-001` (no-persistence
  issue); `TASK-055` (umbrella) / `TASK-056` (P1) / `TASK-057` (P2) /
  `TASK-058` (P3).

## P1 RED (TASK-056)

- Added the contract surface with throw-stubs: `AdamWOptimizer::momentum_
  capacity` / `copy_moments_to` / `copy_moments_from` (optimizers.h/.cpp) and
  `MicroTrainer` / `TransformerTrainer` `save_state(std::ostream&)` /
  `load_state(std::istream&)` (training.h/.cpp, training_transformer.h/.cpp).
- Added `CheckpointTest` (tests/neural/checkpoint_test.cpp, registered in
  tests/CMakeLists.txt) with 6 contracts on the tp=1 path:
  `AdamWMomentExportImportRoundtrip` (optimizer-level oracle: exported m/v
  into a fresh optimizer reproduce the next update byte-exactly);
  `MicroTrainerSaveLoadRoundTripIsByteExact` (weights after warm training);
  `MicroTrainerResumeAfterLoadMatchesUninterrupted` (K-step resume ==
  uninterrupted holdout); `MicroTrainerFreshStateRoundTrip` (zero-moment path);
  `TransformerTrainerSaveLoadRoundTripAndResume` (11N+2 tensors, byte-exact at
  the save/load boundary + resumed trajectory); `RejectsGeometryMismatchAnd-
  Corruption` (dims/kind mismatch, bad magic, truncated stream -> throw).
- RED run (EV-038): 6/6 fail for the right reason — every test hits the stub
  (`... not implemented (v2.32 P1 stub)`). Baseline green before the change:
  OptimizersTest + MicroTrainer + TransformerBlock 26/26. (P1 test bug caught
  during the RED run: the optimizer-level test originally passed host
  `std::vector` pointers as the device params/grads to `step()` — an async
  launch left a sticky illegal-memory-access that only surfaced on the next
  test's first memcpy; fixed to device `Buffer`s like optimizers_test does.)

## P2 implementation (TASK-057, EV-039)

- `src/cuda/neural/checkpoint_io.h` (internal, header-only): the NSCK-v1
  `Writer`/`Reader` (u32 magic/version/kind, tagged tensor records
  count+float[], tagged AdamW moment pairs count+m[]+v[], every read
  validated — bad magic/version/kind, count mismatch, truncated stream all
  throw std::runtime_error).
- `AdamWOptimizer` moment API implemented in optimizers.cpp:
  `momentum_capacity()` (buffer size, 0 for never-stepped), `copy_moments_to`
  (D2H, rejects n > capacity), `copy_moments_from` (allocates + zero-fills
  when fresh/smaller, then H2D; n == 0 resets to zero — the "the saved
  optimizer was fresh" case).
- `MicroTrainer::save_state`/`load_state` (training.cpp): kind 1; dims +
  gate/up/down tensors + 3 moment pairs, validated on load; restores via
  `set_weight` + `read_adamw_moments`. `TransformerTrainer::save_state` /
  `load_state` (training_transformer.cpp): kind 2; dims + 11N+2 tensors +
  11N+2 moment pairs, per-block tensor counts from `block_tensor_sizes`
  (mirrors block_grad_size), restores via `set_block_weight` /
  `set_final_ln_weight`.
- Both save_state() throw when tp_degree() > 1 (rank shards can't roundtrip
  through the full-weight setters — the named follow-up), so a checkpoint is
  never written that cannot load.
- GREEN (EV-039): CheckpointTest 6/6 (incl. byte-exact roundtrip + resumed-vs
  -uninterrupted on both trainers); neural single-GPU regression 113/113
  (baseline + the new suite); multi-GPU cross-suite **5/5 on 2 GPUs**
  (MicroTrainerMultiGpu + DeepBlockMultiGpu trajectory + attention/block
  multigpu), trajectory unchanged. No new device kernels — sanitizer not
  applicable.

## P3 verify + cpp-reviewer (TASK-058, EV-040)

- cpp-reviewer (agent aba7274412a746e17): **APPROVE** — no CRITICAL/HIGH; the
  read path was called out as the strongest part (stream-derived counts never
  trusted for allocation). 3 MEDIUM: (1) load_state wasn't transactional
  (weights applied before a later moment record validated -> partial restore on
  mid-write truncation); (2) the format doesn't serialize the AdamW step
  counter, so "byte-exact resume" silently depends on the caller continuing
  step_no at the interrupted count; (3) copy_moments_to/from only rejected
  n > capacity, not n != capacity (a stale-tail hazard for direct API users).
  LOWs: check-before-write in write_adamw_moments, kMaxCount headroom, the
  load-side tp>1 guard asymmetry, endianness doc nuance, wasted zero-fill.
- Fixes (commit 0b5ee09, CHG-025): two-phase transactional load_state on both
  trainers (read+validate every record, then apply; a rejected load leaves the
  trainer untouched); copy_moments_to requires n == capacity and copy_moments_
  from rejects n < existing capacity; step_no caller-managed contract
  documented on save_state/load_state; symmetric tp>1 load guard;
  kBlockTensorCount shared constant; widened kMaxCount; endianness doc
  clarified. Plus 2 new tests pinning the fixed contracts:
  `RejectedLoadLeavesTrainerUntouched` (truncation at 4 points leaves block-0
  + final-LN byte-identical) and `AdamWMomentCapacityMismatchRejected`.
- Re-verified after the fixes: CheckpointTest **8/8**; neural single-GPU
  regression **115/115**; multi-GPU cross-suite **5/5 on 2 GPUs** (deep-block
  trajectory unchanged).
- Full-suite baseline (EV-040 close pass): **1469 PASSED / 0 FAILED** in one
  in-process run (1545 ran, 153 skipped — MPI/multi-GPU/launcher-gated
  suites, matching the previous baseline's skip set; the prior -j8 15-OOM
  contention failures don't occur single-process). No regression.
- RIL close: TASK-055/056/057/058 resolved (CHG-024 dd93df5, CHG-025 0b5ee09)
  on EV-038/039/040; ISS-001 stays active (tp=1 persistence shipped, tp>1
  rank-shard restore opened as **TASK-059**, 6.0).
