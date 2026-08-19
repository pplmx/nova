# RIL — Round 28 (2026-08-18) — milestone v2.28 "Normalized Transformer Blocks (LayerNorm + Residual) + Deep Multi-Layer Training"

## Round focus

**Open milestone v2.28** on the capability gap Round 27's LEARN named next ("a
deeper/multi-layer transformer"). The v2.21-27 stack trains exactly ONE kind of
model — an unnormalized single block (attention → MLP, logits = MLP(attn(X))).
The neural `layer_norm` module is a forward-only orphan (no backward pass, so
untrainable; its kernels were never validated against a reference; nothing in
the parallel stack uses it), no residual connection exists anywhere in the
stack, and therefore no model can stack more than one block. A real
(pre-norm) transformer block — LN→MHA→(+x)→LN→MLP→(+x) — is the dependency for
every deeper-model milestone, so it IS this milestone (TASK-039, DEC-014).

## Observation / Model

- The only "transformer" the stack can train (v2.26 test) is single-block,
  unnormalized: `attn.forward → mlp.forward → CE-backward → mlp.backward →
  attn.backward → step(7 tensors)`. No LayerNorm, no residual, no stacking.
- `layer_norm.h/.cu` (169 LOC): forward-only `layer_norm()` /
  `layer_norm_inference()` + a `LayerNormResult` host/device struct whose 91-LOC
  test file never validates the kernel against a reference (it checks the
  struct fields and host-side math). No backward, no trainable affine
  integration — the module is dead weight in the training stack.
- Normalization on the TP path is collective-free by construction: block
  inputs/outputs are replicated `[m x hidden]` on every rank (the v2.21-26
  convention; only intermediate shards are rank-local), so a per-row
  LayerNorm over the full hidden dim computes identically on every rank — no
  AllReduce. γ/β are replicated `[hidden]` vectors. Residual adds are pure
  element-wise on replicated buffers. The only comm in the block stays the
  output projection's AllReduce (v2.26).
- Backward is the standard analytic layer-norm chain (dγ = Σᵢ dyᵢ·x̂ᵢ,
  dβ = Σᵢ dyᵢ, dL/dx = inv_std·(H·dx̂ − Σ dx̂ − x̂·Σ dx̂·x̂)/H per row), exact
  element-wise like the v2.27 optimizer parity. Pre-LN + residual (identity
  shortcut) is what lets a deeper stack train on the tiny synthetic task.

## Decisions

- **DEC-014**: milestone direction — normalization + residual + an N-block
  deep trainable transformer on the parallel stack. Rejected: LAMB
  kernelization (performance, orthogonal — deferred as AdamW was in DEC-012);
  a full GPT-scale language model (needs the normalized block first —
  normalization/residual IS the dependency); reusing the forward-only
  LayerNorm as-is (untrainable — superseded by a backward+affine-capable
  layer); fusing LayerNorm into attention/MLP (couples modules; LN stays its
  own layer with its own optimizer steps, one AdamW per tensor).

## P1 RED (TASK-040, EV-022)

- Added the contract surface with provisional throw-stubs: `LayerNorm` class
  (trainable gamma/beta — layer_norm.h/.cu), `TransformerBlock`
  (tensor_parallel_block.h/.cpp), `TransformerTrainer` (training_transformer.h/
  .cpp), 3 new test files (LN trainable, block, deep-block multi-GPU).
- RED run (EV-022): 8 single-GPU contracts — 1 PASS (the host-fp64 LN backward
  reference self-checked against central finite differences — note the probe
  had to perturb the FULL batch forward, and per-column rebuild, to isolate a
  row) + 7 FAILED (throw-stubs). Commit `test(neural) P1 RED ...`.

## P2 implementation (TASK-041, commit cde8ad7)

- layer_norm.cu: `ln_forward_trainable_kernel` (block-per-row stats) + analytic
  backward (`ln_backward_stats_kernel` atomically accumulates dgamma/dbeta,
  `ln_backward_input_kernel` dL/dx), `LayerNorm::step` = one AdamW per affine
  tensor. Collective-free (replicated activations).
- tensor_parallel_block.cpp: real composition, residuals via `elementwise_add`;
  backward wires residual backprop (dL/dx = residual direct + LN1-path).
- training_transformer.cpp: forward (N blocks + final LN) -> device CE-logits
  backward -> per-block backward chain -> 11N+2 per-tensor AdamW.
- Test authoring caught two bogus expectations (affine gamma breaks unit
  variance and zero-mean output) and a test bug (host grad pointers into the
  device AdamW step) — the device path was correct.
- GREEN: `LayerNormTrainableTest` 5/5, `TransformerBlockTest` 3/3 (incl. block
  fwd/bwd vs an independent host-fp64 whole-block reference and N=3 deep
  convergence), single-GPU neural regression subset 28/28.

## Milestone definition (opened)

Phases (TASK-040/041/042):

- **P1 (RED/parity, TASK-040)** — pin device LayerNorm forward vs host-fp64
  reference and analytic backward (d_x, d_gamma, d_beta, finite-difference /
  pass-consistency checked); pin the pre-LN residual TransformerBlock
  forward/backward against a host-fp64 block reference; pin the deep N-block
  mini-transformer training contract (loss descent + accuracy rise) and the
  K-step multi-GPU trajectory parity contract on 2 & 4 GPUs.
- **P2 (implementation, TASK-041)** — device LayerNorm backward kernels +
  affine γ/β wired into the layer_norm API (making the orphan trainable); a
  `TransformerBlock` composing pre-LN attention + pre-LN MLP with residual
  adds and one AdamW per weight tensor incl. each LN γ/β; an N-block stack +
  training loop on the MicroTrainer conventions.
- **P3 (verify, TASK-042)** — LN fwd/bwd parity GREEN; single-GPU N-block
  convergence; multi-GPU K-step N-block shard==single-GPU full-weight parity
  GREEN on 2 & 4 GPUs; cross-suite + single-GPU regression + full-suite
  baseline; cpp-reviewer pass; RIL close.

## P3 verify (TASK-042, EV-023)

- LayerNormTrainableTest 5/5 + TransformerBlockTest 3/3 GREEN (single-GPU).
- Neural multi-GPU cross-suite **19/19 on 2 GPUs and 19/19 on 4 GPUs**
  (DeepBlockMultiGpuPinnedTest shard==single-GPU parity + v2.26 attention
  multi-GPU + v2.25 MicroTrainer multi-GPU + TP multi-GPU regressions) —
  `CUDA_VISIBLE_DEVICES=2,3` / `2,3,4,5`, `NCCL_TESTS_AVAILABLE=1`.
- Single-GPU neural regression: 97 pass / 1 fail on a
  `LossFunctionsTest.CrossEntropyLogitsBackwardDevice` ordering flake that
  **reproduces identically without any v2.28 code** (passes in isolation and
  alongside the v2.28 suites) — pre-existing, unrelated (likely the same
  suite-interleaving fragility family as issue-v19-shared-nccl-context-reset).
- **Full-suite baseline not reproducible in this environment**: all 8 GPUs are
  ~76GB used (~5.8GB free each) as of 2026-08-18, so ~20 memory-heavy tests
  (BlockManager/CUDA-graph, ChunkedPrefill 16K+ prompts, Beam, SegmentedSort)
  fail with `out of memory` (or `invalid device ordinal` under the
  CUDA_VISIBLE_DEVICES remap) — environment memory, **none touch neural/v2.28
  code**. Recorded in the project GPU-selection memory.
- Host-note update: `CUDA_VISIBLE_DEVICES=2,3+` still holds; re-check
  nvidia-smi before multi-GPU runs (all devices now loaded).

## Graph additions (Round 28)

- `DEC-014` (decision, milestone direction); `TASK-039/040/041/042` (umbrella,
  P1/P2/P3 — priority 10.39 / 8.49 / 6.93 / 8.49); `issue-v28-layer-norm-
  untrainable` (the forward-only/untrainable LayerNorm + no-residual/no-stack
  gap); edges DEC-014→TASK-* governs, TASK-039→040→041→042 depends_on,
  TASK-039→issue addresses, issue→comp-neural-parallel located_in.

## cpp-reviewer pass (CHG-018/019)

- **C1 (CRITICAL)**: `TransformerBlock::forward/backward` sized `LayerNorm` rows,
  scratch and residual adds on `batch` instead of `batch*seq` — a latent device
  OOB whenever seq > 1 (all milestone tests used seq=1). All my understood
  checks were correct; the row-count confusion was real. Fixed (799812e) with
  `m = batch*seq` in the block + three new seq>1 regression tests
  (`BlockForward/BackwardMatchesHostReferenceSeq2`,
  `TrainerSupportsSeqGreaterThanOne`). 11/11 → 12/12 after the fix, 2-GPU
  parity re-verified GREEN.
- **L2**: `layer_norm_inference` launched a kernel that dereferences mean/var
  unconditionally with nullptr passed — a guaranteed device fault if ever
  called (dead in-tree). Routed it through `layer_norm()` with internal stats,
  deleted the dead kernel, pinned with `LegacyInferenceMatchesHostReference`
  (81edd9d). Authoring that test exposed that the legacy free functions do NOT
  upload host gamma/beta (callers pass device buffers).
- **M1/M2/M3/M4** (free-fn per-call buffer, one-thread-per-row LN kernels,
  host-side SDPA in the attention, blocking copies in train_step): performance
  / optimization, deferred — noted in EV-024/Learn.
- **L1** (legacy `LayerNormResult` rule-of-five): pre-existing, untouched.

## Verify (final, EV-023/EV-024)

- Single-GPU: `LayerNormTrainableTest` 6/6, `TransformerBlockTest` 6/6 (block
  fwd/bwd parity vs independent host-fp64 whole-block reference; N=3 deep
  convergence loss-descent + accuracy; seq>1 forward/backward/trainer).
- Multi-GPU: neural cross-suite **19/19 on 2 GPUs and 19/19 on 4 GPUs** incl.
  the deep-transformer shard==single-GPU parity (K=12, 3 blocks, all 33 block
  weight tensors + final LN assembled == fp64 single-GPU reference).
- Single-GPU neural regression: green except one **pre-existing unrelated
  ordering flake** (`LossFunctionsTest.CrossEntropyLogitsBackwardDevice` —
  reproduced identically without any v2.28 code; passes in isolation and with
  the v2.28 suites; same suite-interleaving fragility family as
  issue-v19-shared-nccl-context-reset).
- Full-suite baseline: not reproducible in this environment — all 8 GPUs are
  ~76GB used (~5.8GB free), so ~20 memory-heavy non-neural suites
  (BlockManager/CUDA-graph, ChunkedPrefill, Beam, SegmentedSort) fail with
  `out of memory` / `invalid device ordinal` under the CUDA_VISIBLE_DEVICES
  remap. Environment-memory limitation, none touch neural/v2.28 code. Recorded
  in the project GPU-selection memory.

## Learn

- The seq>1 device-OOB (C1) was invisible because every training test used
  seq=1 — the row-count convention (`batch*seq` vs `batch`) needs an explicit
  seq>1 regression whenever a layer composes sub-layers that each interpret
  rows. The reviewer's pass proved its value: it found a real bug the suite
  could not.
- The independent host-fp64 whole-block reference caught nothing wrong here —
  the composition matched — but it remains the right contract (it validated
  the residual backprop wiring; cheap insurance).
- Performance debt to address next (EV-024/M1-M4): one-thread-per-row LN
  kernels (fine at hidden<=64, a bottleneck at scale), per-call stats buffer in
  the free `layer_norm_backward`, host-side SDPA inside the attention that now
  runs N× per train_step, and the blocking copies in train_step/evaluate. The
  next milestone could be a "device-native attention + LN training kernels"
  performance round on the v2.28 stack.

## Milestone status

- **CLOSED** — TASK-039/040/041/042 resolved; issue-v28-layer-norm-untrainable
  resolved (CHG-017); DEC-014; EV-022/023/024; CHG-017 (cde8ad7), CHG-018
  (799812e C1), CHG-019 (81edd9d L2). Round bumped to 29.
- The parallel stack now trains a deep, normalized, residual transformer
  (N-block, LayerNorm, one AdamW per tensor) with shard==single-GPU parity on
  2 & 4 GPUs — the "deeper/multi-layer transformer" capability Round 27 named
  next is realized. Future work (Round 29 candidates): device-native SDPA +
  fused LN/attention training kernels (perf), LAMB kernelization, or deeper
  scale-up (more blocks, longer seq, real data).
