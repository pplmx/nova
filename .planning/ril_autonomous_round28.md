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

## Graph additions (Round 28)

- `DEC-014` (decision, milestone direction); `TASK-039/040/041/042` (umbrella,
  P1/P2/P3 — priority 10.39 / 8.49 / 6.93 / 8.49); `issue-v28-layer-norm-
  untrainable` (the forward-only/untrainable LayerNorm + no-residual/no-stack
  gap); edges DEC-014→TASK-* governs, TASK-039→040→041→042 depends_on,
  TASK-039→issue addresses, issue→comp-neural-parallel located_in.

## Milestone status

- Opened (graph + docs). Pending: P1 RED tests → P2 → P3 → close.
