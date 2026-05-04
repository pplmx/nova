# Project Research Summary

**Project:** Transformer Optimization Features (v2.13)
**Domain:** CUDA/HPC Transformer Inference Optimization
**Researched:** 2026-05-05
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project adds three major inference optimization features to an existing CUDA transformer library: Speculative Decoding (2-4x latency reduction), Beam Search (quality improvement), and KV Cache Improvements (30-70% memory reduction for shared prefixes). The architecture builds on a proven five-layer design (API → Algo → Memory → Device → CUDA) that already includes FlashAttention 2.6 and PagedAttention.

Research indicates the optimal implementation sequence starts with KV Cache foundation, then Beam Search, then Speculative Decoding. This order minimizes architectural rework and allows each phase to benefit from infrastructure built in prior phases. Key risks include distribution mismatch in speculative decoding verification (producing incorrect outputs) and KV cache contamination when rejected tokens corrupt state. Both are addressable with proper snapshot/rollback mechanisms and isolation of speculative KV from verified KV.

The recommended stack centers on FlashInfer 0.6+ for sampling kernels and page-attached KV cache, FlashAttention 3.x for Blackwell/Hopper support, and Torch Sampler (not deprecated TRTLLMSampler) for beam search integration. Existing FlashAttention and KVCacheAllocator components are extendable, not replaceable.

## Key Findings

### Recommended Stack

The core infrastructure requires CUDA 12.x+ (12.8+ for Blackwell), cuDNN 9.x, and CUTLASS 3.x for custom GEMM kernels. For speculative decoding, FlashInfer 0.6+ provides the `chain_speculative_sampling` rejection kernel and paged KV cache support via `append_paged_kv_cache`. FlashAttention 3.x is required for NHD/HND layouts and Blackwell support—existing 2.x installations must be upgraded. The Torch Sampler (TensorRT-LLM's default, replacing deprecated TRTLLMSampler) uses FlashInfer internally for sampling.

**Core technologies:**
- **FlashInfer 0.6+**: Sorting-free sampling kernels, chain speculative sampling, paged KV cache API — primary dependency for spec decode
- **FlashAttention 3.x**: NHD/HND attention layouts, cascading states, TMA/async copy for Hopper+ — required upgrade from existing 2.6
- **TensorRT-LLM Attention (XQA kernel)**: 2.4x Llama-70B throughput improvement on H100 — for decode attention
- **xGrammar**: Guided decoding compatible with speculative decoding — replaces deprecated guidance library
- **CUTLASS 3.x**: FP8/INT4 custom kernels, FP4 tensor support — for quantization extension

### Expected Features

**Must have (table stakes):**
- **Speculative Decoding** — 2-4x latency improvement, requires FlashAttention and sequence manager integration
- **Beam Search Core (K=4)** — Required for translation, summarization, code generation; memory-optimized with shared prefix blocks
- **Cross-Sequence Prefix Caching** — 30-70% memory reduction for shared system prompts and RAG workloads

**Should have (competitive):**
- **Tree-Based Speculative Decoding** — Better acceptance rates via EAGLE3/SnapKV tree attention; unlocks higher draft depth
- **Dynamic Block Sizing** — Adaptive block sizes (16/32/64 tokens) based on workload, 5-15% memory improvement
- **Chunked Prefill** — Streaming prefill for long prompts (>16K tokens) when memory-constrained

**Defer (v2+):**
- Multi-model Speculative Decoding — Model coordination complexity, hardware heterogeneity edge case
- KV Cache Compression (NVFP4) — Requires additional profiling for quality tradeoff

### Architecture Approach

The existing five-layer architecture (API/Scheduler → Algo/FlashAttention → Memory/KVCacheAllocator → Device/Stream → CUDA) is the foundation. New features integrate as follows: Speculative Decoding adds `SpeculativeDecodingRunner` to Scheduler and modifies BlockManager for branched block tables; Beam Search adds `BeamSearchManager` with per-beam KV allocation via modified KVCacheAllocator; KV Cache Improvements add `StreamingCacheManager` and `EvictionPredictor` to KVCacheAllocator, plus persistent attention variants.

Critical architectural constraints: Speculative Decoding requires isolated KV cache per speculation (each draft allocates independent blocks, accepted tokens merge to parent); Beam Search uses reference-counted prefix sharing (only fork blocks that diverge, don't copy full KV); KV Cache improvements enhance both other features by reducing memory pressure.

**Major components:**
1. **SpeculativeDecodingRunner** — Orchestrates draft→verify loop, integrates with Scheduler::step()
2. **BeamSearchManager** — Beam state, scoring, pruning; manages std::vector<Sequence*> per hypothesis
3. **StreamingCacheManager** — Async prefetch/evict with L2 persistence hints; hooks into KVCacheAllocator

### Critical Pitfalls

1. **Distribution Mismatch** — Rejection sampling compares probabilities incorrectly (logits vs softmax). Always compute `fminf(1.0f, target_prob / draft_prob)` and verify with KL divergence against reference.

2. **KV Cache Contamination** — Rejected speculative tokens leave residual attention states that corrupt subsequent tokens. Prevent via snapshot/rollback or prefix-only attention masking during speculation.

3. **Beam Width Explosion** — Naive beam search stores `beam_width * seq_len * vocab` scores. Use compact representation with parent pointers, not full history.

4. **Score Underflow** — Cumulative log probabilities underflow to -inf at long sequences (~2000 tokens). Use length normalization: `score / pow(seq_len, alpha)` or periodic rebase.

5. **PagedAttention Fragmentation** — Variable-length requests fragment memory into small holes. Monitor fragmentation ratio, trigger compaction below 30% threshold.

6. **Attention Sink Dominance** — First-token attention sinks consume disproportionate cache. Implement importance-based eviction, separate sink storage from LRU cache.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: KV Cache Foundation
**Rationale:** Other features depend on efficient KV cache. Building streaming, eviction, and persistence first enables better memory planning for speculative and beam modes.

**Delivers:** StreamingCacheManager with async prefetch, EvictionPredictor with attention-aware policy, L2 persistence hints, persistent attention kernel variant.

**Addresses:** KV Cache Improvements (partial), PagedAttention fragmentation prevention

**Avoids:** Attention sink dominance, fragmentation explosion

**Research Flags:** Medium — L2 persistence hints well-documented; eviction prediction is heuristic/ML choice needs validation

### Phase 2: Beam Search Core
**Rationale:** Beam search has clear algorithm spec, lower implementation complexity (MEDIUM vs HIGH for spec decode), and validates the memory layer changes from Phase 1.

**Delivers:** BeamSearchManager, BeamSequence with beam_id/beam_score, TopKSampler, batch KV operations in BlockManager.

**Uses:** FlashAttention (existing), KVCacheAllocator batch allocation (Phase 1)

**Implements:** Architecture component: beam_search.h

**Research Flags:** Low — Well-documented in vLLM, TGI, HuggingFace; standard patterns

### Phase 3: Speculative Decoding
**Rationale:** Highest user value (2-4x latency) but highest complexity. Benefits from Phase 1 memory improvements and Phase 2 batch handling. Most pitfalls (distribution mismatch, KV contamination) require deep integration testing.

**Delivers:** SpeculativeDecodingRunner, DraftModel interface, VerificationKernel with chain speculative sampling, LogProbTracker.

**Uses:** FlashInfer 0.6+ (sampling kernels, page API), xGrammar (guided decoding compatibility)

**Implements:** Architecture component: speculative_decoding.h, logprob_tracker.h

**Avoids:** Distribution mismatch (proper softmax acceptance check), KV contamination (snapshot/rollback)

**Research Flags:** High — Rejection sampling implementation correctness is critical; need KL divergence test against reference

### Phase 4: Integration & CUDA Graph
**Rationale:** Consolidate all features, ensure no interference, optimize with CUDA Graphs, add persistent KV support for graph capture.

**Delivers:** Persistent cache CUDA Graph support, cross-feature integration tests, benchmark suite.

**Avoids:** CUDA Graph + dynamic batch (mark KV ops non-graph or use conditional nodes), KV cache operations breaking graph capture

**Research Flags:** Medium — TensorRT-LLM patterns available; some custom graph work needed

### Phase Ordering Rationale

- **Phase 1 first:** KV Cache is foundational; streaming/prefetch/eviction needed for memory-bounded speculative and beam modes
- **Beam before Spec Decode:** Lower complexity, validates memory layer, clear algorithm spec
- **Spec Decode last:** Highest complexity, depends on both prior phases, most verification work needed
- **Integration at end:** Ensure features work together (beam + spec decode possible), CUDA Graph optimization

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Speculative Decoding):** Kernel implementation correctness—KL divergence verification methodology, tree attention kernel vs single-draft tradeoffs
- **Phase 4 (Integration):** CUDA Graph conditional nodes for dynamic shapes, persistent KV + batch size variation

Phases with standard patterns (skip research-phase):
- **Phase 2 (Beam Search):** Well-documented in vLLM/TGI; beam_width=4 greedy selection is established pattern

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified against NVIDIA official docs, vLLM, TensorRT-LLM, FlashInfer; version compatibility well-documented |
| Features | MEDIUM-HIGH | Based on established algorithms; MVP scope clear; some differentiator decisions need validation |
| Architecture | MEDIUM-HIGH | Extension patterns clear for existing 5-layer design; build order validated |
| Pitfalls | HIGH | Pitfalls documented with prevention strategies from vLLM issues, academic papers, community |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Tree attention kernel details:** EAGLE3 vs single-draft vs Medusa decision needs workload profiling before Phase 3
- **NVFP4 KV quantization:** Quality vs memory tradeoff unvalidated; defer unless memory pressure at scale
- **Multi-tier KV cache (GPU/CPU/NIC):** Architecture documented but implementation complexity high; Phase 1+ scope
- **Draft model training/loading:** Interface only (neural/draft_model.h); actual model compatibility needs testing

## Sources

### Primary (HIGH confidence)
- NVIDIA TensorRT-LLM Documentation (nvidia.github.io/TensorRT-LLM/) — Speculative decoding, sampling, CUDA integration
- FlashInfer 0.6.9 Documentation (docs.flashinfer.ai/) — Attention kernels, page API, verification kernels
- vLLM Documentation (docs.vllm.ai/) — PagedAttention design, speculative decoding implementation

### Secondary (MEDIUM confidence)
- vLLM GitHub (79k stars) — State-of-the-art reference for implementation patterns
- LightLLM GitHub (4k stars) — Token-level KV cache management reference
- Flash Attention Paper (arXiv:2205.14135) — Core attention algorithm

### Tertiary (LOW confidence)
- EAGLE/SnapKV papers — Tree attention for spec decode; needs implementation validation
- Community patterns (r/MachineLearning, GitHub issues) — Edge cases, unverified

---
*Research completed: 2026-05-05*
*Ready for roadmap: yes*
