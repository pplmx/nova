# v2.13 Transformer Optimization — Requirements

**Milestone:** v2.13 Transformer Optimization
**Created:** 2026-05-05
**Last Updated:** 2026-05-05

---

## Speculative Decoding

### Core Infrastructure

- [ ] **SPEC-01**: User can configure speculative decoding via `SpeculativeDecodingConfig` with draft depth (1-8 tokens), acceptance threshold, and draft model selection
- [ ] **SPEC-02**: Draft model can generate K candidate tokens in a single forward pass using existing FlashAttention infrastructure
- [ ] **SPEC-03**: Target model verifies ALL draft tokens in parallel using tree attention kernels with proper masking
- [ ] **SPEC-04**: User can perform rejection sampling with correct probability comparison (`fminf(1.0f, target_prob / draft_prob)`)
- [ ] **SPEC-05**: User can isolate KV cache per speculation via snapshot/rollback mechanism
- [ ] **SPEC-06**: User can track log probabilities for draft tokens via LogProbTracker for KL divergence verification

### Advanced Features

- [ ] **SPEC-07**: User can enable EAGLE3/SnapKV tree-based decoding for higher acceptance rates
- [ ] **SPEC-08**: User can overlap draft generation with verification using async CUDA streams
- [ ] **SPEC-09**: User can configure guided decoding with xGrammar integration compatible with speculative decoding

---

## Beam Search

### Core Infrastructure

- [ ] **BEAM-01**: User can create `BeamSearchManager` managing N beam hypotheses (1-8 beams)
- [ ] **BEAM-02**: User can run beam search with cumulative log probability scoring and length normalization
- [ ] **BEAM-03**: User can allocate KV cache blocks per beam hypothesis with reference-counted prefix sharing
- [ ] **BEAM-04**: User can perform batch KV operations across multiple beam hypotheses efficiently
- [ ] **BEAM-05**: User can configure TopK/TopP nucleus sampling with beam search integration

### Advanced Features

- [ ] **BEAM-06**: User can use score rebase to prevent underflow at long sequences (>2000 tokens)
- [ ] **BEAM-07**: User can combine beam search with speculative decoding in the same inference loop
- [ ] **BEAM-08**: User can export beam search traces with token scores for debugging/analysis

---

## KV Cache Improvements

### Core Infrastructure

- [ ] **KV-01**: User can configure `StreamingCacheManager` with async prefetch and eviction policies
- [ ] **KV-02**: User can enable cross-sequence prefix caching with block-level hash lookup
- [ ] **KV-03**: User can separate attention sink storage from LRU cache for importance-based eviction
- [ ] **KV-04**: User can monitor PagedAttention fragmentation ratio and trigger compaction below 30% threshold

### Advanced Features

- [ ] **KV-05**: User can use dynamic block sizing (16/32/64 tokens) based on sequence access patterns
- [ ] **KV-06**: User can implement chunked prefill for long prompts (>16K tokens) when memory-constrained
- [ ] **KV-07**: User can configure L2 cache persistence hints for iterative algorithms
- [ ] **KV-08**: User can enable persistent KV cache variants for CUDA Graph capture

---

## Out of Scope

| Feature | Reason for Exclusion |
|---------|---------------------|
| Multi-model speculative decoding | Model coordination complexity; hardware heterogeneity edge case |
| NVFP4 KV quantization | Quality vs memory tradeoff unvalidated; requires profiling |
| Multi-tier KV cache (GPU/CPU/NIC) | Architecture documented but implementation complexity too high for this milestone |
| Disk-based persistent KV cache | Latency and consistency issues; streaming context is preferred |
| Beam width > 8 | Memory O(N*seq_len) with diminishing returns above 4-8 |

---

## Future Requirements (Deferred)

- Multi-model speculative decoding
- KV cache compression (NVFP4)
- Multi-tier KV cache (GPU/CPU/NIC)
- Disaggregated serving (KV transfer between prefill/decode nodes)

---

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| SPEC-01 | Phase 105 | Pending |
| SPEC-02 | Phase 105 | Pending |
| SPEC-03 | Phase 105 | Pending |
| SPEC-04 | Phase 105 | Pending |
| SPEC-05 | Phase 105 | Pending |
| SPEC-06 | Phase 105 | Pending |
| SPEC-07 | Phase 105 | Pending |
| SPEC-08 | Phase 105 | Pending |
| SPEC-09 | Phase 105 | Pending |
| BEAM-01 | Phase 104 | Pending |
| BEAM-02 | Phase 104 | Pending |
| BEAM-03 | Phase 104 | Pending |
| BEAM-04 | Phase 104 | Pending |
| BEAM-05 | Phase 104 | Pending |
| BEAM-06 | Phase 104 | Pending |
| BEAM-07 | Phase 106 | Pending |
| BEAM-08 | Phase 104 | Pending |
| KV-01 | Phase 103 | Pending |
| KV-02 | Phase 103 | Pending |
| KV-03 | Phase 103 | Pending |
| KV-04 | Phase 103 | Pending |
| KV-05 | Phase 106 | Pending |
| KV-06 | Phase 106 | Pending |
| KV-07 | Phase 103 | Pending |
| KV-08 | Phase 103 | Pending |

---

*Requirements defined: 2026-05-05*
