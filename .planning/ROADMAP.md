# Roadmap

## Completed Milestones

| Version | Date | Goal | Phases | Status |
|---------|------|------|--------|--------|
| v1.0 | 2026-04-24 | Production Release | 6 | Shipped |
| v1.1 | 2026-04-24 | Multi-GPU Support | 4 | Shipped |
| v1.2 | 2026-04-24 | Toolchain Upgrade (C++23, CUDA 20) | 2 | Shipped |
| v1.3 | 2026-04-24 | NCCL Integration | 5 | Shipped |
| v1.4 | 2026-04-24 | Multi-Node Support | 3 | Shipped |
| v1.5 | 2026-04-26 | Fault Tolerance | 4 | Shipped |
| v1.6 | 2026-04-26 | Performance & Training | 4 | Shipped |
| v1.7 | 2026-04-26 | Benchmarking & Testing | 4 | Shipped |
| v1.8 | 2026-04-26 | Developer Experience | 4 | Shipped |
| v1.9 | 2026-04-26 | Documentation | 4 | Shipped |
| v2.0 | 2026-04-26 | Testing & Quality | 4 | Shipped |
| v2.1 | 2026-04-26 | New Algorithms | 4 | Shipped |
| v2.2 | 2026-04-27 | Comprehensive Enhancement | 5 | Shipped |
| v2.3 | 2026-04-28 | Extended Algorithms | 4 | Shipped |
| v2.4 | 2026-04-28 | Production Hardening | 5 | Shipped |
| v2.5 | 2026-04-28 | Error Handling & Recovery | 5 | Shipped |
| v2.6 | 2026-04-29 | Transformer & Inference Optimization | 6 | Shipped |
| v2.7 | 2026-04-30 | Comprehensive Testing & Validation | 4 | Shipped |
| v2.8 | 2026-05-01 | Numerical Computing & Performance | 4 | Shipped |
| v2.9 | 2026-05-01 | Architecture Refactor | 4 | Shipped |
| v2.10 | 2026-05-01 | Sparse Solver Acceleration | 5 | Shipped |
| v2.11 | 2026-05-02 | Performance Tooling | 5 | Shipped |
| v2.12 | 2026-05-03 | Advanced Quantization | 5 | ✅ Shipped |

Archived: [v2.12-ROADMAP.md](milestones/v2.12-ROADMAP.md) | [v2.12-REQUIREMENTS.md](milestones/v2.12-REQUIREMENTS.md) | [v2.12-MILESTONE-AUDIT.md](v2.12-MILESTONE-AUDIT.md)

---

## v2.13 Transformer Optimization

**Goal:** GPU-accelerated inference optimizations for transformer models — speculative decoding, beam search, and KV cache improvements

**Granularity:** Standard (4 phases)

**Coverage:** 25/25 requirements mapped

---

## Phases

- [ ] **Phase 103: KV Cache Foundation** — Streaming cache manager, prefix caching, attention sinks, fragmentation monitoring, L2 persistence, persistent attention
- [ ] **Phase 104: Beam Search Core** — Beam search manager with scoring, reference-counted KV sharing, batch operations, sampling integration, score rebasing, trace export
- [ ] **Phase 105: Speculative Decoding** — Draft model speculation, tree verification, rejection sampling, KV isolation, log probability tracking, EAGLE3/SnapKV, async overlap, xGrammar
- [ ] **Phase 106: Integration & CUDA Graph** — Dynamic block sizing, chunked prefill, beam + speculative combination, persistent KV with CUDA Graph

---

## Phase Details

### Phase 103: KV Cache Foundation

**Goal:** Users can manage KV cache with streaming, prefix sharing, and memory efficiency features

**Depends on:** Phase 102 (v2.12 last phase)

**Requirements:** KV-01, KV-02, KV-03, KV-04, KV-07, KV-08

**Success Criteria** (what must be TRUE):

1. User can configure async prefetch and eviction policies via StreamingCacheManager
2. User can share KV blocks across sequences with identical prefixes (hash-based lookup)
3. User can separate attention sink storage from LRU eviction to prevent sink thrashing
4. User can monitor PagedAttention fragmentation ratio and receive compaction alerts below 30%
5. User can configure L2 cache persistence hints for iterative inference workloads
6. User can capture CUDA Graphs with persistent KV cache without memory leaks

**Plans**: TBD

---

### Phase 104: Beam Search Core

**Goal:** Users can perform GPU-accelerated beam search with memory-efficient KV management

**Depends on:** Phase 103

**Requirements:** BEAM-01, BEAM-02, BEAM-03, BEAM-04, BEAM-05, BEAM-06, BEAM-08

**Success Criteria** (what must be TRUE):

1. User can create BeamSearchManager with configurable beam width (1-8 beams)
2. User can run beam search and observe length-normalized scoring without underflow at 2000+ tokens
3. User can share KV blocks across beam hypotheses using reference counting (fork-only-diverge pattern)
4. User can batch KV operations across multiple beams efficiently
5. User can combine TopK and TopP sampling with beam search
6. User can export beam search traces with per-token scores for analysis

**Plans**: TBD

---

### Phase 105: Speculative Decoding

**Goal:** Users can accelerate inference 2-4x using speculative decoding with proper verification

**Depends on:** Phase 104

**Requirements:** SPEC-01, SPEC-02, SPEC-03, SPEC-04, SPEC-05, SPEC-06, SPEC-07, SPEC-08, SPEC-09

**Success Criteria** (what must be TRUE):

1. User can configure speculative decoding via SpeculativeDecodingConfig (draft depth, acceptance threshold, model selection)
2. User can generate K draft tokens and verify all in parallel using tree attention with correct rejection sampling
3. User can isolate speculative KV from verified KV via snapshot/rollback mechanism
4. User can track log probabilities for KL divergence verification via LogProbTracker
5. User can enable EAGLE3/SnapKV tree-based decoding for improved acceptance rates
6. User can overlap draft generation with verification using async CUDA streams
7. User can configure xGrammar-guided decoding compatible with speculative decoding

**Plans**: TBD

---

### Phase 106: Integration & CUDA Graph

**Goal:** Users can combine beam search with speculative decoding and optimize all features with CUDA Graphs

**Depends on:** Phase 105

**Requirements:** BEAM-07, KV-05, KV-06

**Success Criteria** (what must be TRUE):

1. User can combine beam search with speculative decoding in the same inference loop
2. User can use dynamic block sizing (16/32/64 tokens) based on sequence access patterns
3. User can process long prompts (>16K tokens) with chunked prefill when memory-constrained
4. All features work correctly together without KV contamination or memory leaks

**Plans**: TBD

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 103. KV Cache Foundation | 0/1 | Not started | - |
| 104. Beam Search Core | 0/1 | Not started | - |
| 105. Speculative Decoding | 0/1 | Not started | - |
| 106. Integration & CUDA Graph | 0/1 | Not started | - |

---

## Coverage Map

```
SPEC-01 → Phase 105
SPEC-02 → Phase 105
SPEC-03 → Phase 105
SPEC-04 → Phase 105
SPEC-05 → Phase 105
SPEC-06 → Phase 105
SPEC-07 → Phase 105
SPEC-08 → Phase 105
SPEC-09 → Phase 105
BEAM-01 → Phase 104
BEAM-02 → Phase 104
BEAM-03 → Phase 104
BEAM-04 → Phase 104
BEAM-05 → Phase 104
BEAM-06 → Phase 104
BEAM-07 → Phase 106
BEAM-08 → Phase 104
KV-01 → Phase 103
KV-02 → Phase 103
KV-03 → Phase 103
KV-04 → Phase 103
KV-05 → Phase 106
KV-06 → Phase 106
KV-07 → Phase 103
KV-08 → Phase 103

Mapped: 25/25 ✓
```

---

*Roadmap created: 2026-05-05*
*v2.13 Transformer Optimization*
