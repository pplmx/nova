# Milestone v2.6 Requirements

**Milestone:** Transformer & Inference Optimization
**Created:** 2026-04-29
**Status:** Active

## v2.6 Requirements

### FlashAttention Integration

- [ ] **FA-01**: User can select attention backend (Standard/FlashAttention/PagedAttention) via enum configuration
- [ ] **FA-02**: User can compute attention forward pass using FlashAttention with IO-aware tiling
- [ ] **FA-03**: FlashAttention supports BF16 and FP16 datatypes with stable softmax normalization
- [ ] **FA-04**: User can compute attention backward pass for training with deterministic dropout

### KV Cache Management

- [ ] **KV-01**: User can allocate/deallocate KV cache blocks with fixed power-of-2 sizes (16/32/64 tokens)
- [ ] **KV-02**: User can evict least-recently-used KV cache blocks when memory pressure exceeds threshold
- [ ] **KV-03**: User can cache KV blocks by prefix hash for multi-turn conversation reuse
- [ ] **KV-04**: User can query KV cache statistics (total/used/free blocks, fragmentation percentage)

### Paged Attention

- [ ] **PA-01**: User can create sequence with logical blocks mapped to non-contiguous physical blocks via block table
- [ ] **PA-02**: User can append tokens to sequence by allocating additional physical blocks
- [ ] **PA-03**: Block table updates are synchronized on dedicated stream before kernel launch
- [ ] **PA-04**: User can compute attention using block tables with out-of-bounds validation

### Scheduler & Batching

- [ ] **SCHED-01**: User can create and manage multiple sequences with independent KV cache state
- [ ] **SCHED-02**: User can schedule batched forward pass across variable-length sequences with iteration-level scheduling
- [ ] **SCHED-03**: FlashAttention supports grouped-query attention (GQA) and multi-query attention (MQA)

### Sequence Parallelism

- [ ] **SP-01**: User can compute attention with sequence dimension split across tensor parallel ranks
- [ ] **SP-02**: Ring sequence parallelism enables attention over sequences exceeding single-GPU memory
- [ ] **SP-03**: Sequence parallelism integrates with existing tensor parallelism communicator infrastructure

---

## Future Requirements (Deferred)

- **FA-05**: FlashAttention-3 support for Hopper FP8 forward pass
- **FA-06**: FlashAttention-4 support for Blackwell CuTeDSL kernels
- **KV-05**: KV cache compression for reduced memory footprint
- **PA-05**: Copy-on-write KV cache for beam search optimization
- **SCHED-04**: Speculative decoding with rejection sampling
- **SCHED-05**: Disaggregated prefill/decode for large-scale serving
- **SP-04**: Decode context parallelism for decode-phase sequence distribution

---

## Out of Scope

- **Python bindings** — Separate project
- **Triton kernels** — AMD ROCm support out of scope for NVIDIA-focused milestone
- **FlashAttention-4** — Requires Blackwell (B200) which is not yet widely deployed

---

## Traceability

| Requirement | Phase | Success Criteria |
|-------------|-------|------------------|
| FA-01 | Phase 69 | Backend enum compiles, selection changes attention implementation |
| FA-02 | Phase 69 | FlashAttention forward produces identical output to standard attention |
| FA-03 | Phase 69 | BF16/FP16 outputs match within 1e-3 relative error |
| FA-04 | Phase 69 | Backward pass gradients sum correctly across batch dimension |
| KV-01 | Phase 70 | Block allocation completes in O(1) from freelist |
| KV-02 | Phase 70 | LRU eviction triggers when free_blocks < threshold |
| KV-03 | Phase 70 | Prefix cache hit reduces recomputation by 100% for shared prefixes |
| KV-04 | Phase 70 | KVCacheStats reflects actual allocation state |
| PA-01 | Phase 71 | BlockManager.create_sequence returns valid block table |
| PA-02 | Phase 71 | append_tokens allocates new physical blocks |
| PA-03 | Phase 71 | cudaStreamSynchronize called before attention kernel |
| PA-04 | Phase 71 | Paged attention output matches contiguous attention within tolerance |
| SCHED-01 | Phase 72 | Multiple sequences can coexist with independent state |
| SCHED-02 | Phase 72 | Batched forward processes variable-length sequences correctly |
| SCHED-03 | Phase 72 | GQA/MQA produces correct output with num_kv_heads < num_q_heads |
| SP-01 | Phase 73 | Sequence attention output matches single-GPU result |
| SP-02 | Phase 73 | Ring attention handles sequences up to 128K tokens |
| SP-03 | Phase 73 | TP communicator correctly reduces sequence parallel output |

---
*Requirements defined: 2026-04-29*
*v2.6: Transformer & Inference Optimization*
