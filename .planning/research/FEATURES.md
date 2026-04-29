# Feature Research: GPU Transformer Inference

**Domain:** CUDA-based LLM inference optimization library
**Researched:** 2026-04-29
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| FlashAttention | Standard for memory-efficient attention in production; reduces HBM accesses via IO-aware tiling | MEDIUM | Core algorithm from arXiv:2205.14135; v2 is 2x faster than v1, v3 adds FP8 for Hopper, v4 for Blackwell |
| KV Cache | Essential for autoregressive decoding; eliminates recomputation across tokens | LOW | Standard concept; fragmentation is the real challenge (60-80% waste in naive implementations) |
| Paged KV Cache | KV cache management inspired by OS virtual memory; blocks map non-contiguously | MEDIUM | vLLM innovation (SOSP 2023); reduces waste from 60-80% to under 4%; enables copy-on-write for beam search |
| Attention Backend Selection | Hardware-specific kernel optimization (Ampere vs Hopper vs Blackwell) | MEDIUM | vLLM auto-selects: FA2 for Ampere, FA3 for Hopper, FA4 for Blackwell; supports FlashInfer, Triton, FlashMLA |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Continuous Batching | 8x throughput improvement over static batching; reduces p50 latency | HIGH | Iteration-level scheduling (Orca OSDI 2022); enables 23x throughput vs naive (vLLM); dynamic batch size per iteration |
| Sequence Parallelism | Scales context length across TP ranks; enables longer sequences | HIGH | Context Parallelism in vLLM; Decode Context Parallelism (DCP) for decode phase |
| Speculative Decoding | Reduces inter-token latency; 2-4x speedup for memory-bound workloads | HIGH | Multiple methods: EAGLE, MTP, n-gram, suffix; requires rejection sampling; lossless with proper implementation |
| Automatic Prefix Caching | Reuses KV cache for shared prefixes; eliminates redundant computation | MEDIUM | Critical for multi-turn conversations; memory sharing via block tables |
| Disaggregated Prefill/Decode | Separate prefill and decode stages; enables independent scaling | VERY HIGH | For very large batches; reduces prefill blocking |

## Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Static KV pre-allocation | Simple, predictable memory usage | 60-80% waste due to variable sequence lengths; limits batch size | Paged KV cache with on-demand allocation |
| Fixed batch sizes | Easier scheduling | GPU underutilization when sequences finish at different times | Continuous batching with iteration-level scheduling |
| Contiguous KV blocks | Simpler memory layout | Fragmentation across variable-length sequences | Non-contiguous blocks with block table mapping |

## Feature Dependencies

```
FlashAttention
    └──requires──> Optimized KV Cache Management

Paged Attention
    └──requires──> Block-based KV Cache Allocation
    └──enhances──> Continuous Batching (higher batch sizes)

Continuous Batching
    └──requires──> Iteration-level Scheduling
    └──enhances──> Automatic Prefix Caching

Speculative Decoding
    └──requires──> Rejection Sampling
    └──requires──> KV Cache for Draft/Target Models
    └──conflicts──> Pipeline Parallelism (vLLM <= 0.15.0)

Sequence Parallelism
    └──requires──> TP Infrastructure
    └──enhances──> Long Context Support
```

### Dependency Notes

- **FlashAttention requires KV Cache Management:** Cannot achieve IO-efficiency without proper KV cache handling
- **Paged Attention enhances Continuous Batching:** Enables higher batch sizes by eliminating fragmentation
- **Speculative Decoding conflicts with Pipeline Parallelism:** Current limitation in vLLM; requires all tokens in speculative tree to be on same GPU
- **Sequence Parallelism enhances Long Context:** Enables 128k+ context by distributing across TP ranks

## MVP Definition

### Launch With (v1)

Minimum viable product - essential for any credible inference library.

- [ ] **FlashAttention integration** - Core IO-aware attention; FA2 for Ampere/Ada, FA3 for Hopper, FA4 for Blackwell; supports fp16/bf16
- [ ] **Paged KV Cache** - Block-based allocation; <4% memory waste; block table for non-contiguous mapping
- [ ] **Attention backend auto-selection** - Automatic backend selection based on hardware; manual override option
- [ ] **Basic continuous batching** - Iteration-level scheduling; replaces static batching; enables dynamic batch size

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **Automatic Prefix Caching** - Reuse KV cache for shared prefixes; reduces multi-turn latency
- [ ] **Multiple attention backends** - FlashInfer, Triton, FlashMLA for specific hardware optimizations
- [ ] **GQA/MQA support** - Grouped-query attention for memory efficiency with large models

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Speculative Decoding** - EAGLE, MTP, n-gram methods; requires rejection sampling infrastructure
- [ ] **Sequence Parallelism** - Context Parallelism for long sequences; Decode Context Parallelism
- [ ] **Disaggregated Prefill/Decode** - For very large scale deployments; requires KV transfer infrastructure
- [ ] **Sparse Attention** - MLA sparse backends for DeepSeek-style models; reduces KV cache size

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| FlashAttention | HIGH | MEDIUM | P1 |
| Paged KV Cache | HIGH | MEDIUM | P1 |
| Attention Backend Auto-selection | HIGH | LOW | P1 |
| Continuous Batching | HIGH | HIGH | P1 |
| Automatic Prefix Caching | MEDIUM | MEDIUM | P2 |
| FlashInfer/Triton Backends | MEDIUM | MEDIUM | P2 |
| GQA/MQA Support | MEDIUM | LOW | P2 |
| Speculative Decoding | HIGH | VERY HIGH | P3 |
| Sequence Parallelism | MEDIUM | HIGH | P3 |
| Disaggregated Serving | MEDIUM | VERY HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | vLLM | TGI (HuggingFace) | FasterTransformer | Our Approach |
|---------|------|-------------------|-------------------|--------------|
| FlashAttention | FA2/FA3/FA4 | FA2 | Custom | Standard FA library integration; backend selection |
| Paged Attention | Native | No | No | Core innovation; block-based KV cache |
| Continuous Batching | Yes (optimized) | Yes | No | Same as vLLM |
| Speculative Decoding | EAGLE, MTP, n-gram, suffix | No | No | Implement EAGLE first (best latency reduction) |
| Sequence Parallelism | Context + Decode CP | Context CP only | Yes | Context CP first |

## Technical Implementation Notes

### FlashAttention API (v2)

```cuda
// Core functions from flash_attn_interface
flash_attn_func(q, k, v, dropout_p=0.0, causal=False);     // Standard attention
flash_attn_qkvpacked_func(qkv, causal=False);              // Packed QKV (faster)
flash_attn_with_kvcache(q, k_cache, v_cache, ...);         // Incremental decode
```

Supports: fp16, bf16, head dims up to 256, MQA/GQA, causal masking, sliding window, ALiBi

### PagedAttention Memory Layout

```
Key/Value Cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
Block Table: [num_seqs, max_num_blocks_per_seq] -> physical block numbers
```

Each sequence's logical blocks map to non-contiguous physical blocks via block table.

### Continuous Batching Lifecycle

1. Requests arrive -> added to batch at next iteration
2. Prefill phase (one-time cost) -> compute KV cache
3. Decode iterations -> generate one token per iteration
4. Request completes (EOS or max_len) -> removed from batch
5. GPU utilization stays high throughout

## Sources

- FlashAttention paper: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- FlashAttention GitHub: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- vLLM PagedAttention: [vLLM Blog (2023)](https://blog.vllm.ai/2023/06/20/vllm.html)
- vLLM paper: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- vLLM Speculative Decoding: [vLLM Docs](https://docs.vllm.ai/en/latest/features/speculative_decoding.html)
- Continuous Batching: [Anyscale Blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- Orca (continuous batching origin): [OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)
- vLLM Attention Backends: [vLLM Docs](https://docs.vllm.ai/en/latest/design/attention_backends.html)

---

*Feature research for: Transformer & Inference Optimization*
*Researched: 2026-04-29*
