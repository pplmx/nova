# Feature Research — Transformer Optimization Features

**Domain:** LLM Inference Optimization CUDA Library
**Researched:** 2026-05-05
**Confidence:** MEDIUM-HIGH (based on established algorithms, library implementations)

## Feature Landscape

### Table Stakes (Users Expect These)

Features considered essential for competitive LLM inference. Missing these = product feels incomplete compared to vLLM, TGI, or llama.cpp.

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| Speculative Decoding | Standard latency optimization in production deployments | HIGH | FlashAttention, Sequence Manager, Sampling |
| Beam Search | Quality vs latency tradeoff for generation tasks | MEDIUM | Attention infrastructure, Token scoring |
| KV Cache Improvements | Memory efficiency at scale, prefix reuse | MEDIUM | Existing BlockManager, KVCacheAllocator |

### Differentiators (Competitive Advantage)

Features that set this library apart from commodity implementations.

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| Async Speculative Draft Verification | Overlap verification with draft generation | HIGH | Speculative Decoding base, CUDA graphs |
| Tree-Based Decoding (Eagle/SnapKV) | Better speculation acceptance rates | HIGH | Attention kernel modifications |
| Cross-Sequence Prefix Sharing | Memory optimization for shared system prompts | MEDIUM | KVCacheAllocator prefix caching |
| Dynamic Block Sizing | Adaptive block sizes based on sequence patterns | LOW-MEDIUM | Existing block allocation |
| Chunked Prefill | Process long prompts in segments | MEDIUM | Attention infrastructure, Memory pool |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create complexity or conflicts.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full Beam Search with N>4 | Higher quality outputs | Memory O(N*seq_len), diminishing returns | Sampling + reranking |
| Speculative Decoding with 2+ draft models | Better acceptance rates | Complexity, coordination overhead | Single draft with tree decoding |
| Persistent KV Cache (disk) | Infinite context | Latency, complexity, consistency | Streaming context management |
| vLLM-compatible API surface | Easy migration | Constraints our design space | Clean abstraction layer |

## Feature Dependencies

```
[Speculative Decoding Base]
    ├──requires──> [Draft Model Interface]
    │                  └──requires──> [Token Scoring Infrastructure]
    └──requires──> [Tree-Based Verification Kernels]

[Beam Search]
    ├──requires──> [Attention Infrastructure]
    └──conflicts──> [Batched Sampling] (mutually exclusive batch modes)

[KV Cache Improvements]
    ├──enhances──> [Speculative Decoding] (reduces memory pressure)
    ├──enhances──> [Beam Search] (needed for multiple hypotheses)
    └──requires──> [BlockManager] (already built)

[Chunked Prefill]
    └──requires──> [Attention Infrastructure]
```

### Dependency Notes

- **Speculative Decoding requires FlashAttention:** The verification step needs O(1) KV access via existing block table
- **Speculative Decoding requires Sequence Manager:** Track draft and target sequences, coordinate acceptance
- **Beam Search conflicts with continuous batching:** Beam candidates within same batch require synchronization; consider phase separation or priority queues
- **KV Cache improvements enhance Speculative Decoding:** Multiple draft candidates benefit from reduced memory allocation overhead
- **KV Cache improvements enhance Beam Search:** N beam candidates need 2N KV cache blocks per step

## Expected Behavior

### Speculative Decoding

```
Algorithm Flow:
1. Draft model generates K candidate tokens (typically 4-8)
2. Target model verifies ALL candidates in single forward pass
   - Uses tree attention to process draft tokens in parallel
3. Acceptance/rejection based on logits comparison
   - Token accepted if target probability >= draft probability
   - Early exit on first rejection (greedy) or accept N tokens (sampling)
4. Accepted tokens added to output, process repeats

Expected Performance:
- Latency: 2-4x speedup for typical chat workloads
- Memory: ~K extra blocks per speculative sequence
- Throughput: Lower than pure sampling due to verification overhead

Key Implementation Details:
- Draft model: Smaller model (e.g., 7B draft, 70B target) OR
  same model with temperature sampling OR
  tree-based投机 (no separate draft)
- Verification: Single attention pass over draft tokens
- Tree attention: Mask pattern encodes prefix tree structure
```

### Beam Search

```
Algorithm Flow:
1. Start with K beam hypotheses (K = beam_width)
2. For each step:
   - Compute logits for all K * vocab_size
   - Select top K tokens per beam (score + log_prob)
   - Prune to K best overall
   - Expand sequences by chosen tokens
3. Continue until EOS token or max_length
4. Return best beam (or all K for reranking)

Expected Behavior:
- Quality: Significantly better than greedy sampling
- Latency: Kx memory, ~Kx time vs single sample
- Output variance: Lower (deterministic with greedy beam)

Key Implementation Details:
- KV cache: 2K blocks per active beam (K for current, K for history)
- Attention: Cross-beam attention via sequence multiplexing
- Memory management: Beam candidates share prefix KV until divergence
```

### KV Cache Improvements

#### Cross-Sequence Prefix Sharing
```
Use Case: System prompts, few-shot examples shared across requests

Implementation:
- Hash prefix tokens to identify reusable blocks
- Reference count blocks shared across sequences
- Copy-on-write for divergent suffixes
- Release when all references complete

Expected Behavior:
- Memory: 30-70% reduction for shared-prefix workloads
- Latency: Minimal overhead for hash lookup
- Hit rate: High for repeated system prompts, low for unique user input
```

#### Dynamic Block Sizing
```
Use Case: Mixed workload with varying sequence lengths

Implementation:
- Analyze sequence length distribution
- Select block sizes (16, 32, 64 tokens) per workload
- Smaller blocks: Better packing for short sequences
- Larger blocks: Less metadata overhead for long sequences

Expected Behavior:
- Memory: 5-15% improvement in utilization
- Complexity: Minimal, runtime overhead negligible
```

#### Chunked Prefill
```
Use Case: Long prompts that exceed GPU memory

Implementation:
- Split prefill into chunks fitting memory budget
- Process chunks sequentially, accumulate KV cache
- Use chunk boundaries for attention masking

Expected Behavior:
- Latency: Slight overhead per chunk boundary
- Memory: Bounded by chunk size, not full prompt
- Throughput: Lower than full prefill due to fragmentation
```

## MVP Definition

### Launch With (v1)

- [ ] **Speculative Decoding Base** — Single draft model, greedy acceptance
  - Essential for latency-sensitive applications
  - Dependencies: FlashAttention (built), Sequence Manager (built)
  - Why: 2-4x latency improvement is significant value

- [ ] **Beam Search Core** — Width 4, greedy selection
  - Table stakes for quality-focused use cases
  - Dependencies: Attention (built)
  - Why: Required for translation, summarization, code generation

- [ ] **Cross-Sequence Prefix Caching** — Basic hash-based lookup
  - Reduces memory for shared system prompts
  - Dependencies: KVCacheAllocator (built)
  - Why: High impact for multi-turn chat, RAG workloads

### Add After Validation (v1.x)

- [ ] **Tree-Based Speculative Decoding** — Tree attention kernels
  - Trigger: Need >2x acceptance rate improvement
  - Better utilization of draft compute

- [ ] **Beam Search Width Extension** — Support K=8,16
  - Trigger: Quality requirements from specific deployments
  - Memory vs quality tradeoff exploration

- [ ] **Dynamic Block Sizing** — Runtime block size optimization
  - Trigger: Observed fragmentation in production workloads
  - Adaptive to sequence length distribution

### Future Consideration (v2+)

- [ ] **Multi-model Speculative Decoding** — Multiple draft model support
  - Trigger: Hardware heterogeneity (different GPU sizes)
  - Complexity: Model coordination, fallback logic

- [ ] **Chunked Prefill** — Streaming prefill for long prompts
  - Trigger: Long-context applications (>16K tokens)
  - Depends on memory profiling from production

- [ ] **KV Cache Compression** — Value quantization/pruning
  - Trigger: Memory pressure at scale
  - Quality vs memory tradeoff research needed

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Risk |
|---------|------------|---------------------|----------|------|
| Speculative Decoding | HIGH | HIGH | P1 | Medium |
| Beam Search | HIGH | MEDIUM | P1 | Low |
| Prefix Caching | MEDIUM-HIGH | MEDIUM | P1 | Low |
| Tree Attention | MEDIUM | HIGH | P2 | High |
| Dynamic Block Sizing | MEDIUM | LOW | P2 | Low |
| Chunked Prefill | MEDIUM | MEDIUM | P2 | Medium |

**Priority key:**
- P1: Must have for launch (customer commitments, table stakes)
- P2: Should have, add when possible (significant value, manageable cost)
- P3: Nice to have, future consideration (speculative)

**Risk key:**
- Low: Straightforward implementation, clear spec
- Medium: Some unknowns, needs profiling
- High: Significant research needed, unclear approach

## Competitor Feature Analysis

| Feature | vLLM | TGI (HuggingFace) | llama.cpp | Our Approach |
|---------|------|-------------------|-----------|--------------|
| Speculative Decoding | Eagle, Medusa | Draft model API | Yes | Tree-based, unified with attention |
| Beam Search | Yes | Yes | Yes | K=4-16, memory optimized |
| Prefix Caching | Yes | Limited | No | Cross-sequence with ref counting |
| Chunked Prefill | Yes | Yes | Yes | Streaming with chunk boundaries |
| Dynamic Block Sizing | No | No | Yes | Adaptive based on workload |

### Implementation Approach Differences

- **Speculative Decoding:** vLLM uses multiple draft heads (Eagle), we focus on tree attention for single-draft scenarios. Better for latency-critical single-model deployments.
- **Beam Search:** vLLM optimized for throughput, we optimize for memory efficiency with shared prefix blocks.
- **Prefix Caching:** llama.cpp has no prefix caching. vLLM has sequence-level, we add cross-sequence reference counting.

## Integration with Existing Infrastructure

### Existing Components (Already Built)

| Component | Role in New Features |
|-----------|---------------------|
| `FlashAttention` | Core attention for verification, beam scoring |
| `KVCacheAllocator` | Block allocation for speculative/beam sequences |
| `BlockManager` | Sequence management, block tables |
| `Sequence` | Track tokens per hypothesis |
| `MemoryPool` | Workspace allocation for verification |

### Required Extensions

| Extension | Purpose |
|-----------|---------|
| `TreeAttention Kernel` | Process draft tokens in tree structure |
| `DraftModel Interface` | Pluggable draft model strategy |
| `BeamScorer` | Beam hypothesis management, pruning |
| `CrossSequenceCache` | Reference-counted prefix sharing |
| `ChunkedPrefillPipeline` | Streaming prefill orchestration |

## Sources

- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/serving/spec_decode.html
- HuggingFace TGI: https://huggingface.co/docs/text-generation-inference/conceptual/speculative_decoding
- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- Medusa (Speculative Decoding): https://arxiv.org/abs/2305.04437
- Eagle (Speculative Decoding): https://arxiv.org/abs/2401.15077
- StreamingLLM: https://arxiv.org/abs/2309.17453

---

*Feature research for: Transformer Optimization — Speculative Decoding, Beam Search, KV Cache Improvements*
*Researched: 2026-05-05*
