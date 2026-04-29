# Project Research Summary

**Project:** Nova CUDA Library Enhancement
**Domain:** GPU Transformer Inference Optimization
**Researched:** 2026-04-29
**Confidence:** HIGH

## Executive Summary

This milestone focuses on optimizing transformer inference by integrating FlashAttention, implementing paged KV cache management, and enabling sequence parallelism. The existing five-layer CUDA architecture (memory → device → algo → api) provides a solid foundation, requiring a new `inference/` layer to orchestrate the tight coupling between these components. NVIDIA TransformerEngine (TE) C API is the recommended foundation - it provides FlashAttention via TE's `DotProductAttention`, native paged KV cache support via block tables, and CUDA Graphs integration via `make_graphed_callables`.

Research confirms that FlashAttention v2 reduces HBM bandwidth from O(N²d) to O(Nd) with IO-aware tiling, while vLLM's paged attention reduces KV cache memory waste from 60-80% to under 4%. These are table-stakes features for production LLM inference libraries.

## Key Findings

### Recommended Stack

**Core technologies:**
- **TransformerEngine 2.14+** — Native C/C++ API, paged KV cache via `DotProductAttention`, CUDA Graphs support, FP8/FP4 quantization
- **FlashAttention 2.5.x** — IO-aware exact attention for Ampere/Ada; 2-4x faster than standard attention with O(n) memory
- **FlashAttention 3 beta** — For Hopper (H100/H800) with FP8 forward pass support
- **cuDNN 9.x** — Low-level attention primitives; TE uses cuDNN internally

### Expected Features

**Must have (table stakes):**
- FlashAttention integration (FA2 for Ampere/Ada, FA3 for Hopper, FA4 for Blackwell)
- Paged KV Cache with block-based allocation and <4% memory waste
- Attention backend auto-selection based on hardware detection
- Basic continuous batching with iteration-level scheduling

**Should have (competitive):**
- Automatic prefix caching for multi-turn conversations
- GQA/MQA support for memory efficiency
- Multiple attention backends (FlashInfer, Triton, FlashMLA)

**Defer (v2+):**
- Speculative decoding (EAGLE, MTP, n-gram) - requires rejection sampling infrastructure
- Sequence parallelism via ring attention
- Disaggregated prefill/decode for large-scale deployments

### Architecture Approach

New `cuda/inference/` module for orchestration layer:
```
include/cuda/inference/
├── block_manager.h      # Paged attention block manager
├── sequence_manager.h   # Sequence lifecycle management
├── scheduler.h          # Batching and scheduling
├── cache_manager.h      # KV cache coordination
└── paged_attention.h    # Paged attention kernels
```

FlashAttention kernels integrate into existing `cuda/algo/` layer. KV cache allocator extends `cuda/memory/`. Sequence parallelism extends `cuda/distributed/`.

### Critical Pitfalls

1. **Softmax numerical overflow** — Use stable softmax with max subtraction; exp(large) overflows
2. **Head dimension alignment** — FlashAttention v2 requires head_dim % 8 == 0; pad unsupported dims
3. **Workspace allocation** — Query size per-config; allocate dynamically, not at init
4. **CPU-GPU block table sync** — Use dedicated sync stream; cudaStreamSynchronize before kernel launch
5. **KV cache fragmentation** — Block-based fixed-size allocation (power of 2); never per-token allocation

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 69: FlashAttention Integration
**Rationale:** Core attention primitive; must be stable before adding paged cache
**Delivers:** FlashAttention kernel, attention backend selection, stable softmax
**Implements:** `cuda/algo/flash_attention.h`
**Avoids:** Softmax overflow, head dim misalignment, workspace sizing bugs

### Phase 70: Paged KV Cache Foundation
**Rationale:** Memory infrastructure needed by paged attention
**Delivers:** KVCacheAllocator, block allocation/deallocation, LRU eviction
**Implements:** `cuda/memory/kv_cache_allocator.h`
**Avoids:** Memory fragmentation, stale block references

### Phase 71: Paged Attention Integration
**Rationale:** Combines FlashAttention with paged KV cache for production efficiency
**Delivers:** BlockManager, block table management, CPU-GPU synchronization
**Implements:** `cuda/inference/block_manager.h`
**Avoids:** Block table consistency issues, out-of-bounds access

### Phase 72: Sequence Manager & Scheduler
**Rationale:** Enables continuous batching and multi-sequence management
**Delivers:** SequenceManager, continuous batching scheduler
**Implements:** `cuda/inference/scheduler.h`
**Avoids:** Variable sequence length contamination

### Phase 73: Sequence Parallelism Extension
**Rationale:** Long context support via distributed computation
**Delivers:** SequenceParallelAttention, ring sequence parallelism
**Implements:** `cuda/distributed/sequence_parallel.h`
**Avoids:** TP/PP gradient deadlocks

### Phase 74: Integration & Testing
**Rationale:** End-to-end validation and optimization
**Delivers:** CUDA Graph capture, NVTX annotations, throughput benchmarks
**Avoids:** Integration gaps, performance regressions

### Phase Ordering Rationale

1. FlashAttention first: Core kernel must be correct before orchestration
2. KV cache second: Infrastructure dependency for paged attention
3. Block manager third: Combines FA + KV cache with block table logic
4. Scheduler fourth: Multi-sequence support builds on single-sequence block manager
5. Sequence parallelism last: Depends on single-GPU working correctly
6. Integration final: All components must work together

### Research Flags

Phases needing deeper research during planning:
- **Phase 69:** TE C API header integration vs. compiled library; FATBIN kernel bundling
- **Phase 72:** CUDA Graph compatibility with dynamic block allocation
- **Phase 73:** Ring attention vs. context parallelism tradeoff

Phases with standard patterns (well-documented):
- **Phase 70:** Block allocator is straightforward; similar to OS memory management
- **Phase 71:** Block table pattern from vLLM is production-proven

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | NVIDIA official docs, FlashAttention repo, vLLM production use |
| Features | HIGH | Based on vLLM, TGI, FasterTransformer analysis |
| Architecture | HIGH | Five-layer pattern fits naturally; inference module follows existing patterns |
| Pitfalls | HIGH | Based on FlashAttention implementation, vLLM source analysis |

**Overall confidence:** HIGH

### Gaps to Address

- **TE C API availability:** Verify TE C API is installable without PyTorch dependency
- **Block size tuning:** Optimal block size (16/32/64) may vary by GPU architecture
- **Backward pass:** FlashAttention backward for training support timeline

## Sources

### Primary (HIGH confidence)
- [NVIDIA TransformerEngine 2.14 Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) — TE C API, DotProductAttention
- [FlashAttention GitHub (Dao-AILab)](https://github.com/Dao-AILab/flash-attention) — Algorithm, kernel architecture
- [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/) — FA3 FP8 support for Hopper

### Secondary (MEDIUM confidence)
- [vLLM Architecture](https://github.com/vllm-project/vllm) — PagedAttention, continuous batching patterns
- [Ring Attention Paper](https://arxiv.org/abs/2310.02589) — Sequence parallelism algorithm

---
*Research completed: 2026-04-29*
*Ready for roadmap: yes*
