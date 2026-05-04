# Architecture Research: Transformer Inference Optimization

**Domain:** CUDA transformer inference optimization
**Researched:** 2026-05-05
**Confidence:** MEDIUM-HIGH

## Existing Architecture (5-Layer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer (scheduler.h)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │    Scheduler    │───▶│ SequenceManager │    │ InferenceGraphExecutor  │  │
│  │  (batch logic)  │    │  (state mgmt)   │    │  (CUDA Graph capture)   │  │
│  └────────┬────────┘    └─────────────────┘    └─────────────────────────┘  │
├───────────┴─────────────────────────────────────────────────────────────────┤
│                         ALGO Layer (flash_attention.h)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FlashAttention + PagedAttention (static forward, KV cache lookup)  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                      MEMORY Layer (kv_cache_allocator.h)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  KVCacheAllocator (block-based, LRU eviction, prefix caching)       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                      DEVICE Layer (stream.h, buffer.h)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐    ┌───────────────────┐    ┌────────────────────┐   │
│  │  Stream (async)   │    │  Buffer (unified) │    │  AMPManager (fp8)  │   │
│  └───────────────────┘    └───────────────────┘    └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Existing Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `Scheduler` | Batch composition, request management | `inference/scheduler.h` |
| `SequenceManager` | Sequence state machine (Running/Waiting/Finished) | `inference/scheduler.h` |
| `BlockManager` | KV block allocation per sequence | `inference/block_manager.h` |
| `KVCacheAllocator` | Physical KV block pool, LRU, prefix cache | `memory/kv_cache_allocator.h` |
| `PagedAttention` | Static forward with block_table lookup | `inference/block_manager.h` |
| `InferenceGraphExecutor` | CUDA Graph capture/replay | `production/inference_graph.h` |

---

## New Components Required

### 1. Speculative Decoding

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Speculative Decoding Integration                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Scheduler                                                                │
│       │                                                                    │
│       ├──▶ SpeculativeDecodingRunner (NEW)                                │
│       │         │                                                          │
│       │         ├── DraftModel (lighter transformer, kvcache-aware)       │
│       │         │         │                                               │
│       │         │         └── PagedAttention (reuse existing)             │
│       │         │                                                          │
│       │         ├── VerificationKernel (accept/reject per token)          │
│       │         │         │                                               │
│       │         │         └── Uses: BlockManager.get_sequence()           │
│       │         │                                                          │
│       │         └── LogProbAccumulator (for acceptance tracking)          │
│       │                   │                                               │
│       │                   └── Uses: SequenceManager state                  │
│       │                                                                    │
│       └──▶ BlockManager (MODIFIED)                                        │
│                 │                                                          │
│                 └── Multi-head KV tracking per speculation batch          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**NEW components:**

| Component | File | Purpose | Integration Point |
|-----------|------|---------|-------------------|
| `SpeculativeDecodingRunner` | `inference/speculative_decoding.h` | Orchestrates draft→verify loop | `Scheduler::step()` |
| `DraftModel` | `neural/draft_model.h` | Lightweight speculation model | Injected via config |
| `VerificationKernel` | `algo/speculative_verify.cuh` | Batch token acceptance | Uses `FlashAttention` |
| `LogProbTracker` | `inference/logprob_tracker.h` | Track token acceptance rates | Writes to Sequence metadata |

**MODIFIED components:**

| Component | Change | Why |
|-----------|--------|-----|
| `Scheduler` | Add `speculative_batch_` field, modify `step()` | Drive speculation loop |
| `BlockManager` | Support branched block tables for draft sequences | Each speculation has own KV |
| `Sequence` | Add `parent_id` for tree-structured sequences | Track draft lineage |

---

### 2. Beam Search

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Beam Search Integration                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Scheduler                                                                │
│       │                                                                    │
│       ├──▶ BeamSearchManager (NEW)                                        │
│       │         │                                                          │
│       │         ├── std::vector<Sequence*> beams_                          │
│       │         │         │                                               │
│       │         │         └── Each beam = separate KV cache allocation    │
│       │         │                                                          │
│       │         ├── beam_width_                                           │
│       │         │                                                          │
│       │         └── TopKSelector (beam scoring + pruning)                  │
│       │                   │                                               │
│       │                   └── Uses: LogProbs from each beam                │
│       │                                                                    │
│       └── KVCacheAllocator (MODIFIED)                                      │
│                 │                                                          │
│                 └── Support batched beam allocation per layer              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**NEW components:**

| Component | File | Purpose | Integration Point |
|-----------|------|---------|-------------------|
| `BeamSearchManager` | `inference/beam_search.h` | Beam state, scoring, pruning | `Scheduler::on_token_generated()` |
| `BeamSequence` | `inference/beam_sequence.h` | Extended Sequence with beam_id | Aggregates into `BlockManager` |
| `TopKSampler` | `algo/beam_sampler.h` | Top-k selection for beams | Uses `FlashAttention` output |

**MODIFIED components:**

| Component | Change | Why |
|-----------|--------|-----|
| `Sequence` | Add `beam_score`, `beam_id` | Track beam membership and score |
| `BlockManager` | Batch KV operations across beams | Parallel block allocation |
| `KVCacheAllocator` | Batch allocation for beam_width sequences | Efficiency |
| `Scheduler` | Support `beam_width > 1` in config | Enable beam search mode |

---

### 3. KV Cache Improvements

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      KV Cache Improvement Integration                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   KVCacheAllocator (MODIFIED)                                              │
│       │                                                                    │
│       ├──▶ StreamingCacheManager (NEW)                                    │
│       │         │                                                          │
│       │         ├── PrefetchQueue (async KV fetch)                        │
│       │         │         │                                               │
│       │         │         └── Uses: DeviceContext for DMA                 │
│       │         │                                                          │
│       │         ├── EvictionPredictor (NEW)                               │
│       │         │         │                                               │
│       │         │         └── ML-based or heuristics-based eviction       │
│       │         │                                                          │
│       │         └── L2CacheHint (NEW)                                     │
│       │                   │                                               │
│       │                   └── CUDA L2 persistence hints via cudaStreamAttr │
│       │                                                                    │
│       ├── PagedAttention (MODIFIED)                                       │
│       │         │                                                          │
│       │         └── AttentionWithPersistentCache (NEW kernel variant)     │
│       │                                                                    │
│       └── InferenceGraphExecutor (MODIFIED)                                │
│                 │                                                          │
│                 └── PersistentCacheReplay (static shape optimization)      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**NEW components:**

| Component | File | Purpose | Integration Point |
|-----------|------|---------|-------------------|
| `StreamingCacheManager` | `memory/streaming_cache.h` | Async prefetch/evict | `KVCacheAllocator` constructor |
| `EvictionPredictor` | `memory/eviction_predictor.h` | Intelligent cache eviction | `KVCacheAllocator::evict()` |
| `L2PersistenceHint` | `memory/l2_persistence.h` | CUDA L2 set-aside for KV | `Stream` configuration |
| `PersistentAttention` | `algo/persistent_attention.h` | FlashAttention variant with pinned KV | Replaces `PagedAttention` |
| `KVCacheCompressor` | `memory/kv_compressor.h` | Token-level KV compression | Post-append hook in allocator |

**MODIFIED components:**

| Component | Change | Why |
|-----------|--------|-----|
| `KVCacheAllocator` | Add `eviction_predictor_`, `streaming_manager_` | Enable intelligent caching |
| `BlockManager` | Support variable block sizes | Different cache tiers |
| `PagedAttention` | Add persistent cache variant | L2-resident attention |
| `InferenceGraphExecutor` | Static shape assumption with persistent cache | CUDA Graph optimization |

---

## Integration Points Summary

| Feature | Attention Module | Sequence Manager | Cache Layer | Scheduler |
|---------|-----------------|------------------|-------------|-----------|
| **Speculative Decoding** | Draft uses `FlashAttention` | Tree-structured states | Branched block tables | `step()` modified |
| **Beam Search** | Parallel forward per beam | Per-beam `Sequence*` | Batch allocation | `get_batch()` modified |
| **KV Improvements** | Persistent cache variant | Extended metadata | New eviction/persist | Minimal change |

---

## Data Flow Changes

### Speculative Decoding Flow

```
Scheduler::step()
    │
    ├─▶ SpeculativeDecodingRunner::speculate()
    │       │
    │       ├─▶ DraftModel.forward() → draft_tokens[N]
    │       │       │
    │       │       └── Append to draft KV (BlockManager)
    │       │
    │       └─▶ VerificationKernel::verify()
    │               │
    │               ├─▶ TargetModel.forward(draft_tokens)
    │               │
    │               ├─▶ Compare logprobs
    │               │
    │               └─▶ Accept/reject per token → accepted_tokens[]
    │
    └─▶ BlockManager.append_tokens() for accepted tokens
```

### Beam Search Flow

```
Scheduler::get_batch()
    │
    └─▶ BeamSearchManager::get_next_beams()
            │
            ├─▶ For each beam:
            │       │
            │       ├─▶ FlashAttention.forward() → logprobs
            │       │
            │       └─▶ BlockManager.forward() for KV update
            │
            ├─▶ TopKSampler::select_topk(beam_width)
            │       │
            │       └─▶ Scores = existing_score + logprobs
            │
            └─▶ Prune to beam_width, duplicate KV blocks if branching
```

### KV Cache Improvement Flow

```
KVCacheAllocator::append()
    │
    ├─▶ allocate_blocks_internal()
    │
    ├─▶ StreamingCacheManager::async_prefetch()
    │       │   (if prefix match found in slow tier)
    │       │
    │       └─▶ L2PersistenceHint::pin_for_persistence()
    │
    └─▶ EvictionPredictor::should_evict()
            │
            └─▶ Decide based on access pattern prediction
```

---

## Build Order (Dependency-Aware)

```
Phase 1: KV Cache Foundation
├── NEW: memory/streaming_cache.h
├── NEW: memory/eviction_predictor.h  
├── MOD: memory/kv_cache_allocator.h (add predictor, streaming hooks)
└── MOD: algo/persistent_attention.h (L2-aware kernel)

Phase 2: Beam Search Core
├── NEW: inference/beam_search.h
├── NEW: inference/beam_sequence.h
├── MOD: inference/scheduler.h (beam mode, batch composition)
├── MOD: inference/block_manager.h (batch KV ops)
└── MOD: memory/kv_cache_allocator.h (batch allocation)

Phase 3: Speculative Decoding
├── NEW: inference/speculative_decoding.h
├── NEW: inference/logprob_tracker.h
├── NEW: neural/draft_model.h (interface only, model-agnostic)
├── MOD: inference/scheduler.h (speculation loop)
├── MOD: inference/block_manager.h (branched sequences)
└── MOD: inference/scheduler.h (verify pass integration)

Phase 4: Integration & CUDA Graph
├── MOD: production/inference_graph.h (persistent cache support)
├── NEW: algo/persistent_attention_test.cpp
├── MOD: tests/inference/ (add spec/beam/KV tests)
└── MOD: tests/benchmark/inference_benchmark.cpp
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Speculative Decoding Without KV Isolation

**What people do:** Share KV cache between draft and target model.

**Why wrong:** Verification requires independent target KV to compute acceptance correctly. Sharing corrupts the verification signal.

**Do this instead:** Each speculation iteration allocates isolated KV blocks. Accepted tokens merge into parent sequence.

### Anti-Pattern 2: Beam Search With Copy-on-Write

**What people do:** Copy entire KV cache for each beam.

**Why wrong:** Memory explosion. For beam_width=4, seq_len=2048, memory grows 4x.

**Do this instead:** Use reference counting. Only fork blocks that diverge (prefix shared, suffix copied).

### Anti-Pattern 3: Blocking Prefetch in Cache Append

**What people do:** Synchronously fetch from CPU/GPU-NIC before returning from `append()`.

**Why wrong:** Destroys streaming pipeline parallelism. GPU stalls waiting for cache miss.

**Do this instead:** Async prefetch with callback notification. Return immediately, complete fetch before next attention pass.

### Anti-Pattern 4: CUDA Graph With Variable Batch Size

**What people do:** Capture CUDA Graph with batch_size=1, replay with batch_size=32.

**Why wrong:** Graph captures exact kernel shapes. Dynamic batch requires node updates or separate graphs.

**Do this instead:** Use persistent KV + single-sequence graph, or implement conditional graph nodes for dynamic shapes.

---

## Scaling Considerations

| Scale | KV Cache | Speculative Decoding | Beam Search |
|-------|----------|---------------------|-------------|
| 1-4 sequences | LRU sufficient | May not help (overhead > gain) | beam_width=2-4 |
| 5-32 sequences | LRU + prefix cache | draft_depth=3-5 | beam_width=2-4 |
| 32-128 sequences | Streaming + L2 persistence | draft_depth=5-8 | beam_width=4-8 |
| 128+ sequences | Multi-tier (GPU/CPU/NIC) | Only high-hit prefix workloads | beam_width=2 (overhead) |

---

## Sources

- [vLLM Speculative Decoding Architecture](https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode/)
- [NVIDIA Transformer Engine Beam Search](https://github.com/NVIDIA/TransformerEngine)
- [CUDA L2 Cache Persistence Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- Existing nova codebase: `include/cuda/inference/`, `include/cuda/memory/kv_cache_allocator.h`

---

*Architecture research for: Transformer Inference Optimization*
*Researched: 2026-05-05*
