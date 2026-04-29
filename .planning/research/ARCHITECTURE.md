# Architecture Research: Transformer & Inference Optimization

**Domain:** Nova CUDA Library - Transformer Inference Architecture
**Researched:** 2026-04-29
**Confidence:** HIGH (based on FlashAttention paper, vLLM architecture, and existing codebase patterns)
**For:** Transformer & Inference Optimization milestone

## Executive Summary

Transformer inference optimization requires integration at four key points: memory management (KV cache), attention computation (FlashAttention), block management (paged attention), and distributed computation (sequence parallelism). The existing five-layer architecture (`memory → device → algo → api`) provides a solid foundation, but a new `inference/` layer is needed to orchestrate the tight coupling between these components. This document recommends a layered approach where FlashAttention kernels live in `algo/`, KV cache in `memory/`, block management in `inference/`, and sequence parallelism extends `distributed/`.

## Current State Analysis

### Existing Infrastructure

| Component | Location | Status | Gap for Inference |
|-----------|----------|--------|-------------------|
| Transformer Layer | `cuda/neural/` | MHA, FFW patterns | No FlashAttention integration |
| Memory Pool | `cuda/memory/` | Fragmentation tracking | No KV cache-specific allocator |
| Distributed Matmul | `cuda/distributed/` | TP/PP patterns | No sequence parallelism |
| Graph Executor | `cuda/production/` | CUDA Graph capture | No attention kernel capture |
| NVTX | `cuda/observability/` | Domain annotations | No inference-specific spans |

### Five-Layer Architecture with Inference Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Inference Layer (NEW)                           │
│          (BlockManager, Scheduler, SequenceManager)                  │
│                    Depends on: algo, memory, distributed            │
├─────────────────────────────────────────────────────────────────────┤
│                         API Layer                                    │
│              (include/cuda/api/) - Public interface                 │
│                     Depends on: algo layer                          │
├─────────────────────────────────────────────────────────────────────┤
│                        Algorithm Layer                               │
│          (include/cuda/algo/) - FlashAttention, kernels             │
│                     Uses: device, memory, inference                 │
├─────────────────────────────────────────────────────────────────────┤
│                         Device Layer                                 │
│           (include/cuda/device/) - Device management                │
│              Shared: reduce kernels, warp/block primitives          │
├─────────────────────────────────────────────────────────────────────┤
│                        Memory Layer                                  │
│           (include/cuda/memory/) - Buffer, MemoryPool, KVCache      │
│         Reusable by: all algorithm domains                          │
├─────────────────────────────────────────────────────────────────────┤
│                        Stream Layer                                  │
│            (include/cuda/stream/) - Async operations                │
│         Reusable by: all algorithm domains                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. FlashAttention Integration

### 1.1 FlashAttention CUDA Kernel Architecture

FlashAttention computes attention in tiles that fit in SRAM, reducing HBM bandwidth from O(N²d) to O(Nd) while maintaining numerical stability via online softmax.

**Key architectural decisions from FlashAttention v2/v3:**

| Aspect | Standard Attention | FlashAttention | Why |
|--------|-------------------|----------------|-----|
| Memory access | O(N²d) HBM reads | O(Nd) HBM reads | Tiled computation |
| SRAM usage | Minimal | O(Nd) per tile | Reuse in registers |
| Numerical stability | Exp-sum normalization | Online normalization | Single pass |
| Autoregressive mask | Post-compute | Streaming inference | Causal masking in-place |

**Tile size selection:**
- SMEM limit: 64KB-228KB depending on compute capability
- Block size: 64x64 or 128x64 (threads x head_dim)
- Number of splits: 1 for long sequences, >1 for memory-constrained cases

### 1.2 Integration with Existing Algo Layer

```cpp
// include/cuda/algo/flash_attention.h
#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/stream/stream.h"
#include <cstddef>

namespace nova::algo {

// FlashAttention algorithm interface
class FlashAttention {
public:
    struct Config {
        bool enable_dropout = false;
        float dropout_scale = 1.0f;
        bool enable_causal_mask = true;
        int num_splits = 1;  // For very long sequences
        int tile_size = 64;
        bool is_fp16 = true;
        bool is_bias = false;
    };

    FlashAttention() = default;
    explicit FlashAttention(const Config& config) : config_(config) {}

    // Forward pass: Q (query), K (key), V (value) -> O (output)
    void forward(
        memory::Buffer& output,
        const memory::Buffer& query,    // [seq_len, batch, num_heads, head_dim]
        const memory::Buffer& key,
        const memory::Buffer& value,
        memory::Buffer& softmax_lse,     // [num_heads, num_blocks] log-sum-exp
        const Stream& stream
    );

    // Backward pass for training
    void backward(
        memory::Buffer& dq,
        memory::Buffer& dk,
        memory::Buffer& dv,
        const memory::Buffer& output,
        const memory::Buffer& dout,
        const memory::Buffer& query,
        const memory::Buffer& key,
        const memory::Buffer& value,
        const memory::Buffer& softmax_lse,
        const Stream& stream
    );

    Config config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }

private:
    Config config_;
};

// Factory for creating FlashAttention with appropriate kernel
std::unique_ptr<FlashAttention> create_flash_attention(
    const FlashAttention::Config& config,
    int num_heads,
    int head_dim
);

}  // namespace nova::algo
```

### 1.3 Replacement Pattern for Standard Attention

FlashAttention replaces standard attention in the `TransformerLayer` while maintaining the same interface:

```cpp
// include/cuda/neural/transformer_layer.h (extension)
namespace nova::neural {

// Configurable attention backend
enum class AttentionBackend {
    Standard,      // Original implementation
    FlashAttention, // FlashAttention v2/v3
    PagedAttention // vLLM-style block attention
};

template<typename Config>
class TransformerLayer {
public:
    void set_attention_backend(AttentionBackend backend) {
        attention_backend_ = backend;
        
        switch (backend) {
            case AttentionBackend::FlashAttention:
                attention_impl_ = std::make_unique<algo::FlashAttention>(
                    flash_attention_config_
                );
                break;
            case AttentionBackend::Standard:
                attention_impl_ = std::make_unique<MHAImpl>(...);
                break;
            // PagedAttention handled separately via KVCacheManager
        }
    }

private:
    AttentionBackend attention_backend_ = AttentionBackend::FlashAttention;
    std::unique_ptr<AttentionImpl> attention_impl_;
    algo::FlashAttention::Config flash_attention_config_;
};

}  // namespace nova::neural
```

---

## 2. KV Cache Allocator

### 2.1 KV Cache Memory Layout

The KV cache stores key and value tensors for all tokens in the sequence, accessed during autoregressive decoding.

**Memory layout options:**

| Layout | Shape | Access Pattern | Memory Efficiency |
|--------|-------|----------------|-------------------|
| Contiguous | [seq_len, batch, num_heads, head_dim] | Sequential | High for prefill, wasted during decode |
| Paged (vLLM) | Fixed blocks with pointers | Random access | High fragmentation control |
| Hybrid | Contiguous + paged swap | Mixed | Best of both worlds |

**For Nova:** Paged layout is recommended for inference workloads due to variable sequence lengths.

### 2.2 KV Cache Allocator Design

```cpp
// include/cuda/memory/kv_cache_allocator.h
#pragma once

#include "cuda/memory/buffer.h"
#include <cstddef>
#include <vector>
#include <unordered_map>

namespace nova::memory {

class KVCacheAllocator {
public:
    struct Block {
        void* data;                  // GPU pointer to block memory
        int block_id;                // Unique identifier
        int num_tokens;              // Tokens in this block
        bool is_allocated;           // Currently in use
        int64_t sequence_id;         // Which sequence owns this
        Block* prev;                 // Linked list for sequence
        Block* next;
    };

    struct Config {
        int num_heads = 32;
        int head_dim = 128;
        int max_block_tokens = 16;   // Tokens per block (power of 2)
        int num_blocks = 1024;       // Total blocks (determines max memory)
        int num_layers = 32;         // For multi-layer models
        bool enable_prefix_caching = true;
    };

    explicit KVCacheAllocator(const Config& config);
    ~KVCacheAllocator();

    // Allocate blocks for a new sequence
    std::vector<Block*> allocate(int64_t sequence_id, int num_tokens);

    // Append tokens to existing sequence (during decoding)
    std::vector<Block*> append(int64_t sequence_id, int num_tokens);

    // Free all blocks for a sequence
    void free(int64_t sequence_id);

    // Evict blocks when memory pressure (LRU)
    void evict(int num_blocks_needed);

    // Get blocks for a sequence (for attention computation)
    std::vector<Block*> get_blocks(int64_t sequence_id) const;

    // Block-level access for paged attention
    Block* get_block(int64_t sequence_id, int block_index) const;

    // Prefix caching: find blocks matching prefix tokens
    struct PrefixMatch {
        int64_t sequence_id;
        int num_matching_tokens;
        int first_block_index;
    };
    std::optional<PrefixMatch> find_prefix_match(
        const void* prefix_tokens,
        int prefix_length
    ) const;

    // Statistics
    struct KVCacheStats {
        int total_blocks;
        int allocated_blocks;
        int free_blocks;
        float fragmentation_percent;
        size_t total_memory;
        size_t used_memory;
    };
    KVCacheStats get_stats() const;

private:
    Config config_;
    std::vector<Block> blocks_;                    // All blocks
    std::vector<Block*> free_list_;                // Available blocks
    std::unordered_map<int64_t, std::vector<Block*>> sequence_blocks_;
    std::unordered_map<int64_t, std::pair<Block*, Block*>> sequence_ranges_;
    
    // Prefix cache index: hash(tokens) -> block
    std::unordered_map<uint64_t, Block*> prefix_cache_;
};

}  // namespace nova::memory
```

### 2.3 Integration with Memory Layer

The `KVCacheAllocator` extends the memory layer with inference-specific allocation patterns:

```cpp
// Extension to cuda/memory/memory_pool.h

namespace cuda::memory {

class MemoryPool {
public:
    // Existing methods...

    // New: Get KV cache allocator (if configured)
    KVCacheAllocator* get_kv_cache() const {
        return kv_cache_.get();
    }

    // New: Set KV cache allocator
    void set_kv_cache(std::unique_ptr<KVCacheAllocator> allocator) {
        kv_cache_ = std::move(allocator);
    }

private:
    std::unique_ptr<KVCacheAllocator> kv_cache_;
};

}  // namespace cuda::memory
```

---

## 3. Paged Attention Block Manager

### 3.1 vLLM Block Manager Design Patterns

The vLLM block manager implements paged attention with physical/physical block mapping:

**Key concepts:**
- **Logical blocks:** Tokens as they appear in the sequence
- **Physical blocks:** Fixed-size GPU memory allocations
- **Block table:** Mapping from logical to physical blocks

**vLLM patterns to adopt:**

| Pattern | vLLM Implementation | Nova Adaptation |
|---------|---------------------|-----------------|
| Block allocation | Lazy allocation on token generation | Batch allocation on sequence start |
| Block eviction | LRU when memory exhausted | Coordinated with KVCacheAllocator |
| Cross-request sharing | Prefix caching via hash | Nova prefix_cache_ map |
| GPU-CPU sync | cudaEvent polling | Stream synchronization |

### 3.2 Paged Attention Integration

```cpp
// include/cuda/inference/block_manager.h (NEW MODULE)
#pragma once

#include "cuda/memory/kv_cache_allocator.h"
#include "cuda/algo/flash_attention.h"
#include <memory>
#include <shared_mutex>
#include <unordered_map>

namespace nova::inference {

class BlockManager {
public:
    struct Config {
        int max_model_len = 8192;
        int block_size = 16;
        int num_cpu_blocks = 2048;
        int num_gpu_blocks = 4096;
        bool enable_cuda_graph = true;
    };

    explicit BlockManager(const Config& config);
    ~BlockManager();

    // Sequence management
    struct Sequence {
        int64_t id;
        int64_t created_at;
        int num_tokens;
        std::vector<int> logical_blocks;  // Block table (logical -> physical)
        
        // For FlashAttention: contiguous views of KV cache
        memory::Buffer k_cache_view;
        memory::Buffer v_cache_view;
    };

    // Create a new sequence
    Sequence* create_sequence(int64_t sequence_id, int max_tokens);

    // Add tokens to sequence (during autoregressive generation)
    void append_tokens(int64_t sequence_id, int num_tokens);

    // Get sequence
    Sequence* get_sequence(int64_t sequence_id);
    const Sequence* get_sequence(int64_t sequence_id) const;

    // Free sequence
    void free_sequence(int64_t sequence_id);

    // Batch forward pass with paged attention
    void forward_batch(
        const std::vector<int64_t>& sequence_ids,
        const memory::Buffer& query,
        memory::Buffer& output,
        const Stream& stream
    );

    // Memory management
    void maybe_evict();
    int get_num_free_blocks() const;

private:
    void allocate_blocks_for_sequence(Sequence* seq, int num_blocks);

    Config config_;
    std::unordered_map<int64_t, std::unique_ptr<Sequence>> sequences_;
    std::shared_mutex sequence_mutex_;
    
    std::unique_ptr<memory::KVCacheAllocator> kv_cache_;
    std::unique_ptr<algo::FlashAttention> attention_;
    
    // Block allocation tracking
    std::vector<int> block_refcount_;  // Reference count per block
    std::vector<int> free_blocks_;
    int num_allocated_blocks_ = 0;
};

// Paged attention kernel interface
class PagedAttention {
public:
    // Compute attention using block tables
    static void forward(
        memory::Buffer& output,
        const memory::Buffer& query,
        const memory::Buffer& key_cache,
        const memory::Buffer& value_cache,
        const std::vector<int>& block_table,  // Logical -> physical mapping
        int num_tokens,
        int num_heads,
        int head_dim,
        int block_size,
        const Stream& stream
    );
};

}  // namespace nova::inference
```

### 3.3 New Inference Module Structure

```
include/cuda/inference/
├── block_manager.h          # Paged attention block manager
├── sequence_manager.h       # Sequence lifecycle management
├── scheduler.h              # Batching and scheduling
├── cache_manager.h          # KV cache coordination
├── paged_attention.h        # Paged attention kernels
└── types.h                  # Common inference types
```

---

## 4. Sequence Parallelism Extension

### 4.1 Sequence Parallelism Patterns

Sequence parallelism distributes sequence dimension across GPUs, complementing tensor parallelism (which splits attention heads).

**Parallelism strategies:**

| Strategy | Dimension | Communication | Use Case |
|----------|-----------|---------------|----------|
| Tensor Parallelism | Head dim | AllReduce per layer | Single-node multi-GPU |
| Pipeline Parallelism | Layer | P2P activation pass | Multi-node scaling |
| Sequence Parallelism | Sequence | AllReduce attention output | Long context |

**Integration with existing distributed/ module:**

```cpp
// include/cuda/distributed/sequence_parallel.h (new file)
#pragma once

#include "cuda/distributed/common.h"
#include "cuda/memory/buffer.h"

namespace nova::distributed {

// Sequence parallelism for attention
class SequenceParallelAttention {
public:
    struct Config {
        int num_model_parallel_gpus = 1;
        int sequence_parallel_size = 1;  // Spans this many GPUs
        bool reduce_scatter_output = true;
    };

    explicit SequenceParallelAttention(const Config& config);

    // All-gather keys/values across sequence dimension
    void gather_kv(
        memory::Buffer& gathered_k,
        memory::Buffer& gathered_v,
        const memory::Buffer& local_k,
        const memory::Buffer& local_v,
        const Communicator& comm
    );

    // Reduce-scatter attention output
    void scatter_output(
        memory::Buffer& local_output,
        const memory::Buffer& full_output,
        const Communicator& comm
    );

    // Combined all-reduce for non-attention layers
    void all_reduce_sequence(
        memory::Buffer& data,
        const Communicator& comm
    );

private:
    Config config_;
};

// Ring sequence parallelism (for very long sequences)
class RingSequenceParallelism {
public:
    struct Config {
        int num_gpus;
        int ring_size;
    };

    explicit RingSequenceParallelism(const Config& config);

    // Ring attention: pass KV around the ring
    void ring_attention(
        memory::Buffer& query,
        memory::Buffer& key,
        memory::Buffer& value,
        memory::Buffer& output,
        const Communicator& comm
    );

private:
    Config config_;
};

}  // namespace nova::distributed
```

### 4.2 Integration with Existing Distributed Module

The sequence parallelism extension adds to existing TP/PP patterns:

```cpp
// include/cuda/distributed/matmul.h (existing) - extension
namespace nova::distributed {

class TensorParallelMatmul {
public:
    // Existing methods...

    // New: Check if sequence parallelism is enabled
    bool has_sequence_parallelism() const {
        return seq_parallel_size_ > 1;
    }

    // New: Get sequence parallel communicator
    Communicator get_sequence_comm() const {
        return seq_comm_;
    }

private:
    int seq_parallel_size_ = 1;
    Communicator seq_comm_;
};

}  // namespace nova::distributed
```

---

## 5. Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Inference Session                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐   │
│  │  Scheduler  │────▶│  SequenceManager │────▶│     BlockManager        │   │
│  │ (Batching)  │     │ (Lifecycle)      │     │ (Paged Attention)       │   │
│  └─────────────┘     └──────────────────┘     └───────────┬─────────────┘   │
│          │                    │                            │                 │
│          ▼                    ▼                            ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      KVCacheAllocator                                │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │    │
│  │  │Free List    │  │Sequence Map│  │ Prefix Cache (hash → block) │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
           ┌────────────────┐ ┌────────────────┐ ┌─────────────────┐
           │ FlashAttention │ │  PagedAttn     │ │ SequenceParallel│
           │    (Algo)      │ │  Kernel        │ │  (Distributed)  │
           └────────────────┘ └────────────────┘ └─────────────────┘
```

---

## 6. Memory Layout for Attention

### 6.1 KV Cache Memory Layout (Paged)

```
Physical Block N (16 tokens, 32 heads, 128 head_dim, FP16)
┌──────────────────────────────────────────────────────────────────────┐
│ [Token 0] K: [H0][H1][H2]...[H31]  V: [H0][H1][H2]...[H31]          │
│ [Token 1] K: [H0][H1][H2]...[H31]  V: [H0][H1][H2]...[H31]          │
│ ...                                                                 │
│ [Token 15] K: [H0][H1][H2]...[H31] V: [H0][H1][H2]...[H31]          │
└──────────────────────────────────────────────────────────────────────┘

Block Table (per sequence)
┌─────────────────────────────────────────────────────────────────────┐
│ Logical: [0] [1] [2] [3] ... [N-1] (token indices)                  │
│ Physical: [3] [7] [2] [15] ... [1] (block indices, may be sparse)   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 FlashAttention Tile Layout

```
FlashAttention Forward Pass (tile size 64)

            K dimension (seq_len)
            ├─────────────────────────┤
     ┌──────┴──────┬──────┬──────┐
     │   Tile 0    │  T1  │  T2  │ ...
 Q   │  Q·K^T /√d  │      │      │
     ├──────┬──────┴──────┴──────┤
     │  S   │ Softmax online     │
     ├──────┴────────────────────┤
     │  P·V → O                  │
     └───────────────────────────┘
```

---

## 7. Anti-Patterns to Avoid

### Anti-Pattern 1: KV Cache Fragmentation

**What:** Allocating individual tokens causes memory fragmentation.

**Why bad:** Memory exhaustion before GPU memory is actually full.

**Instead:** Allocate in power-of-2 block sizes (16, 32, 64 tokens).

### Anti-Pattern 2: Synchronous KV Cache Access

**What:** CPU waits for GPU KV cache updates during decoding.

**Why bad:** Decoding throughput limited by CPU-GPU synchronization.

**Instead:** Double-buffer KV cache, overlap token generation with attention.

### Anti-Pattern 3: Per-Sequence KV Cache Allocation

**What:** Allocate/deallocate KV cache per batch item.

**Why bad:** Allocation overhead dominates for small batches.

**Instead:** Pre-allocate block pool, assign blocks on demand.

### Anti-Pattern 4: Ignoring Prefix Caching

**What:** Not sharing KV cache for common prefixes (system prompts).

**Why bad:** Repeated computation for identical prefixes.

**Instead:** Hash prefixes, reuse cached KV blocks across sequences.

---

## 8. Scalability Considerations

| Scale | KV Cache Strategy | Attention Strategy | Distributed Strategy |
|-------|-------------------|-------------------|---------------------|
| 1 GPU, short seq | Contiguous allocation | FlashAttention v2 | None |
| 1 GPU, long seq | Paged attention | FlashAttention + splits | Sequence parallel |
| Multi-GPU, single node | Paged attention | Tensor + sequence parallel | TP + SP |
| Multi-node | Paged + prefix cache | Pipeline parallel | TP + PP + SP |

### 100 Users / 1K Tokens

- KV cache: ~2GB per GPU
- Attention: FlashAttention v2 sufficient
- Batching: 32-64 sequences per batch

### 10K Users / 8K Tokens

- KV cache: ~16GB per GPU
- Attention: FlashAttention with num_splits > 1
- Batching: 8-16 sequences per batch (memory constrained)

### 1M Users / 32K Tokens

- KV cache: Prefix caching essential
- Attention: Ring attention or sequence parallel
- Distributed: Multi-node TP + PP required

---

## 9. Implementation Order

### Phase 1: Memory Infrastructure (Week 1-2)
1. Create `cuda/memory/kv_cache_allocator.h`
2. Implement block allocation and free list
3. Add prefix caching infrastructure
4. Write memory tests

### Phase 2: FlashAttention Integration (Week 2-3)
1. Create `cuda/algo/flash_attention.h`
2. Integrate with existing TransformerLayer
3. Add backward pass for training
4. Benchmark against standard attention

### Phase 3: Block Manager (Week 3-4)
1. Create `cuda/inference/block_manager.h`
2. Implement SequenceManager
3. Add paged attention kernel
4. Integrate with KVCacheAllocator

### Phase 4: Distributed Extension (Week 4-5)
1. Create `cuda/distributed/sequence_parallel.h`
2. Implement ring sequence parallelism
3. Integrate with existing TP/PP patterns
4. Multi-GPU benchmarking

### Phase 5: Integration and Optimization (Week 5-6)
1. CUDA Graph capture for attention
2. NVTX annotations per inference stage
3. End-to-end throughput benchmarks
4. Production readiness review

---

## Sources

- [FlashAttention Paper (v2)](https://arxiv.org/abs/2205.14135) - **HIGH confidence**
- [FlashAttention v3 Paper](https://arxiv.org/abs/2404.05117) - **HIGH confidence**
- [vLLM Architecture](https://github.com/vllm-project/vllm) - **HIGH confidence** (open source reference)
- [Ring Attention Paper](https://arxiv.org/abs/2310.02589) - **HIGH confidence**
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) - **HIGH confidence**
- [CUDA Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - **HIGH confidence**

---

*Architecture research for: Nova CUDA Library Transformer & Inference Optimization*
*Researched: 2026-04-29*
