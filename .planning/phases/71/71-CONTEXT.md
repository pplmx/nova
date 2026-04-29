# Phase 71: Paged Attention Integration - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement paged attention by combining FlashAttention kernels with block-based KV cache management. This phase creates the BlockManager that orchestrates logical-to-physical block mapping, CPU-GPU synchronization, and bounds validation for production inference workloads.

</domain>

<decisions>
## Implementation Decisions

### Block Manager Architecture
- BlockManager owns KVCacheAllocator and FlashAttention
- Sequence struct holds logical block mapping
- Block table stored as vector<int> per sequence

### CPU-GPU Synchronization
- Dedicated sync_stream for block table updates
- cudaStreamSynchronize before kernel launch
- Event-based synchronization for fine-grained control

### Bounds Validation
- Validate block indices before kernel access
- Throw on out-of-bounds access
- Optional relaxed mode for performance

### Memory Layout
- Packed layout: [num_blocks][num_heads][head_dim][block_size_tokens]
- Block table: [num_seqs][max_blocks_per_seq]

### the agent's Discretion
- Block table storage format (CPU-only vs GPU-mirrored)
- Sync granularity (per-sequence vs batch)
- Error handling strategy

</decisions>

<codebase>
## Existing Code Insights

### Reusable Assets
- cuda::algo::FlashAttention - from Phase 69
- cuda::memory::KVCacheAllocator - from Phase 70
- cuda::stream::Stream - existing stream management

### Established Patterns
- Config structs with validation
- Error throwing on invalid inputs
- Stream synchronization patterns

### Integration Points
- BlockManager → uses KVCacheAllocator for allocation
- BlockManager → uses FlashAttention for kernel execution
- Sequences managed via map<int64_t, Sequence>

</codebase>

<specifics>
## Specific Ideas

- BlockManager class with Sequence struct
- create_sequence(), append_tokens(), free_sequence()
- forward_batch() for paged attention computation
- Block table copy with sync_stream synchronization

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
