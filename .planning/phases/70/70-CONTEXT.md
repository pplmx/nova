# Phase 70: Paged KV Cache Foundation - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement memory-efficient KV cache allocation with block-based management, LRU eviction policies, and prefix caching for multi-turn conversations. This phase provides the memory infrastructure needed by paged attention, replacing variable-length per-sequence allocation with fixed power-of-2 block sizes to eliminate fragmentation.

</domain>

<decisions>
## Implementation Decisions

### Block Size Selection
- Power-of-2 sizes: 16, 32, 64 tokens per block
- Configurable via KVCacheAllocatorConfig
- Default to 16 tokens for fine-grained allocation

### LRU Eviction Policy
- Trigger when free_blocks falls below threshold
- Evict oldest sequence first (LRU tracking)
- Configurable eviction threshold percentage

### Prefix Caching
- Hash-based prefix lookup using xxHash or std::hash
- Cache prefix tokens up to configured max prefix length
- Copy-on-write for shared prefixes across sequences

### Concurrency Safety
- Thread-safe allocation/deallocation with mutex
- Atomic reference counting for block sharing
- Lock-free reads where possible

### the agent's Discretion
- Specific hash function implementation
- Eviction batch size (evict N blocks at once)
- Statistics update frequency (per-allocation vs lazy)

</decisions>

<codebase>
## Existing Code Insights

### Reusable Assets
- cuda::memory::Buffer<T> - RAII buffer wrapper
- cuda::memory::MemoryPool - existing memory management
- CUDA_CHECK macro - error handling
- RAII patterns for resource management

### Established Patterns
- Config structs with sensible defaults
- Stats structs with clear field names
- Thread-safe operations with mutex protection

### Integration Points
- KVCacheAllocator will be used by BlockManager (Phase 71)
- Can optionally use MemoryPool infrastructure
- Follows Buffer<T> memory model

</codebase>

<specifics>
## Specific Ideas

- KVCacheAllocator class with Block struct
- Block::prev/next for sequence-linked list
- Prefix hash map: uint64_t → Block*
- LRU queue with sequence age tracking
- KVCacheStats for memory reporting

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
