# Phase 104: Beam Search Core - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can perform GPU-accelerated beam search with memory-efficient KV management. This builds on the KV Cache Foundation from Phase 103 to implement BeamSearchManager with configurable beam width, length normalization, reference-counted KV sharing, batch operations, and TopK/TopP sampling integration.
</domain>

<decisions>
## Implementation Decisions

### Beam Width
- Support 1-8 beams as per research
- Default to K=4 (good balance of quality vs memory)
- Dynamic beam width adjustment based on available memory

### Length Normalization
- Use length penalty: score / pow(length, alpha)
- Default alpha = 0.7 (from research)
- Score rebase to prevent underflow at long sequences (>2000 tokens)

### KV Sharing
- Fork-on-diverge pattern from Phase 103 prefix sharing
- Reference-counted blocks for prefix reuse
- Batch KV operations across multiple beams

### Agent's Discretion
- Beam scoring tie-breaking strategy
- Sampling temperature defaults
- Early termination criteria

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cuda/memory/KVCacheAllocator` - existing allocator with fork_prefix_blocks
- `cuda/inference/BlockManager` - wraps KVCacheAllocator, manages sequences
- `cuda/memory/StreamingCacheManager` - from Phase 103

### Established Patterns
- RAII with unique_ptr
- Config structs with sensible defaults
- std::vector for beam state management

### Integration Points
- KVCacheAllocator::fork_prefix_blocks for beam KV sharing
- BlockManager for sequence management
- FlashAttention for scoring operations

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Reference vLLM and HuggingFace beam search implementations.
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>
