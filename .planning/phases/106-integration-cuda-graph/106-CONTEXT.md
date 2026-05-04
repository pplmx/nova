# Phase 106: Integration & CUDA Graph - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can combine beam search with speculative decoding and optimize all features with CUDA Graphs. Implements dynamic block sizing, chunked prefill for long prompts, and persistent KV support for graph capture.
</domain>

<decisions>
## Implementation Decisions

### Beam + Spec Decode Combo
- Use speculative decoding for acceptance, then beam search for final output
- Share KV cache infrastructure between both modes

### Dynamic Block Sizing
- Default sizes: 16, 32, 64 tokens
- Selection based on access patterns and memory pressure

### Chunked Prefill
- Process long prompts (>16K tokens) in chunks
- Streaming prefill with KV cache updates

### Agent's Discretion
- CUDA Graph conditional node patterns
- Persistent KV memory allocation strategy

</deferred>
## Deferred Ideas

None — phase scope complete
