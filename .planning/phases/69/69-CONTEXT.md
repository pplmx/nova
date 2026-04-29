# Phase 69: FlashAttention Integration - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement FlashAttention-2 kernel integration with attention backend selection, stable softmax normalization, and backward pass support for training. This phase establishes the foundation for paged attention and sequence parallelism by providing an optimized attention implementation that reduces memory from O(N²d) to O(Nd).

</domain>

<decisions>
## Implementation Decisions

### Attention Backend Selection
- Use enum-based backend selection: Standard/FlashAttention/PagedAttention
- Default to FlashAttention when hardware supports it
- Fallback to standard attention for unsupported configurations

### FlashAttention Integration
- Use CUB for warp-level reductions (already in codebase)
- Implement stable softmax with max subtraction to prevent overflow
- Support both FP16 and BF16 datatypes
- Dynamic workspace allocation based on query shape

### Backward Pass
- Implement deterministic dropout for training compatibility
- Use cuRAND for random number generation with seed propagation
- Gradient computation follows standard attention backward pattern

### the agent's Discretion
- Block size selection: 64x64 or 128x64 based on compute capability
- Tile size configuration per GPU architecture
- Kernel launch parameters optimized for current CUDA best practices

</decisions>

<codebase>
## Existing Code Insights

### Reusable Assets
- cuda::neural::MultiHeadAttention - existing attention implementation
- cuda::neural::softmax_stable() - stable softmax already exists
- cuda::memory::Buffer<T> - RAII buffer wrapper
- cuda::stream::Stream - stream management
- CUDA_CHECK macro - error handling pattern

### Established Patterns
- Header-only public API with implementation in detail/ or src/
- CUDA_CHECK for all CUDA API calls
- RAII resource management
- Config structs with sensible defaults
- Enum-based variant selection

### Integration Points
- Replace cuda::neural::MultiHeadAttention internals
- Add new cuda::algo::FlashAttention namespace
- Extend existing attention config with backend enum

</codebase>

<specifics>
## Specific Ideas

- Attention backend enum in transformer/attention.h
- FlashAttention class in algo/flash_attention.h
- Stable softmax kernel with online normalization
- Dynamic workspace allocation function
- Backward pass with dropout state propagation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
