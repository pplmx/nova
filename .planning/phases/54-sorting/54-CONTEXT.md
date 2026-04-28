# Phase 54: Foundation & Sorting - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement GPU sorting primitives for key-value pairs, top-k selection, and binary search using CUDA warp primitives. These algorithms should integrate with the existing Buffer and MemoryPool patterns in the library.
</domain>

<decisions>
## Implementation Decisions

### Technology
- Use CUB (CUDA Unbound) library for radix sort primitives
- Use warp shuffle instructions for binary search (no shared memory bank conflicts)
- Top-K via segmented sort pattern (CUB DeviceRadixSort with segment flags)

### API Design
- Follow existing algo/ namespace conventions (reduce.h pattern)
- Return sorted data in-place or via output Buffer for flexibility
- Key-value pairs use separate Buffers for keys and values

### Integration
- Reuse cuda::detail::KernelLauncher and calc_grid_* utilities
- Align with existing memory/ buffer patterns (Buffer<T>)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- cuda::detail::KernelLauncher for kernel launches
- cuda::detail::calc_grid_1d() for grid calculation
- cuda::memory::Buffer<T> for GPU memory management
- Existing reduce.h as API template

### Established Patterns
- Header-only declarations, .cu implementations
- Template-based generic algorithms
- CUDA_CHECK for error handling

### Integration Points
- include/cuda/algo/ directory for algorithm headers
- src/algo/ for implementations
- Tests in tests/algo/

</code_context>

<specifics>
## Specific Ideas

- Support ascending and descending sort order
- Top-K should avoid full sort (O(n log k) complexity)
- Binary search should work on arbitrary CUDA array types

</specifics>

<deferred>
## Deferred Ideas

- Multi-GPU distributed sorting (NCCL) — Phase 59+
- Sorting networks for small arrays — future optimization

</deferred>
