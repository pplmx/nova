# Phase 73: Sequence Parallelism Extension - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement distributed attention computation across tensor parallel ranks for long context support. This phase adds sequence parallelism that distributes the sequence dimension across GPUs, complementing existing tensor parallelism which splits attention heads.

</domain>

<decisions>
## Implementation Decisions

### TP/SP Integration
- Use existing NCCL infrastructure for communication
- Sequence parallel communicator separate from TP communicator
- All-gather KV across sequence dimension

### Ring Attention
- Pass KV around ring for attention computation
- Each rank holds portion of sequence
- Final all-reduce for output

### Fallback Behavior
- Single-GPU: Sequence parallelism disabled gracefully
- No TP: Fallback to single-node attention

### the agent's Discretion
- Communication pattern (ring vs all-gather-all-reduce)
- Buffer sizing strategy
- Overlap computation with communication

</decisions>

<codebase>
## Existing Code Insights

### Reusable Assets
- cuda::distributed - existing TP/PP infrastructure
- cuda::nccl - existing NCCL integration
- cuda::distributed::Communicator - existing communicator

### Established Patterns
- Communicator creation and destruction
- AllReduce patterns for gradient synchronization
- Device mesh singleton

### Integration Points
- SequenceParallelAttention extends distributed module
- Integrates with FlashAttention for local computation
- Uses existing NCCL context

</codebase>

<specifics>
## Specific Ideas

- SequenceParallelAttention class
- RingSequenceParallelism for long sequences
- Communicator setup for sequence dimension
- All-gather and reduce-scatter operations

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
