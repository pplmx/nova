# Phase 72: Sequence Manager & Scheduler - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement multi-sequence orchestration with continuous batching and GQA/MQA support. This phase adds the scheduler that manages batched inference across variable-length sequences with iteration-level scheduling, replacing static batching with dynamic batch composition.

</domain>

<decisions>
## Implementation Decisions

### Continuous Batching
- Iteration-level scheduling: recompose batch at each token generation
- Preempt finished sequences, add new requests
- Dynamic batch size per iteration

### Sequence Manager
- Track sequence state (running, waiting, finished)
- Manage sequence lifecycle (create, update, complete)
- Track prefix cache for multi-turn

### GQA/MQA Support
- num_kv_heads < num_q_heads configuration
- Automatic broadcast of KV to Q heads
- FlashAttention handles the attention computation

### the agent's Discretion
- Batch composition strategy (max batch size vs fill rate)
- Preemption policy (abort or pause)
- Scheduling priority (FIFO vs age-based)

</decisions>

<codebase>
## Existing Code Insights

### Reusable Assets
- cuda::inference::BlockManager - from Phase 71
- cuda::stream::Stream - existing stream management
- Existing attention configs support GQA/MQA

### Established Patterns
- Config structs with validation
- State machines for lifecycle management
- Thread-safe sequence access

### Integration Points
- Scheduler uses BlockManager for memory
- Scheduler calls BlockManager::forward_batch
- Sequence state tracked in SequenceManager

</codebase>

<specifics>
## Specific Ideas

- Scheduler class with batch composition
- SequenceManager for lifecycle management
- Continuous batching loop
- GQA/MQA config passthrough

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
