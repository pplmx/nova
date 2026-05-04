# Phase 105: Speculative Decoding - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can accelerate inference 2-4x using speculative decoding with proper verification. Implements draft model speculation, tree attention kernels, rejection sampling with correct probability comparison, KV isolation via snapshot/rollback, and optional EAGLE3/SnapKV tree-based decoding.
</domain>

<decisions>
## Implementation Decisions

### Draft Generation
- Support self-speculative (same model, temperature) and multi-model (small draft, large target)
- Default draft depth: 4 tokens
- Use async CUDA streams for draft generation

### Verification
- Tree attention for parallel verification of all draft tokens
- Rejection sampling: `acceptance = fminf(1.0f, target_prob / draft_prob)`
- KL divergence verification via LogProbTracker

### KV Isolation
- Snapshot before speculation, rollback on rejection
- Track speculative vs verified tokens with sequence IDs
- No KV contamination between speculation attempts

### Agent's Discretion
- EAGLE3 vs SnapKV selection criteria
- Default acceptance threshold (0.8)
- Max speculation depth (8)

</decisions>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>
