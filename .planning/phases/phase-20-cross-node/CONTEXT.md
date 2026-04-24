# Phase 20: Cross-Node Communicators

**Phase:** 20 of v1.4 Multi-Node Support
**Started:** 2026-04-24
**Goal:** Implement hierarchical communicators and graceful degradation.

## Context

- Phase 18 completed with MpiContext for rank discovery
- Phase 19 completed with TopologyDetector for network awareness
- Existing NCCL single-node implementation (v1.3)
- Need MultiNodeContext for cluster-scale communicator management

## Requirements

- CNOD-01: MultiNodeContext extending NcclContext for cluster scale
- CNOD-02: Intra-node NCCL communicator (per-node GPU groups)
- CNOD-03: Inter-node NCCL communicator (cross-node GPU groups)
- CNOD-04: Hierarchical collectives (node-local then cross-node)
- CNOD-05: Graceful degradation when MPI/NCCL-NET unavailable

## Success Criteria

1. MultiNodeContext manages both local and global communicators
2. Intra-node comm uses existing single-node NcclContext
3. Inter-node comm connects GPUs across nodes via NCCL NET
4. AllReduce uses node-local reduction then cross-node reduction
5. Fallback mode works without MPI (single-node or P2P)

## Pitfalls

- Communicator deadlock from improper group nesting
- Memory pressure from too many communicators
- Missing NCCL-NET driver causing silent fallback
