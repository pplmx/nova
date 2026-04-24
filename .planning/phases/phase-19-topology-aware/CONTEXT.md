# Phase 19: Topology-Aware Collectives

**Phase:** 19 of v1.4 Multi-Node Support
**Started:** 2026-04-24
**Goal:** Detect node topology and select optimal NCCL collective algorithms.

## Context

- Phase 18 completed with MpiContext for rank discovery
- Existing NCCL single-node implementation (v1.3)
- Need topology detection for efficient multi-node collectives

## Requirements

- TOPO-01: Node topology detection (intra-node vs inter-node paths)
- TOPO-02: Network interface card (NIC) enumeration and selection
- TOPO-03: Topology-aware NCCL communicator splitting by NIC
- TOPO-04: Bandwidth-aware collective algorithm selection (ring vs tree vs collnet)
- TOPO-05: Topology validation with performance profiling

## Success Criteria

1. TopologyMap reports intra/inter-node bandwidth estimates
2. NIC selection prefers RDMA-capable interfaces
3. NCCL communicators respect topology boundaries
4. Algorithm selection based on NCCL_TUNING env + profiling
5. Validation reports topology mismatches before training

## Pitfalls

- Incorrect topology assumptions causing NCCL timeouts
- Mixing Ethernet and InfiniBand without awareness
- Collective algorithm mismatch for data size
