---
gsd_state_version: 1.0
milestone: v1.5
milestone_name: Fault Tolerance
status: planning
last_updated: "2026-04-26"
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

**Project:** Nova CUDA Library Enhancement
**Last Updated:** 2026-04-26 (v1.5 planning)

## Current Position

| Field | Value |
|-------|-------|
| **Milestone** | v1.5 Fault Tolerance |
| **Overall Progress** | 0% (0/4 phases, 0/12 plans) |
| **Total Requirements** | 20 |
| **Status** | **PLANNING - Phase 21 not started** |

## Phase Progress

| Phase | Status | Requirements | Plans |
|-------|--------|--------------|-------|
| 21: Checkpoint/Restart | 🚧 Planning | CKPT-01 to CKPT-05 | 0/3 |
| 22: Comm Error Recovery | Pending | COMM-01 to COMM-05 | 0/3 |
| 23: Memory Error Detection | Pending | MEM-01 to MEM-05 | 0/3 |
| 24: Job Preemption | Pending | PEMP-01 to PEMP-05 | 0/3 |

## v1.5 Summary

Milestone v1.5 adds production-grade fault tolerance for cluster deployments.

**Goals:**
- GPU checkpoint/restart with full state serialization (weights + optimizer + RNG)
- Communication error recovery for NCCL/TCP failures
- Memory error detection and ECC error handling
- Job preemption signal handling for scheduler integration

**Requirements:** 20 total (CKPT-01 to CKPT-05, COMM-01 to COMM-05, MEM-01 to MEM-05, PEMP-01 to PEMP-05)

## Milestone History

| Milestone | Status | Date | Requirements |
|-----------|--------|------|--------------|
| v1.0 Production Release | ✅ Shipped | 2026-04-24 | 58 |
| v1.1 Multi-GPU Support | ✅ Shipped | 2026-04-24 | 13 |
| v1.2 Toolchain Upgrade | ✅ Shipped | 2026-04-24 | 9 |
| v1.3 NCCL Integration | ✅ Shipped | 2026-04-24 | 26 |
| v1.4 Multi-Node Support | ✅ Shipped | 2026-04-24 | 15 |
| v1.5 Fault Tolerance | 🚧 Planning | 2026-04-26 | 20 |

## Previous Decisions (v1.4)

| Decision | Implementation |
|----------|----------------|
| D-01: MPI optional with graceful fallback | Single-node works without MPI |
| D-02: RDMA-aware algorithm selection | Prefer CollNet for InfiniBand |
| D-03: Hierarchical collectives | Node-local then cross-node reduction |

## v1.5 Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Checkpoint granularity | Full state for complete recovery | v1.5 planning |
| Error recovery strategy | Detect → isolate → recover → retry | v1.5 planning |
| Signal handling | SIGTERM/SIGUSR1 for graceful shutdown | v1.5 planning |

---
*State updated: 2026-04-26 after v1.5 milestone started*
*20 requirements: CKPT-01 to CKPT-05, COMM-01 to COMM-05, MEM-01 to MEM-05, PEMP-01 to PEMP-05*
