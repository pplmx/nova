---
phase: 14-core-collectives
plan_count: 3
completed_plans: 3
requirements:
  - COLL-01: NCCL AllReduce interface
  - COLL-02: NCCL Broadcast interface
  - COLL-03: NCCL Barrier interface
  - COLL-04: Stream ordering for all operations
  - COLL-05: Type conversion utilities
subsystem: cuda-nccl
tags:
  - cuda
  - nccl
  - collective-ops
  - multi-gpu
key_files:
  created:
    - include/cuda/nccl/nccl_collective.h
    - include/cuda/nccl/nccl_all_reduce.h
    - include/cuda/nccl/nccl_broadcast.h
    - include/cuda/nccl/nccl_barrier.h
    - src/cuda/nccl/nccl_collective.cpp
    - src/cuda/nccl/nccl_all_reduce.cpp
    - src/cuda/nccl/nccl_broadcast.cpp
    - src/cuda/nccl/nccl_barrier.cpp
    - tests/cuda/nccl/test_nccl_collectives.cpp
  modified:
    - CMakeLists.txt (NCCL_SOURCES, test includes)
    - tests/CMakeLists.txt (test registration)
    - include/cuda/nccl/nccl_types.h (stub types for non-NCCL builds)
    - include/cuda/nccl/nccl_error.h (type fixes)
    - include/cuda/nccl/nccl_context.h (type fixes)
    - src/cuda/nccl/*.cpp (conditional compilation fixes)
dependency_graph:
  requires: [COLL-01, COLL-02, COLL-03, COLL-04, COLL-05]
  provides:
    - nccl::NcclAllReduce class
    - nccl::NcclBroadcast class
    - nccl::NcclBarrier class
tech_stack:
  patterns:
    - Dependency injection (NcclContext reference in constructors)
    - Safe wrapper pattern (safe_nccl_call with async polling)
    - Stream-ordered operations (cudaStream_t passed to all NCCL calls)
    - Template-based error handling with timeout
  added:
    - NcclCollective base class
    - NcclAllReduce with all_reduce_async/all_reduce
    - NcclBroadcast with broadcast_async/broadcast
    - NcclBarrier with barrier_async/barrier
    - Comprehensive unit tests
decisions:
  - Use NcclContext from Phase 13 for communicator caching
  - safe_nccl_call wrapper with 30s timeout for async error detection
  - Stream-ordered operations (cudaStream_t passed to all NCCL calls)
  - Per-device communicator caching
  - Stub types for non-NCCL builds to enable compilation without NCCL
metrics:
  duration_minutes: ~30
  completed_date: "2026-04-24"
  files_created: 9
  files_modified: 12
  lines_added: ~1200
---

# Phase 14: Core Collectives Summary

Stream-based NCCL AllReduce, Broadcast, and Barrier implementations.

## One-liner

NCCL-based collective operations (AllReduce, Broadcast, Barrier) with stream ordering, async error detection, and P2P fallback.

## What Was Built

### NcclCollective Base Class
Base class providing common infrastructure for NCCL collective operations:
- NcclContext reference for communicator access
- Helper methods: `has_nccl()`, `get_comm(device)`, `get_stream(device)`, `device_count()`
- Integration with safe_nccl_call error handling

### NcclAllReduce
Stream-based all-reduce using NCCL:
- `all_reduce_async()` - async operation with safe_nccl_call wrapper
- `all_reduce()` - blocking wrapper that syncs stream
- `to_nccl_op()` - maps distributed::ReductionOp to ncclRedOp_t
- `to_nccl_dtype()` - maps cudaDataType to ncclDataType_t

### NcclBroadcast
Broadcast from root device to all others:
- `broadcast_async()` - async broadcast with safe_nccl_call
- `broadcast()` - blocking wrapper
- Supports explicit root_rank specification

### NcclBarrier
Multi-device synchronization:
- `barrier_async()` - async barrier on device 0's stream
- `barrier_async(device, stream)` - with device selection
- `barrier()` - blocking wrappers
- Unlike cudaStreamSynchronize, barriers ALL devices in mesh

### Unit Tests
Comprehensive test suite in `tests/cuda/nccl/test_nccl_collectives.cpp`:
- GPU detection with graceful skip on <2 GPUs
- AllReduce sum and in-place tests
- Broadcast from root tests
- Barrier synchronization tests
- Type conversion validation tests
- Tests skip gracefully when NCCL disabled

## Deviations from Plan

### Auto-fixed Issues

**[Rule 2 - Missing Stub Types] Added stub definitions for non-NCCL builds**
- Found during: Plan 14-03 Integration
- Issue: Code using NCCL types failed to compile when NCCL disabled
- Fix: Added stub definitions in nccl_types.h:
  - `ncclComm_t`, `ncclRedOp_t`, `ncclDataType_t`, `ncclResult_t`
  - Stub values for all NCCL types
  - Inline no-op functions for ncclAllReduce, ncclBroadcast, ncclBarrier
- Files modified: include/cuda/nccl/nccl_types.h
- Commit: 7b33bfe

**[Rule 3 - Blocking Issue] #ifdef vs #if for NOVA_NCCL_ENABLED**
- Found during: Build verification
- Issue: When cmake sets NOVA_NCCL_ENABLED=0, #ifdef still evaluated to TRUE
  because the symbol is defined. This caused NCCL code paths to be compiled
  even when disabled.
- Fix: Changed all `#ifdef NOVA_NCCL_ENABLED` to `#if NOVA_NCCL_ENABLED`
  throughout the NCCL module (Phase 13 and Phase 14 files)
- Files modified: All nccl headers and source files
- Commit: 7b33bfe

## Threat Flags

None - all operations use NCCL's internal security model.

## Self-Check

- [x] NcclAllReduce class exists in include/cuda/nccl/nccl_all_reduce.h
- [x] NcclBroadcast class exists in include/cuda/nccl/nccl_broadcast.h
- [x] NcclBarrier class exists in include/cuda/nccl/nccl_barrier.h
- [x] All classes use safe_nccl_call wrapper
- [x] All operations are stream-ordered (cudaStream_t passed to NCCL)
- [x] Build passes with and without NOVA_NCCL_ENABLED
- [x] Unit tests compile and register with nova-tests
- [x] Type conversion utilities implemented

## Known Stubs

None - all stubs are intentional placeholders for when NCCL is disabled.

## Related Commits

- 7b33bfe: feat(14-core-collectives): add NCCL collectives integration and tests
- 729e8a5: feat(14-core-collectives): add NCCL Broadcast and Barrier
- fdea03d: feat(14-core-collectives): add NCCL AllReduce implementation
