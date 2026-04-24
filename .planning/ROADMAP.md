# Roadmap: Nova CUDA Library Enhancement

**Created:** 2026-04-23
**Updated:** 2026-04-24
**Granularity:** Standard

## Milestones

- ✅ **v1.0 Production Release** — Phases 1-6 (shipped 2026-04-24)
- ✅ **v1.1 Multi-GPU Support** — Phases 7-10 (shipped 2026-04-24)
- ✅ **v1.2 Toolchain Upgrade** — Phases 11-12 (shipped 2026-04-24)
- 🔄 **v1.3 NCCL Integration, Tensor & Pipeline Parallelism** — Phases 13-17

## Phase Progress

<details>
<summary>✅ v1.0 Production Release (Phases 1-6) — SHIPPED 2026-04-24</summary>

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Performance Foundations | Device-aware kernels, memory metrics, validation, benchmarks | PERF-01 to PERF-06, BMCH-01 to BMCH-04 | ✅ Complete |
| 2 | Async & Streaming | CUDA streams, pinned memory, pool improvements | ASYNC-01 to ASYNC-04, POOL-01 to POOL-04 | ✅ Complete |
| 3 | FFT Module | Fast Fourier Transform implementation | FFT-01 to FFT-04 | ✅ Complete |
| 4 | Ray Tracing Primitives | Intersection tests and BVH helpers | RAY-01 to RAY-04 | ✅ Complete |
| 5 | Graph Algorithms | BFS and PageRank on GPU | GRAPH-01 to GRAPH-04 | ✅ Complete |
| 6 | Neural Net Primitives | Matmul, softmax, ReLU, layer norm | NN-01 to NN-04 | ✅ Complete |

</details>

<details>
<summary>✅ v1.1 Multi-GPU Support (Phases 7-10) — SHIPPED 2026-04-24</summary>

| # | Phase | Goal | Requirements | Requirements |
|---|-------|------|--------------|--------------|
| 7 | Device Mesh Detection | GPU enumeration, peer access matrix, async P2P copy | MGPU-01 to MGPU-04 | ✅ Complete |
| 8 | Multi-GPU Data Parallelism | All-reduce, broadcast, all-gather, barrier sync | MGPU-05 to MGPU-08 | ✅ Complete |
| 9 | Distributed Memory Pool | Per-device pools, auto-allocation, cross-device tracking | MGPU-09 to MGPU-11 | ✅ Complete |
| 10 | Multi-GPU Matmul | Row-wise split matmul, single-GPU fallback | MGPU-12 to MGPU-13 | ✅ Complete |

</details>

<details>
<summary>✅ v1.2 Toolchain Upgrade (Phases 11-12) — SHIPPED 2026-04-24</summary>

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 11 | Toolchain Analysis | Audit current versions, plan upgrade path | TC-01 to TC-03 | ✅ Complete |
| 12 | Toolchain Upgrade | Implement C++23, CUDA 20, CMake 4.0+ upgrades | TC-04 to TC-09 | ✅ Complete |

</details>

---

## v1.3 NCCL Integration, Tensor & Pipeline Parallelism

**Status:** Phase 14 Planning - 1/5 phases

**Goal:** Enable efficient multi-GPU training with NCCL-based collectives, tensor parallelism for large layers, and pipeline parallelism for deep models.

### Phase Overview

| # | Phase | Goal | Requirements | Plans | Status |
|---|-------|------|--------------|-------|--------|
| 13 | NCCL Foundation | Library detection, NcclContext, communicator init, error handling | NCCL-01 to NCCL-05 | 3/3 | ✅ Complete |
| 14 | Core Collectives | AllReduce, Broadcast, Barrier with async stream-based operations | COLL-01 to COLL-05 | 3/3 | 🔄 Planning |
| 15 | Extended Collectives | AllGather, ReduceScatter, group ops, unified fallback | EXTD-01 to EXTD-05 | Pending | 🔄 Pending |
| 16 | Tensor Parallelism | Column/row parallel matmul, transformer layer patterns | TENS-01 to TENS-06 | Pending | 🔄 Pending |
| 17 | Pipeline Parallelism | 1F1B scheduler, P2P primitives, activation buffer management | PIPE-01 to PIPE-06 | Pending | 🔄 Pending |

### Phase Details

<details>
<summary>Phase 13: NCCL Foundation ✅ COMPLETE</summary>

**Goal:** Set up NCCL integration infrastructure with proper error handling.

**Commits:** 098fa79, ed9176d, b80a747

**Files Created:**
- `cmake/FindNCCL.cmake` - Version validation, NCCL 2.25+ required
- `include/cuda/nccl/nccl_types.h` - CUDA to NCCL dtype mapping
- `include/cuda/nccl/nccl_context.h` - NcclContext with DI + singleton
- `include/cuda/nccl/nccl_error.h` - safe_nccl_call() wrapper
- `include/cuda/nccl/nccl_validation.h` - Shared memory & version validation
- `src/cuda/nccl/nccl_context.cpp` - Full NcclContext implementation
- `src/cuda/nccl/nccl_error.cpp` - safe_stream_wait() implementation
- `src/cuda/nccl/nccl_validation.cpp` - Validation implementations
- Updated `CMakeLists.txt` - NOVA_ENABLE_NCCL option, cuda_nccl library

**Requirements:**
- NCCL-01: Library detection and version validation via CMake find module ✅
- NCCL-02: NcclContext with dependency injection pattern and DeviceMesh integration ✅
- NCCL-03: Communicator initialization and lifecycle management per device ✅
- NCCL-04: Shared memory validation (require 512MB+) with clear error messages ✅
- NCCL-05: Async error polling infrastructure with ncclCommGetAsyncError ✅

**Decisions Applied:**
- D-01: DI with singleton fallback (NcclContext::instance())
- D-02: safe_nccl_call() wrapper with automatic polling
- D-03: Optional NCCL with P2P fallback (NCCL_ENABLE option)
- D-04: Per-device singleton caching (get_comm(device))

**Plans:**
- `13-01-PLAN.md` — CMake integration and version validation ✅
- `13-02-PLAN.md` — NcclContext implementation ✅
- `13-03-PLAN.md` — Error handling and validation ✅

</details>

<details>
<summary>Phase 14: Core Collectives</summary>

**Goal:** Implement stream-based NCCL collective operations replacing P2P fallbacks.

**Requirements:**
- COLL-01: Stream-based all-reduce replacing P2P ring-allreduce fallback
- COLL-02: Broadcast wrapper for weight synchronization across devices
- COLL-03: Barrier implementation for explicit synchronization points
- COLL-04: Safe NCCL call wrapper with async error detection
- COLL-05: Stream-ordered collectives passing cudaStream_t to all operations

**Success Criteria:**
1. All-reduce produces identical results to existing P2P ring-allreduce
2. Broadcast correctly distributes weights from rank 0 to all devices
3. Barrier synchronizes all devices in mesh before proceeding
4. Safe wrapper detects and reports NCCL errors without hanging
5. Collectives integrate with existing MeshStreams infrastructure

**Pitfalls Addressed:**
- Cross-collective deadlocks (single-threaded dispatch)
- Timeout hangs (proper async error polling)
- Stream serialization issues (explicit cudaStream_t passing)

**Plans:**
- `14-01-PLAN.md` — NCCL AllReduce implementation ✅
- `14-02-PLAN.md` — NCCL Broadcast and Barrier ✅
- `14-03-PLAN.md` — Integration, tests, and CMakeLists updates ✅

</details>

<details>
<summary>Phase 15: Extended Collectives</summary>

**Goal:** Add advanced collective operations and unified fallback path.

**Requirements:**
- EXTD-01: All-gather for row-parallel activation gathering
- EXTD-02: Reduce-scatter for alternative gradient aggregation
- EXTD-03: Group operations with ncclGroupStart/End batching
- EXTD-04: Unified NCCL/legacy fallback path for deployment flexibility
- EXTD-05: Communicator caching for repeated collective operations

**Success Criteria:**
1. All-gather correctly assembles partitioned activations across devices
2. Reduce-scatter performs gradient aggregation with configurable root
3. Group operations batch multiple collectives for efficiency
4. Unified path seamlessly falls back to P2P when NCCL unavailable
5. Communicator caching reduces initialization overhead

**Pitfalls Addressed:**
- Single-rank collective calls (always use group operations)
- Blocking operations (async with explicit streams)
- Deployment flexibility (unified fallback path)

</details>

<details>
<summary>Phase 16: Tensor Parallelism</summary>

**Goal:** Implement tensor parallelism patterns for transformer layers.

**Requirements:**
- TENS-01: TensorParallelMatmul with column-parallel strategy
- TENS-02: TensorParallelMatmul with row-parallel strategy
- TENS-03: ColumnParallelLayer for QKV projection pattern
- TENS-04: RowParallelLayer for output projection pattern
- TENS-05: Integration with existing DistributedMatmul infrastructure
- TENS-06: Memory-aware TP degree selection with profiling

**Success Criteria:**
1. Column-parallel matmul partitions weights along output dimension
2. Row-parallel matmul partitions weights along input dimension
3. ColumnParallelLayer produces identical results to single-GPU reference
4. RowParallelLayer produces identical results to single-GPU reference
5. Integration works with existing memory pool and device mesh
6. Memory profiling reports working set per TP degree

**Pitfalls Addressed:**
- Memory explosions from replicated optimizer states
- TP degree selection without memory awareness
- Integration with existing DistributedMatmul

</details>

<details>
<summary>Phase 17: Pipeline Parallelism</summary>

**Goal:** Implement pipeline parallelism scheduling for deep model training.

**Requirements:**
- PIPE-01: PipelineScheduler with 1F1B schedule implementation
- PIPE-02: P2P send/recv primitives for inter-stage communication
- PIPE-03: Activation buffer management with ping-pong overlap
- PIPE-04: Communicator splitting via ncclCommSplit for TP+DP
- PIPE-05: Interleaved schedule option for reduced bubble overhead
- PIPE-06: Stage balance validation within 10% compute variance

**Success Criteria:**
1. 1F1B schedule overlaps forward and backward passes
2. P2P primitives transfer activations/gradients between stages
3. Ping-pong buffering hides communication latency
4. Communicator split supports TP+DP parallelism simultaneously
5. Interleaved schedule reduces bubble overhead to <10%
6. Balance validation reports compute variance across stages

**Pitfalls Addressed:**
- Unbalanced stage compute (stage balance validation)
- Bubble overhead (microbatch count tuning M >= 4*K)
- Multi-communicator deadlocks (NCCL 2.26+ launch ordering)

</details>

---

## Backlog

Deferred work from future milestones.

### Phase 999.1: Distributed Batch Normalization (BACKLOG)

**Goal:** Implement cross-GPU batch normalization synchronization
**Source phase:** Future v1.3+ work
**Deferred at:** Pending
**Requirements:** DBN-01, DBN-02, DBN-03

### Phase 999.2: Multi-Node Support (BACKLOG)

**Goal:** Enable NCCL communication across multiple nodes
**Source phase:** Future v1.4+
**Deferred at:** Pending
**Requirements:** MULN-01, MULN-02, MULN-03

---

*Roadmap updated: 2026-04-24 after v1.3 roadmap creation*
