# RIL — Round 15 (2026-08-11) — milestone v2.16 "Distributed Multi-GPU Verification"

## New milestone: v2.16 Distributed Multi-GPU Verification

Opens the next milestone cycle after v2.15 (Test Quality Assurance) closed at
Round 14 with no active task above threshold. Theme: turn the RIL's last
un-verified functionality category — the real NCCL multi-GPU collective path —
from "env-gated and never run" into running, asserted, multi-GPU tests, and
make the NCCL integration correct for multi-rank use.

Scope (task ladder):
- `task-r15-nccl-collectives` — enable + fix the NCCL collective suite (this round).
- `task-r15-dist-pool-mgpu` — verify the multi-GPU distributed pool tests run (this round).
- future: `task-dist-matmul-multiprocess` — implement a real multi-process execution
  harness for distributed matmul (currently an unconditional, documented skip).
- future: MPI build-flag enablement (no MPI headers installed on this host; env-bound).

Rationale: the RIL skip-audit campaign (Rounds 13-14) established that skips hide
real bugs. The 14 NCCL collective tests were the largest remaining skip block and,
once actually run, exposed three genuine defects (below).

## Task: task-r15-nccl-collectives — RESOLVED

### Observation (evidence)
`build/bin/nova-tests` compiled with `-DNOVA_NCCL_ENABLED=1` (NCCL is on in this
build despite the `NOVA_ENABLE_NCCL=OFF` default preset). With
`NCCL_TESTS_AVAILABLE=1` and 2 GPUs, `NcclCollectivesTest.AllReduceSum` FAILED:
`result.ok()=false`, `invalid argument (run with NCCL_DEBUG=WARN for details)`,
recv all-zero. Evidence was previously impossible because no harness ever set
`NCCL_TESTS_AVAILABLE` — grep confirms only the test file names it. These 14 tests
had never executed.

### Root causes (each evidence-backed)
1. `issue-nccl-mesh-comm-init` — `NcclContext::initialize()` (default config,
   `NcclContextConfig::device_ids` empty → DeviceMesh path) created per-device
   streams but left `communicators_` as nullptrs; `has_nccl()` only checks
   `!communicators_.empty()`, so guards passed and every collective called NCCL
   with a null comm → `invalid argument`. Confirmed: run before fix returned
   error for `get_comm(0)` = nullptr.
2. `issue-nccl-init-deadlock` — `ncclCommInitRank` is itself a per-rank
   collective; the sequential init loop deadlocked at rank 0 (hang at SetUp,
   reproduced by the pure `TypeConversion` test which also blocked in
   `initialize()`).
3. `issue-nccl-comm-routing` — every collective hardcoded `get_comm(0)` ("device
   0's communicator"), only valid when the caller's buffers/stream live on device
   0. Must resolve the rank's own communicator via the current device.
4. `issue-nccl-barrier-host-ptr` — `NcclBarrier::barrier_async()` passed a host
   stack `int dummy` as a device pointer to `ncclAllReduce` → `illegal memory
   access` (crash after the context fix made the barrier reachable).

### Fix (fix(nccl) bfb9e80)
- `NcclContext::initialize()` now routes through a single path that, with NCCL
  enabled, creates one stream + communicator per device. Communicators are
  initialized one thread per rank (ncclCommInitRank is collective). Non-NCCL
  builds unchanged (streams only; guarded paths).
- New `NcclContext::current_comm()` / `NcclCollective::current_comm()` resolve the
  communicator for the currently-active device; all collectives (all-reduce,
  all-gather, reduce-scatter, broadcast, barrier, group handle) route through it.
  `barrier_async(int device, ...)` keeps the explicit-device form.
- `NcclBarrier` now uses a device-resident `cuda::memory::Buffer<int>`.
- Tests rewritten as genuine per-rank multi-GPU tests: one thread per device,
  rank-distinct seeds, cross-rank result assertions (e.g. all-reduce sum over
  ranks, all-gather concatenation, reduce-scatter per-rank chunk sums, broadcast
  root→all). `run_per_rank` uses a start barrier (NCCL ops complete only once all
  ranks post) and rethrows the first thread exception once, never from inside
  join().

### Verification
- `CUDA_VISIBLE_DEVICES=0,1 NCCL_TESTS_AVAILABLE=1 NcclCollectivesTest.*` →
  **14/14 pass** (twice, stability). AllReduceInPlace sum factor
  `ndev(ndev+1)/2` verified in-kernel semantics on 2 ranks.
- `DistributedMemoryPoolTest.{AutoAllocateSelectsDevice,DeallocateFromCorrectDevice}`
  (previously skipped on 1 GPU) pass on 2 GPUs.
- Standard single-GPU suite unaffected (69/69 distributed/mesh/tensor-parallel
  subset on GPU 2).
- Full-suite regression run (CUDA_VISIBLE_DEVICES=2): **1449 ran / 1423 pass /
  0 fail / 26 skip / 1 disabled, EXIT=0** — same test surface as the Round-14
  baseline (1420 pass), no regressions; remaining 26 skips are env-bound or
  documented (14 NCCL need `NCCL_TESTS_AVAILABLE` + 2 GPUs, 8 MPI build-flag
  off, 2 distributed pool need 2 GPUs, 1 PeerCopy needs 2 GPUs, 1 matmul
  multi-process design limitation).

### Code review pass (cpp-reviewer on bfb9e80 → fix 6c84803)
The review verified the concurrency design (per-thread cudaSetDevice, disjoint
comm-index writes, exceptions re-thrown only after all joins, barrier/group
buffer lifetimes) and found one HIGH + two MEDIUM + three LOW, all fixed:
- HIGH: `barrier_async(int device, ...)` still passed a host-stack `int` to
  `ncclAllReduce` — the same illegal-memory-access bug; now uses a device
  buffer (both overloads) and `ncclInt32`.
- MEDIUM: collectives' "returns NcclResult, never throws" contract was violated
  by `current_comm()` throwing when the calling thread's device is not in the
  group — reachable from `TensorParallelMatmul` and from `NcclGroupHandle`'s
  destructor during unwinding (a throw there would `std::terminate`). All five
  collectives + group handle now convert the throw into an error `NcclResult`.
- MEDIUM: `create_streams_and_comms()` could leak partial streams/comms and
  stall the surviving ranks on a failed rank's init; the join-complete failure
  path now cleans up partial resources before rethrowing.
- LOW (tests): `run_per_rank` always signals the start barrier so a rank that
  fails before reaching it cannot hang the others; `SafeNcclCallDetectsErrors`
  passes rank 0's comm explicitly.
Re-verified after hardening: NCCL multi-GPU 14/14; full suite 1449/1423/0,
EXIT=0.

### Known skips remaining (verified reasons)
- 14 NCCL tests: skip without `NCCL_TESTS_AVAILABLE` (now real when set).
- 8 MPI tests: build-flag off; no MPI headers on host.
- `DistributedMatmulTest.MultiGpu_RequiresMultiProcess`: unconditional but
  documented design limitation (multi-process execution model) — decision
  `decision-dist-matmul-multiprocess`, candidate follow-up task.
- 2 distributed pool tests: skip on <2 GPUs (env-bound), verified on 2 GPUs.

## Graph delta
nodes:
  - id: task-r15-nccl-collectives
    type: task
    status: resolved
    confidence: 0.95
  - id: issue-nccl-mesh-comm-init
    type: issue
    status: resolved
    confidence: 0.97
  - id: issue-nccl-init-deadlock
    type: issue
    status: resolved
    confidence: 0.97
  - id: issue-nccl-comm-routing
    type: issue
    status: resolved
    confidence: 0.95
  - id: issue-nccl-barrier-host-ptr
    type: issue
    status: resolved
    confidence: 0.97
  - id: change-r15-nccl-context
    type: change
    status: active
  - id: change-r15-nccl-tests
    type: change
    status: active
  - id: decision-nccl-current-device-routing
    type: decision
    status: active
  - id: issue-dist-matmul-multiprocess
    type: issue
    status: stale
    confidence: 0.6
edges:
  - from: change-r15-nccl-context
    type: resolves
    to: issue-nccl-mesh-comm-init
  - from: change-r15-nccl-context
    type: resolves
    to: issue-nccl-init-deadlock
  - from: change-r15-nccl-context
    type: resolves
    to: issue-nccl-comm-routing
  - from: change-r15-nccl-context
    type: resolves
    to: issue-nccl-barrier-host-ptr
  - from: change-r15-nccl-tests
    type: validates
    to: issue-nccl-mesh-comm-init
  - from: decision-nccl-current-device-routing
    type: supersedes
    to: decision-get-comm-0
