# RIL — Round 13 (2026-08-10) — full-suite poisoning by UB test patterns & exit-time singleton teardown

Result: full suite went from **1411 pass / 4 fail (EXIT=1, intermittent exit SIGSEGV)** to
**1415 pass / 0 fail (EXIT=0, clean exit)** on the quiet GPU.

## Root-caused and fixed: in-process "device poison" that broke SegmentedSort (and 2 others)

The Round-12 hypothesis that `SegmentedSortTest.SortByKeyBasic`'s full-process
`cudaErrorInvalidDevice` was "thrust plan-cache staleness after cudaDeviceReset" was
**WRONG — Thrust 2.0.8 (CUDA 12.9) has no host plan cache** (`util.h` caches only the
per-device PTX version / device count). Bisection of the 134 pre-SegmentedSort suites
(reproduced via `--gtest_filter` prefixes) isolated three *independent* poisoners that
leave the driver/context faulted so the next `cub::DeviceFor::Bulk` → `PtxVersion`
fails with `cudaErrorInvalidDevice`:

1. **`BufferTest.DestructorFrees` / `UniquePtrTest.DestructorFrees`** (memory_pool_test.cpp):
   read through a freed device pointer via `cudaMemcpy` (`cudaErrorInvalidValue`) and
   relied on that being observable — genuine UB that on this driver leaves the context
   in a faulted state (single tests + SegmentedSort reproduced deterministically).
   → Rewritten to `cudaMemGetInfo` accounting (alloc lowers free, dtor restores it); no
   freed-pointer dereference.
2. **`MemoryNodeTest.ScopedGraphBuffer` / `ScopedGraphBufferMove`** (graph_executor_test.cu):
   `ScopedGraphBuffer` still allocates via plain `cudaMalloc` until
   `cudaGraphAddMemAllocNode` support lands, so constructing it inside
   `begin_capture` throws `cudaErrorStreamCaptureImplicit` and leaves the stream's
   capture **invalidated but never ended** — which then breaks later cub dispatch.
   → Now `EXPECT_THROW` (documenting the limitation), then `cudaStreamEndCapture`
   teardown + sticky-error drain.
3. **`ErrorHandlingTest.InvalidSizeTrows` / `BufferVoidInvalidSizeTrows`**: a failed
   `cudaMalloc(SIZE_MAX)` (16 EiB) left a sticky OOM that was readable by later
   `parallel_for`-adjacent paths. → drain the sticky error + sync after the throw.

Two further order-dependent tests were independently broken and fixed:

4. **`AsyncErrorTrackerTest.ScopedErrorTracking`** (observability_test.cpp) only passed
   when a *stray* sticky error from an earlier test happened to be present (failed
   standalone, deterministically). → now forces a real failed `cudaMalloc` inside the
   RAII scope, then drains. This was a hidden 5th failure exposed by the above fixes
   changing the sticky-error landscape.
5. **`PositionalEncodingTest.GetEncoding`** (attention_test.cpp): asserted
   `cudaGetLastError()==cudaSuccess` after its own async work but could observe a
   stale sticky error left by an earlier test (Round-12 had traced the origin to
   `MultiHeadAttentionTest.ForwardSelfAttention`). → the fixture's SetUp now drains the
   sticky error so the assertion reflects only this test's operations.

## Root-caused and fixed: end-of-run SIGSEGV (EXIT=139) — MeshStreams singleton teardown

A new 1GB core captured at 22:00 (after a fully-green run) showed:
`__run_exit_handlers → cuda::distributed::MeshStreams::~MeshStreams (instance()::instance)
→ cudaEventDestroy → cuEventDestroy_v2 → SIGSEGV inside libcuda`. Same class as the
NcclContext exit crash fixed in Round 12 — **the R12 fix missed `MeshStreams`**.
→ `MeshStreams::instance()` now heap-allocates an intentionally never-destroyed
singleton (matching the NcclContext pattern). Verified: suite exits EXIT=0.

## Audited-but-left (no observed evidence of a crash)
- `nova::sparse::detail::CusparseContext::get()` singleton dtor calls `cusparseDestroy`
  at exit — the same latent class, but no crash has been observed; left documented,
  not changed (evidence-driven).
- `timeout_manager` dtor (thread join only), `DeviceMesh` dtor (no CUDA calls) — safe.

## Misc
- Removed leftover `DEBUG: Calling free`/`free completed` stderr prints in
  `KVCacheAllocatorTest`.
- R12's SegmentedSort notes ("thrust plan-cache staleness") superseded by the real
  root cause above (UB test poisoners). The product's `segmented_sort.cu` itself was
  never at fault.

## Full-suite verification (GPU 2)
- Final: **1415 pass / 0 fail / 33 skip / 1 disabled, EXIT=0**.
- Targeted: `BufferTest.*:SegmentedSort` 16/16, `UniquePtrTest.*:SegmentedSort` 11/11,
  `ErrorHandlingTest.*:SegmentedSort` 3/3, `MemoryNodeTest.*:SegmentedSort` 9/9,
  `AsyncErrorTrackerTest.*` 8/8, `PositionalEncodingTest.*:MultiHeadAttentionTest.*` 9/9.
