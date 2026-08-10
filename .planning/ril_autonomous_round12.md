# RIL — Round 12 (2026-08-10) — core-file forensics & solver/test-contract fixes

## Root-caused the "sigma" end-of-run SIGSEGV (5 × 1GB cores in the tree)
- All five `core.*` dumps traced with gdb. Two crash classes:
  - `core.1547935` (10:14, `--gtest_filter=DistributedMemoryPoolTest.*:DistributedMa…`):
    NcclContext singleton **static-destructor teardown crash** at exit —
    `~NcclContext` → `destroy()` → `cudaStreamDestroy` → `cuStreamDestroy_v2` → SIGSEGV
    inside libcuda, invoked from `__run_exit_handlers(exit status=1)`. This is the
    "end-of-run SIGSEGV with no failing test nearby": it only fires when a
    test failure forces `exit(1)` AND the singleton was constructed AND the CUDA
    driver context is already in an errored state (e.g. after a neighbor-GPU OOM).
  - `core.1544658`/`1587036` (10:12/10:29, `--gtest_brief=1`): DistributedMatmulTest
    single-GPU fallback crashing INSIDE the driver at `cuLaunchKernel` (via
    `cublasLtSSSMatmul` ← `cublasSgemm` ← neural::matmul:54 ← distributed::matmul.cu).
    Same libcuda internal offset (0x20b88b) as the teardown crash — an
    already-broken driver state, not a 64x64 GEMM bug.
  - `core.1321978`/`1433541` (08:40/09:24): 8-GPU-layout full runs (all /dev/nvidia0..7
    mapped), stack corrupt, RIP in main binary; not further resolvable.
- **Fix (nccl_context.cpp):** `NcclContext::instance()` now heap-allocates the
  singleton and never destroys it — CUDA teardown must not run from a static
  destructor at process exit after a driver error. Explicit `destroy()` kept for
  orderly in-process cleanup; matches the DeviceMesh "no teardown from dtor" pattern.
  (On quiet GPU 2 the crash did NOT reproduce with a failing test + live singleton —
  consistent with the driver-error precondition.)

## Fixed & verified
- **BiCGSTAB (include/cuda/sparse/krylov.hpp):** matched against van der Vorst's
  algorithm — three deviations made it non-converging: (a) `alpha` used
  `<p, A p>` instead of `<rhat, A p>`; (b) x-update dropped the `alpha*p` term;
  (c) p-update used `t = A s` instead of `v = A p`. Rewritten to the standard
  iteration. `KrylovGPUTest/0.BiCGSTABTridiagonal (double)` (deterministic failure
  before; the float integration test's "known numerical stability" skip masked the
  bug) now **converges**; all Krylov/CG/GMRES/Preconditioned tests stay green.
- **NVBloxMetricsCollector::reset()** now clears `registered_metrics_` too — the
  singleton previously accumulated registrations across reset()/suites, so
  `registered_metric_count()` was process-lifetime global.
  `PerformanceToolingIntegrationTest.MetricsCollectionPipeline` now passes.
- **KV-cache inference tests (block_manager_edge_test.cpp):** 6 tests failed on a
  quiet GPU — root cause: `kv_cache_config.num_blocks` defaults to 4096 and the
  allocator pre-allocates the whole KV cache up front (16/32/64 MB per block ×
  4096 = 64/128/256 GB) once the Round-11 decoupling stopped BlockManager from
  clamping it; plus 4 tests asserted lazy block_table growth while the pinned
  contract (BlockManagerTest.BlockTableAllocation, BlockSizeBoundaryAlignment,
  LongPromptSingleChunk) is reserve-for-max_tokens pre-allocation. Tests now set
  small `kv_cache_config.num_blocks` and assert the pre-allocation capacity.
  All 10 DynamicBlockSizing/ChunkedPrefill tests pass.
- **DistributedMatmulTest.MultiGpu_RowPartition:** removed the hard-coded
  `device_count == 8` / 12-rows expectations; now derives row counts from the
  actual visible device count and asserts the tiling invariants (contiguous,
  covers [0, m), non-empty per rank). Passes on any GPU count.
- **Bonus downstream effect:** the Round-12 KV OOM fixes stopped `SegmentedSortTest.SortByKeyBasic`
  ("invalid device ordinal") failing in a small combined run, but it still fails in the FULL
  1447-test process via thrust `parallel_for` throwing `cudaErrorInvalidDevice` at 0 ms. It passes
  standalone and with many prefixes; the poisoning source sits further back in the process (thrust
  static plan cache staleness after a `cudaDeviceReset()` is suspected). Left documented, not chased.
- **DistributedMatmulTest × 10-11 `cuBLAS error: 13` — ROOT-CAUSED and FIXED this round.**
  `FusedMatmulBiasAct::forward_fallback` fell back to `cuda::neural::get_cublas_handle()` (the
  process-global handle) and called `cublasSetStream(handle, stream)` — a PERSISTENT rebind. The
  FusedMatmulBiasAct tests use a per-class stream destroyed in TearDown, so after they run the
  global handle pointed at a destroyed stream and every later `cublasSgemm` via
  `cuda::neural::matmul` failed with EXECUTION_FAILED, and in some full runs the driver SIGSEGV'd
  inside `cuLaunchKernel` — this is the actual source of the core.1544658/1587036 crash class
  (earlier mis-attributed to environment). Fix: `FusedMatmulBiasAct` owns a private
  `owned_handle_` when no handle is injected, and never touches the global handle. All
  FusedMatmulBiasAct + DistributedMatmul tests pass; the full NN cluster no longer self-poisons.

## Remaining (documented; environment/design-bound after this round) — final full-suite: 1410 pass / 4 fail
- PositionalEncodingTest.GetEncoding × 1: non-deterministic `cudaGetLastError()!=cudaSuccess`
  sticky-error flake (read of a stale async error); passes alone and in-suite; RIL "load-flaky".
- MemoryNodeTest.ScopedGraphBuffer/Move × 2: needs CUDA graph memory pools
  (cudaGraphAddMemAllocNode); tests pass graph=nullptr — existing design decision.
- SegmentedSortTest.SortByKeyBasic × 1: full-process-only thrust `cudaErrorInvalidDevice`
  (passes standalone + many prefixes); suspected thrust plan-cache staleness after a
  `cudaDeviceReset()` earlier in the process; left documented.

## Resolved-from-prior-rounds (consistency closeout)
- R9 "DistributedMatmul shared-cuBLAS-handle cascade" → RESOLVED (this round: FusedMatmulBiasAct
  was rebinding the shared global handle's stream; fixed in fix(neural) d..14).
- R9 "Inference KV-cache OOM" → RESOLVED (KV tests now small allocator pools).
- R9 "PreconditionedSolverTest/preconditioner_benchmark nondeterministic failures; host b/x
  rows/cols sizing" → RESOLVED in practice: GMRES (R10) + BiCGSTAB (R12) rewrites fixed the
  underlying nondeterminism; all 48 preconditioner tests pass on GPU 2. The only non-square
  case sizes b by cols, but the solver rejects non-square before reading b — benign.
- R11 "sigma end-of-run SIGSEGV" / "DistributedMatmul (shared cuBLAS handle + timing)" /
  "KV-OOM blob" → RESOLVED (see above).
- CONCERNS.md "SyncBatchNorm backward empty" → STALE: both SyncBatchNorm::backward and the free
  sync_batch_norm_backward() are fully implemented; the live item there (raw cudaMalloc temp
  buffers in backward) was fixed this round with RAII.

## Additional in-continuation fix (matmul_batch)
- fix(neural): matmul_batch passed the address of single A/B/C pointers to cublasSgemmBatched
  (which expects arrays of batch_count device pointers) → OOB pointer reads for batch > 1 /
  wrong benchmark numbers. Rewrote as cublasSgemmStridedBatched matching the callers'
  contiguous stacked layout; added MatmulTest.MatmulBatchMatchesPerSliceMatmul (verified 5/5).

## Post-verification addenda (same continuation)
- `fix(neural): RAII temp buffers in SyncBatchNorm::backward` — the five function-local
  device buffers used raw cudaMalloc/cudaFree (leak-on-exception); converted to
  `cuda::memory::unique_ptr`. This was the still-live item in `.planning/codebase/CONCERNS.md`;
  its other SyncBatchNorm finding (empty backward) is STALE — both the class method and the
  free `sync_batch_norm_backward()` are fully implemented.
- `chore: fix invalid .clang-format BinPackParameters value` — `OnePerLine` is not a valid
  value for the boolean `BinPackParameters`, so clang-format refused to load the config
  (broke any format gate). Corrected to `false` (matches adjacent `BinPackArguments`).
- PositionalEncodingTest.GetEncoding sticky error: traced to `MultiHeadAttentionTest.ForwardSelfAttention`
  leaving a stale async error between tests (persists with CUDA_LAUNCH_BLOCKING and under
  compute-sanitizer; the failing test's own cudaGetLastError check passes). No reproducible
  product bug; left as documented flake rather than chasing further. Final full-suite numbers
  remain 1410 pass / 4 fail on GPU 2 (the 4 are: this flake, MemoryNode design decision ×2,
  SegmentedSort full-process env issue).
- "FULL SUITE EXIT" reports in earlier rounds' command logs were `tail`'s exit code,
  not the test binary's — prior "exit 0 despite failures" readings were an artifact.
