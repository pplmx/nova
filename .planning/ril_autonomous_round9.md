# RIL — Round 9 (2026-08-10) — GPU-enabled bug sweep

Environment now verified GPU-capable (8x A100, ~4-8GB free depending on neighbors).
This round fixed a long tail of genuine bugs that were previously invisible because the
suite crashed early or because prior rounds ran GPU-less.

## Library bugs fixed (with evidence)
- F1 `FusedMatmulBiasAct` ctor: cudaGetDevice(nullptr) returned cudaErrorInvalidValue and
  left a STICKY last-error that poisoned every subsequent cudaGetLastError() check. Fixed
  with valid out-ptr. Verified: isolated GELU test + full suite.
- F9 `stream.h::Stream::query()`/`event.h::Event::query()`: `CUDA_CHECK(err)` expanded to
  `cudaError_t err = err;` (self-init from uninitialized -> UB). Fixed: throw CudaException directly.
- F10 `GraphExecutor::begin_capture()` (no-arg): stack-local Stream destroyed at return left
  `capture_stream_` dangling -> cuStreamEndCapture UAF -> SIGSEGV (root cause of the suite's
  recurring exit-139). Fixed: owned `std::optional<Stream>` member; wired through move/reset.
- F11 `BandwidthTracker::measure_transfer(DeviceToHost)`: d_ptr never allocated ->
  cudaMemcpyAsync(DtoH) from nullptr -> invalid argument. Fixed: allocate d_ptr for both
  HtD and DtH.
- `memory_node.cu`: add_*_allocation fabricated degenerate 3D self-copy memcpy nodes (missing
  `extent`) and graph memcpy nodes from host pointers - both always failed with
  cudaErrorInvalidValue. Fixed: cudaGraphAddEmptyNode anchors.
- `sssp.cu` bellman_ford: host loops/writes over the CALLER's DEVICE distances + thrust::seq
  over device_ptr -> SIGSEGV; also read host CSR arrays correctly (host copies). Fixed: compute
  on a host vector, publish via cudaMemcpyAsync. delta_stepping (RIL Round-3 known-broken,
  single-pass + host-kernel mismatch) now delegates to Bellman-Ford for correct results.
- `spmv.cu` spmv_csc_kernel: accumulated y[col] and ignored row_indices (wrong math, e.g.
  y[2]=21 vs 19). Fixed: scatter to y[row] via atomicAdd + memset y in wrapper.
- `ImageBuffer` (UCHAR3/FLOAT3): parameterized ctor left d_data_ uninitialized for size 0 ->
  garbage pointer. Fixed default member initializer.
- `HierarchicalAllReduce::all_reduce`: with NOVA_NCCL_ENABLED and null comms it did nothing
  (recv stayed 0) and both copy paths used `count` bytes not count*element_size. Fixed
  single-rank identity fallback + nccl_type_size().
- `ErrorInjector::should_inject` did not record counts; ScopedErrorInjection ctor never threw.
  Fixed: count-on-fire + throw CudaException at scope construction (test contract).
- `speculative_decoding.cpp`: verify_draft_tokens/generate_draft_tokens read device Buffer
  memory from host (SIGSEGV) and used config vocab_size (32000) vs actual buffer size (OOB).
  Fixed: host staging copies + operate on min buffer size.
- `BeamHypothesis` aggregate: uninitialized POD fields (garbage parent_beam). Fixed default
  initializers.
- `buffer-inl.h` (Round 7 legacy): cuda_check_impl threw std::runtime_error while goldens
  require CudaException. Aligned to CudaException (source-compatible).

## Test bugs fixed
- RadixSortTest sort count used d_keys_.size()=1024 while data was 10/100 (flaky/wrong).
- attention_test: host vectors passed to DtoD CUDA calls (invalid argument).
- graph_executor capture tests: cudaMalloc/free during capture (not permitted) + blocking
  cudaMemset on default stream; now pre-allocate outside + cudaMemsetAsync on capture stream.
- TotalAllocatedTracking: double end_capture() on one capture.
- IntegrationTest fixture name collision across two TUs (gtest fatal) -> ProfilingIntegrationTest.
- SSSP test expectations contradicted graph data (d[2]=4 not <=3; d[2]=2 not 1; d[1]=1 not 5).
- speculative test bodies wrote device memory from CPU + AcceptanceRatioFullReject data peaked
  target at the same token -> made reject impossible.
- HierarchicalAllReduce/ImageBuffer/SpMV tests: none.

## Deferred / environmental
- MemoryNodeTest.ScopedGraphBuffer/Move (2 tests) need CUDA graph memory pools - design-decision
  (see ril_autonomous_round8.md). Never worked (always crashed pre-F10).
- Inference KV OOM (~74 tests; BeamSearch/StreamingCache/AttentionSink/...) - tests allocate
  multi-GB KV caches (default kv config = 64GB on an 80GB box); this shared box has ~4-8GB
  free so they OOM when neighbors use the GPU and pass when it is free. Confirmed environmental:
  a full run passed all but 1 of them once memory freed up. BlockManager now propagates
  num_gpu_blocks into kv num_blocks (single source of truth), but tests still need >4GiB free.
- DistributedMatmulTest: opts a shared cuBLAS handle whose stream is mutated by per-test
  streams in earlier suites; fails (error 13) only in a full-process run. Hypothesis:
  shared-handle + cublasSetStream(per-test stream) retained across test boundaries. A real
  fix (per-call handles or stream restore) is a design change; tracked.
- PositionalEncodingTest.GetEncoding occasionally fails under GPU-load variance (passes in
  isolation and in most runs).

## Verification
- Build clean (gcc 13 / nvcc 12.9, arch 52, both -Wall -Wextra).
- Full suite: failures dropped from ~23 deterministic + 5 crash sites + 85-run blob to a
  residual that is (a) environmental OOM when neighbors eat the GPU, (b) 2 deferred pools,
  (c) intermittent DistributedMatmul/GetEncoding under load. One run reached only 1 failed test.

## Addendum (later in Round 9)
- `speculative_decoding.cpp` verify_draft_tokens/generate_draft_tokens: host reads of device
  Buffer memory (SIGSEGV) + config-vs-buffer size mismatch (OOB copy). Fixed host staging +
  min-buffer-size loops. Tests also wrote device memory from CPU lambdas; fixed.
- `memory_safety.cpp` check_uninitialized: scanned device memory from host (SIGSEGV). Fixed
  host staging copy; unreadable -> unsafe.
- `preconditioner.hpp` JacobiPreconditioner: setup read SparseMatrix device arrays from host,
  and raw-pointer apply() read device diagonal_ from host. Fixed via copy_to_host staging.
  (ILU already did this correctly - Jacobi was the outlier.)
- `reordering.hpp` RCM: find_starting_node/bfs_level_order/compute_matrix_bandwidth all read
  device CSR arrays from host. Fixed via copy_to_host staging.
- Tests: `preconditioner_test.cpp` create_laplacian_matrix had `row_offsets[i]=0` inside the
  build loop, collapsing every row to an empty range (valid diagonal data discarded). Fixed
  to set row_offsets[0] once. precond_solver: CG/GMRES "SquareMatrixValidation" built a VALID
  diag(1,2) matrix but expected INVALID_MATRIX (wrong expectation -> SUCCESS + solution check);
  BiCGSTAB's builds a 3x2 non-square matrix and correctly expects INVALID_MATRIX (left intact).
- PreconditionerBenchmark/solver flakiness on the shared A100: failures drift run-to-run
  (sticky illegal-access replay at cudaMalloc, "0ms" passes/vs fails). Part real solver bugs
  (e.g. host+b mismatch in some tests), part shared-GPU load. Requires a dedicated-GPU debug
  pass; tracked below.

## Actively tracked follow-ups (not fixed this round)
- Sparse solvers (PreconditionedSolverTest/preconditioner_benchmark_test): nondeterministic
  failures under shared-GPU load; some tests size host b/x by rows/cols incorrectly. Needs
  per-test audit on a quiet GPU. (issue, category 8)
- DistributedMatmul shared-cuBLAS-handle cascade in full-suite runs (hypothesis: per-test
  stream set on the shared g_cublas_handle persists across tests). (issue, category 8)
- Inference KV-cache OOM when neighbors consume GPU memory (environmental; pass when free).
- MemoryNodeTest.ScopedGraphBuffer needs CUDA graph memory pools (design).

- RCMReorderer: three real defects fixed (commit 4b48a03):
  (1) permutation inverted (perm[i]=level_order[i] vs old->new that
  apply_permutation consumes) -> reordered matrix scrambled, bandwidth never
  reduced; (2) apply_permutation wrote into all-zero row]) offsets -> every row
  overwrote at offset 0; (3) disconnected graphs read level_order out of bounds
  (single BFS component). Tests switched to a tridiagonal band-1 path (dense
  band-5 where RCM can't reduce was the wrong premise). RCMReordererTest 14/14.
