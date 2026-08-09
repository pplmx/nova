# RIL — Round 10 (2026-08-10) — sparse solvers + latent device-memory sweep

## Fixed and verified
- GMRES::solve rewritten (commit d11276b): the Arnoldi process was broken
  (H[j][j] overwritten inside the inner-product loop; a nonsense H[k][j]
  update formula replacing the true inner product) and the converged solution
  was discarded by an `x[i] = h_x[i]` overwrite -> spurious breakdown, x always
  0. Standard GMRES(m) with reorthogonalization, Givens rotations, back-sub and
  restart accumulation. PreconditionedSolverTest 0 failures (was intermittent).
  GMRESGPU is a separate, working implementation.
- preconditioned_solver/benchmark tests called spmv(A, host.data(), ...) which
  resolves to the DEVICE-pointer overload -> cusparse illegal memory access;
  switched to the host-vector overload. This sticky error also poisoned later
  tests in the suite.

## Fixed (latent, compile-verified but env-unrunnable)
- linalg.cu cholesky_decomposition: host-indexed device Buffer to zero the
  upper triangle (SIGSEGV on use); host staging copy now.
- kv_cache_allocator.h prefill_chunk: cudaMemcpyAsync kind HostToDevice while
  sourcing chunk.embedding (device Buffer<float>) -> DeviceToDevice.
- test_nccl_collectives.cpp: seven tests wrote/read device Buffer data via raw
  host pointers (latent SIGSEGV when NCCL present); host staging + copies.

## Environment note
Shared 8x A100 (~4.4GB free/device when free). ChunkedPrefill / inference KV
tests still OOM here (allocate 4-64GB); pass only when neighbors free memory.
Full run observed completing with 1271 tests passing; residual failures are
environmental OOM + a few load-flaky cuBLAS tests (verify in isolation) + the
deferred ScopedGraphBuffer pool design.
