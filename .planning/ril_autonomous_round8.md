# RIL — Round 8 Session (2026-08-10) — GPU-present environment audit

## Environment correction
Prior rounds recorded "no GPU available / CUDA driver absent". **Outdated**: this env has 8x A100-SXM4-80GB,
cudaGetDeviceCount=8, cudaSetDevice succeeds on all (nvidia-smi driver 595.45.04, CUDA 13.2, ~4GB free/GPU).
Many previously "GPU-gated, cannot verify" tests now RUN and reveal real bugs and test bugs.

## Evidence-backed findings

### F1 [LIBRARY BUG — correctness/stability, category 10]
`src/cuda/neural/fusion/fused_matmul_bias_act.cu:40` — `cudaGetDevice(nullptr)` (NULL int*) returns
cudaErrorInvalidValue(1) and leaves a STICKY last-error. Every FusedMatmulBiasAct construction poisons
CUDA error state → `cudaGetLastError()`==1 after forward; cascades into subsequent tests that check
cudaGetLastError (e.g. MultiHeadAttentionTest). 
- evidence: `/tmp/devnull` probe — cudaGetDevice(nullptr)->1; cudaGetLastError->1 until next successful call.
- evidence: isolated `FusedMatmulBiasActTest.MatmulBiasGELU` fails (cudaGetLastError=1); /tmp/gelu_repro all 3 activations fail.
- fix: pass a valid `int dev;` to cudaGetDevice.

### F2 [LIBRARY CONTRACT BUG — correctness, category 10]
`include/cuda/memory/buffer-inl.h` `cuda_check_impl` throws `std::runtime_error`; goldens (ErrorHandlingTest,
canonical cuda/device/error.h) require `cuda::device::CudaException`. 
- evidence: `ErrorHandlingTest.InvalidSizeTrows`/`BufferVoidInvalidSizeTrows` fail: "Actual: it throws std::runtime_error".
- fix: throw `cuda::device::CudaException` (is-a runtime_error → source-compatible for existing catch sites).

### F3 [TEST BUG — test quality, category 4]
`tests/algo_sort_test.cu` RadixSortTest: SetUp sizes d_keys_=1024, but AscendingSort/AllSame assign
h_keys_ of 10/100 and sort d_keys_.size()=1024 → sorts uninitialized garbage; flaky; AscendingSort+AllSame
currently fail ("0 vs 42", isSorted=false). Verified via /tmp/radix_repro: correct usage sorts fine; impl NOT buggy.
- fix: sort/readback `h_keys_.size()` (or resize d_keys_).

### F4 [TEST BUG — test quality, category 4]
`tests/neural/transformer/attention_test.cpp`: PositionalEncoding Forward/GetEncoding + MultiHeadAttention
ForwardSelfAttention pass **host std::vector pointers** to device-to-device CUDA API (attention.cu:149/164
cudaMemcpyAsync DtoD) → cudaErrorInvalidValue. MultiHeadAttention::forward also incomplete (stub, see F6).
- fix: allocate device buffers for those tests.

### F5 [TEST BUG — test quality, category 4]
`tests/production/graph_executor_test.cu` CaptureAndLaunch/ScopedCaptureRAII: cudaMalloc/cudaFree INSIDE
cudaStreamBeginCapture → operation not permitted during capture → end_capture fails
"operation failed due to a previous error during capture".
- fix: pre-allocate buffers outside the capture scope.

### F6 [LIBRARY STUB — debt, defer]
`MultiHeadAttention::forward` allocates buffers but never computes attention (no qkv proj, no scores, no output).
Same family as RIL Round-4 FusedMatmulBiasAct redundant-branch gap. Needs real kernel; large scope; GPU-verify needed. DEFER (decision).

### F7 [MAINTAINABILITY — category 3]
`include/cuda/device/cublas_context.h` defines UNGUARDED local `CUBLAS_CHECK` (throws std::runtime_error)
vs canonical guarded CUBLAS_CHECK in error.h (throws CublasException) — include-order-dependent redefinition
(mirror of Round 6/7 CUDA_CHECK issue). Fix: drop local macro, include error.h (CublasException is-a runtime_error).

### F8 [SUITE STABILITY — investigation]
Full-suite run: cuBLAS error-13 cascade on DistributedMatmul (passes in isolation) + SIGSEGV (exit 139) after
GraphExecutorTest. Partially explained by F1 sticky errors + external GPU load (nondeterministic). Residual
risk remains; re-check after F1-F7 fixes.
