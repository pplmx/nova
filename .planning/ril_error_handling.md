# Repository Intelligence Layer (RIL)

## Error Handling Pattern: CUDA_CHECK Adoption

### Decision: Wrap unchecked CUDA runtime calls with CUDA_CHECK

**Date:** 2026-07-23
**Status:** Active
**Author:** Autonomous Engineering Agent

### Context

The codebase has an established pattern of wrapping `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and `cudaSetDevice` calls with `CUDA_CHECK` (defined in `include/cuda/device/error.h`). Recent commits (b9a5a1e, 79d4557, 3951998, 44a4be1) have been systematically applying this pattern to previously unchecked calls.

However, several files still contained unchecked CUDA runtime calls that could silently fail, leading to undefined behavior (null pointer dereferences, memory corruption) rather than clear error messages.

### Namespace Collision Issue

**Critical finding:** The `CUDA_CHECK` macro from `include/cuda/device/error.h` expands to `cuda::device::CudaException`. In files where the enclosing namespace is `nova::quantize` or `nova::quantize::cuda`, the unqualified name `cuda` resolves to the local namespace (e.g., `nova::quantize::cuda`) rather than the top-level `::cuda`, causing compilation errors.

**Affected namespaces:**
- `nova::quantize` → `cuda` resolves to `nova::quantize::cuda` (collision)
- `nova::quantize::cuda::detail` → same collision

**Safe namespaces** (where `cuda` resolves to top-level `::cuda`):
- `cuda::neural` (nested inside `cuda`)
- `cuda::neural::fusion` (nested inside `cuda`)
- `cuda::production` (nested inside `cuda`)

**Mitigation:** For files in `nova::quantize` or `nova::quantize::cuda` namespaces, use explicit `::cuda::device::CudaException` (fully-qualified) instead of the `CUDA_CHECK` macro, or use inline error checking with `cudaError_t err = ...; if (err != cudaSuccess) throw ::cuda::device::CudaException(err, __FILE__, __LINE__);`.

### Files Modified (Round 1: 2026-07-23)

1. `src/cuda/neural/sync_batch_norm.cu` (in `cuda::neural` namespace)
   - `SyncBatchNorm::backward()`: 3x `cudaMalloc` → `CUDA_CHECK(cudaMalloc(...))`
   - `SyncBatchNorm::backward()`: 3x `cudaFree` → `CUDA_CHECK(cudaFree(...))`
   - `sync_batch_norm_forward_training()`: 2x `cudaMemcpy` → `CUDA_CHECK(cudaMemcpy(...))`

2. `src/cuda/neural/fusion/kernel_fusion.cu` (in `cuda::neural::fusion` namespace)
   - Added `#include "cuda/device/error.h"`
   - `FusedLayerNormSoftmax::forward()`: 1x `cudaMalloc` → `CUDA_CHECK(cudaMalloc(...))`
   - `FusedLayerNormSoftmax::forward()`: 1x `cudaFree` → `CUDA_CHECK(cudaFree(...))`

3. `src/cuda/neural/fusion/fused_matmul_bias_act.cu` (in `cuda::neural::fusion` namespace)
   - Added `#include "cuda/device/error.h"`
   - `FusedMatmulBiasAct` constructor: 1x `cudaMalloc` → `CUDA_CHECK(cudaMalloc(...))`

4. `src/cuda/production/algo_wrapper.cu` (in `cuda::production` namespace)
   - Added `#include "cuda/device/error.h"`
   - `GraphReduceWrapper::capture()`: 1x `cudaMalloc` → `CUDA_CHECK(cudaMalloc(...))`
   - `GraphReduceWrapper::capture()`: 1x `cudaFree` → `CUDA_CHECK(cudaFree(...))`

5. `src/cuda/quantize/calibrator.cpp` (in `nova::quantize` namespace — collision zone)
   - Added `#include "cuda/device/error.h"`
   - `HistogramCalibrator::calibrate()`: 1x `cudaMalloc` → inline `::cuda::device::CudaException`
   - `HistogramCalibrator::calibrate()`: 1x `cudaMemcpy` → inline `::cuda::device::CudaException`
   - `HistogramCalibrator::calibrate()`: 1x `cudaFree` → inline `::cuda::device::CudaException`

6. `src/cuda/quantize/int8_kernels.cu` (in `nova::quantize::cuda::detail` namespace — collision zone)
   - Added `#include "cuda/device/error.h"`
   - `build_histogram()`: 1x `cudaMalloc` + 1x `cudaMemcpy` → inline `::cuda::device::CudaException`
   - `compute_minmax()`: 2x `cudaMalloc` → inline `::cuda::device::CudaException`

### Remaining Unchecked Calls

**Intentionally unchecked (error injection tests):**
- `src/cuda/production/error_injection.cu` — `run_memory_pressure_test()` intentionally captures `cudaError_t` and handles failures gracefully as part of stress testing.
- `src/cuda/production/error_injection.cu` — `run_concurrent_stream_test()` line 184 `cudaMalloc` is in a test path where failures are expected.
- `src/testing/integration.cpp` — line 75 `cudaMalloc(&ptr, 1024 * 1024) == cudaSuccess` is a device availability probe.
- `src/testing/boundary_testing.cpp` — `cudaMalloc` with error return used for boundary testing.

**Pre-existing test failures (not caused by our changes):**
- `SyncBatchNormTest` — 4 tests fail with "CUDA error: out of memory" because no CUDA device is available in the CI/test environment. This was confirmed by running the same tests against the pre-change code (identical failures).

### Verification

- `cmake --build build --target cuda_impl` — SUCCESS
- `cmake --build build --target nova-tests` — SUCCESS
- `nova-tests --gtest_filter='*Fusion*'` (non-GPU tests) — PASS
- `nova-tests --gtest_filter='*SyncBatchNorm*'` — Pre-existing failures (CUDA OOM, same before and after changes)
- `nova-tests --gtest_filter='*INT8*:*Calibrat*'` — All skipped (CUDA not available)

### Files Modified (Round 2: 2026-08-09)

7. `src/cuda/neural/tensor_parallel_matmul.cpp` (in `cuda::neural` namespace — safe, nested inside `cuda`)
   - Added `#include "cuda/device/error.h"`
   - `TensorParallelMatmul::matmul()`: `cudaStreamCreate` / `cudaStreamSynchronize` / `cudaStreamDestroy` → `CUDA_CHECK(...)`
   - `recommend_tp_degree()`: `cudaGetDeviceCount(&device_count)` → `CUDA_CHECK(cudaGetDeviceCount(&device_count))`, initializing `device_count = 0` instead of leaving it uninitialized on failure.

Unknown machines in `recommend_tp_degree()` previously read an uninitialized `device_count` if `cudaGetDeviceCount` failed silently; now it throws a clear `CudaException` instead of returning a garbage TP degree.

Note: destructor/teardown paths (e.g. `cudaEventDestroy` in `kernel_stats.cpp`, `profiler.cpp`, `csr_graph.cu::free_device`) keep unchecked calls **intentionally** because throwing from a `noexcept` destructor is undefined behavior. These are documented as-is and are not regressions.
