# Stack Research: CUDA Algorithm Libraries for Nova v2.3

**Domain:** Production CUDA parallel algorithms library
**Researched:** 2026-04-28
**Confidence:** HIGH — NVIDIA official documentation (cuBLAS 13.2, cuSOLVER 13.2, cuFFT 13.2) verified

## Executive Summary

Nova v2.3 extends the CUDA library with production-quality parallel algorithms across sorting, linear algebra decomposition, numerical methods, and signal processing domains. This research identifies that NVIDIA provides production-grade implementations for most target algorithms via cuSOLVER, CUB, and cuFFT. Custom implementations are needed only for wavelet transforms, Monte Carlo methods, and root-finding/interpolation. The recommended approach is thin wrapper/proxy patterns around NVIDIA libraries with Nova-idiomatic APIs, leveraging existing five-layer architecture.

## Standard Algorithms in NVIDIA Libraries

### cuBLAS (v13.2) — Linear Algebra Foundation

**What it provides:**
- **Level-1:** Vector ops (dot, nrm2, axpy, scal, rot, amax/amin, asum, copy, swap)
- **Level-2:** Matrix-vector ops (gemv, symv, trmv, ger, syr)
- **Level-3:** Matrix-matrix ops (gemm, symm, trmm, syrk, herk, trsm)
- **Extensions:** GEAM (general matrix-matrix addition), DGMM (diagonal scaling), batched gemm variants

**Key capabilities:**
- Tensor core support for FP16/BF16/FP8 mixed-precision GEMM
- Batched operations for multiple small matrices
- Strided batched for large batch sizes
- 64-bit integer interface for large matrices
- Floating-point emulation (BF16x9 for FP32, fixed-point for FP64) for performance on older hardware

**Integration point:** Nova already uses cuBLAS for matrix operations. Extended wrappers needed for batched operations and precision variants.

### cuSOLVER (v13.2) — Decompositions and Eigenvalue Problems

**What it provides (Dense — cuSolverDN):**

| Routine | Function | Notes |
|---------|----------|-------|
| `potrf` | Cholesky factorization | Symmetric/Hermitian positive definite |
| `getrf` | LU factorization | General matrix with partial pivoting |
| `geqrf` | QR factorization | General matrix |
| `sytrf` | LDL factorization | Symmetric indefinite (Bunch-Kaufman) |
| `gesvd` / `gesvdj` | SVD | Jacobi algorithm for faster convergence |
| `syevd` / `syevj` | Symmetric eigenvalue | Divide-and-conquer / Jacobi |
| `geev` | Non-symmetric eigenvalue | Schur decomposition |
| `orgqr` / `ormqr` | QR reconstruction | Build Q from reflectors |

**Key features:**
- **64-bit API** (`cusolverDnX*`): Supports matrices > 2^31 elements
- **Iterative Refinement Solver (IRS)**: GMRES-based refinement for higher accuracy with lower precision
- **Jacobian solvers** (`*j` variants): Configurable tolerance, max sweeps, eigenvalue sorting
- **Deterministic mode**: Bit-exact results (disabled by default for performance)
- **Workspace pattern**: `*_bufferSize` returns required workspace, user allocates

**cuSolverSP (DEPRECATED):** Sparse solvers moved to cuDSS
**cuSolverRF (DEPRECATED):** Refactorization moved to cuDSS
**cuSolverMG (DEPRECATED):** Multi-GPU moved to cuSOLVERMp

**Integration point:** Use cuSolverDN for SVD (`gesvdj`) and eigenvalue decomposition (`syevd`). Nova already has matrix operations — wrappers needed for factorization and decomposition.

### cuFFT (v13.2) — Fast Fourier Transform

**What it provides:**
- 1D, 2D, 3D transforms
- Complex-to-complex (C2C), real-to-complex (R2C), complex-to-real (C2R)
- Batched transforms (multiple independent FFTs)
- Strided layouts for non-contiguous data
- Multi-GPU transforms (up to 16 GPUs)
- Callback routines for custom load/store operations
- Half-precision (FP16) and BFloat16 transforms (SM 7.5+)
- Link-time optimized kernels (CUDA 12.6+)

**Key features:**
- Plan-based API (FFTW-style): Create plan once, reuse for multiple transforms
- Just-in-time kernel compilation for optimal performance on target architecture
- CUDA Graphs support for recording and replaying FFT sequences
- Multi-GPU with automatic data decomposition and communication

**Integration point:** Nova already has FFT via existing image processing. Multi-GPU FFT and precision variants are extensions.

### CUB (CUDA Unbound) — Low-Level Primitives

**What it provides:**
- **Warp-level primitives**: Shuffle, reduction, scan, prefix sum
- **Block-level collectives**: Cooperative I/O, sort, scan, reduction, histogram
- **Device-level algorithms**: Parallel sort, prefix scan, reduction, histogram, radix sort
- **Utilities**: PTX intrinsics, iterators, device/thread block management

**Key capabilities:**
- Architecture-specific specializations (5.0 through 9.0)
- Compatible with CUDA dynamic parallelism
- Header-only library (include `cub/cub.cuh`)

**Integration point:** CUB powers Nova's existing sort, reduce, scan. New algorithms (top-k, binary search) should use CUB device primitives.

### Thrust — STL-like GPU Algorithms

**What it provides (via CUB backend):**
- `thrust::sort`, `thrust::stable_sort`
- `thrust::reduce`, `thrust::transform_reduce`
- `thrust::scan`, `thrust::exclusive_scan`
- `thrust::copy_if`, `thrust::remove`
- `thrust::device_vector`, memory allocators

**Notes:**
- Thrust is header-only
- Nova does NOT use Thrust (CUB directly instead)
- Thrust's CUDA backend uses CUB; could provide Thrust host API wrappers for users familiar with STL

## State-of-the-Art Approaches by Domain

### Sorting & Searching

**CUB Radix Sort (PRODUCTION)**
```
cub::DeviceRadixSort::SortKeys(...)
cub::DeviceRadixSort::SortPairs(...)
```
- O(n log n) for arbitrary keys
- Stable sort variants available
- Key-value sorting (sort keys, reorder values in parallel)

**CUB Block Sort (WARP/BLOCK LEVEL)**
```
cub::BlockSort<BlockTile, BlockSize, T, BlockedThreadArrive>
```
- In-register sorting for small arrays
- Memory-efficient for block-local data

**Top-K Operations**
- **NOT in CUB directly**: Implement using reduce-by-key + segmented sort
- Pattern: Segmented sort, take first K elements
- Alternative: Use CUB's `DeviceReduce::ArgMax` for max-K queries

**Binary Search**
- **NOT in CUB**: Implement using warp-level binary search
- Pattern: Each thread searches segment, warp cooperative binary search
- Use `__shfl_*` for warp-level value exchange

**Recommendation:** Wrap CUB sort, implement top-K via segmented sort, implement binary search with warp shuffle.

### Linear Algebra Operations (SVD, Eigendecomposition)

**SVD via cuSOLVER**
```
cusolverDnXgesvd(handle, ...)
// or Jacobi (faster convergence):
cusolverDnXgesvdp(handle, ...)
// or randomized for approximate SVD:
cusolverDnXgesvdr(handle, ...)
```
- **gesvd**: One-sided bidiagonalization, U/S/V^T returned
- **gesvdj**: Jacobi-based, faster for small matrices, configurable tolerance
- **gesvdjBatched**: Batched SVD for multiple small matrices
- **gesvdaStridedBatched**: Strided batched with absolute error tolerance
- **gesvdr**: Randomized SVD for low-rank approximation (much faster for tall matrices)

**Eigenvalue Decomposition via cuSOLVER**
```
cusolverDnXsyevd(handle, ...)     // Divide-and-conquer
cusolverDnXsyevdx(handle, ...)    // Eigenvalue range
cusolverDnXgeev(handle, ...)      // Non-symmetric (left/right eigenvectors)
cusolverDnXstedc(handle, ...)     // Divide-and-conquer for tridiagonal
```
- **syevd**: All eigenvalues/vectors of symmetric matrix
- **syevdx**: Subset by index range or value range
- **syevj**: Jacobi variant with configurable tolerance
- **geev**: Non-symmetric (real Schur form, left/right eigenvectors)

**Recommendation:** Use cuSOLVERDN wrappers. For randomized SVD (gesvdr), use for approximate PCA/compression use cases.

### Numerical Methods

**Monte Carlo Integration**
- **NOT in NVIDIA libraries**: Custom implementation required
- Approach: Parallel pseudo-random number generation with cuRAND
- Pattern: Each thread generates samples, reduce to mean/variance
- cuRAND: `curandState` per thread, skip ahead for independence
- Variance reduction: Antithetic variates, control variates (implement manually)

**Numerical Integration**
- **NOT in NVIDIA libraries**: Custom implementation required
- Approaches:
  1. Trapezoidal/Simpson: Parallel prefix with reduction
  2. Gaussian quadrature: Lookup weights, vectorized evaluation
  3. Monte Carlo: See above

**Root Finding**
- **NOT in NVIDIA libraries**: Custom implementation required
- Approaches:
  1. Bisection: Binary search with convergence check
  2. Newton-Raphson: Requires gradient computation
  3. Brent's method: Requires function evaluations on host
- **Recommendation**: Bisection for simple cases, limited Newton iterations for speed

**Interpolation**
- **NOT in NVIDIA libraries**: Custom implementation required
- Approaches:
  1. Linear: Vectorized lookup + weighted average
  2. Cubic spline: Pre-compute coefficients on host, evaluate on device
  3. Lagrange: Parallel evaluation of basis polynomials
- **Recommendation**: Linear interpolation first, cubic spline as extension

### Signal Processing (Wavelets, Convolution)

**Wavelet Transform**
- **NOT in NVIDIA libraries**: Custom implementation required
- Approaches:
  1. Haar wavelets: Simple high/low pass filter, subsample
  2. Daubechies: Convolution with pre-computed filters
  3. Biorthogonal: Separable 2D via row/column transforms
- **Pattern**: Use cuFFT for convolution-based wavelet (e.g., Morlet)
- **Recommendation**: Implement Haar (trivial), add Daubechies via convolution

**Convolution**
- **cuFFT for large kernels**: FFT(x) * FFT(kernel), IFFT
- **Direct convolution**: For small kernels, shared-memory tiled convolution
- **cuDNN (if available)**: For standard neural network convolutions
- Pattern: Use FFT for kernel > 16 elements, direct otherwise

**Filtering**
- **cuFFT-based**: Frequency-domain multiplication
- **Direct**: IIR (recursive) or FIR (finite impulse response)
- FIR filter: `convolution` of signal with filter kernel
- IIR filter: Recurrence relation per output element (sequential per row)

**Recommendation:** Use cuFFT for FFT-based wavelet and filtering. Direct convolution for small kernels.

## CUDA Capabilities Required

### Warp-Level Primitives

**Available via CUB or PTX intrinsics:**

| Operation | CUB API | PTX Intrinsic |
|-----------|---------|---------------|
| Shuffle | `WarpShuffle` | `__shfl_*` |
| Reduction | `WarpReduce` | `__reduce_*` |
| Scan | `WarpScan` | Manual with `__shfl_up` |
| Broadcast | `WarpScan` | `__shfl_sync` |

**Key patterns:**
```cpp
// Warp shuffle broadcast
auto value = __shfl_sync(mask, val, root_lane);

// Warp reduction
auto sum = WarpReduce(temp_storage).Sum(value);

// Warp scan (prefix sum)
WarpScan(temp_storage).InclusiveSum(value, result);
```

**Required for:** Binary search, parallel reduction patterns, warp-cooperative algorithms.

### Shared Memory Patterns

**Tiled convolution pattern:**
```cpp
__shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];

// Cooperative load into shared memory
if (isValid) tile[ty][tx] = input[row][col];
__syncthreads();

// Compute with shared memory accesses
float sum = 0;
for (int i = 0; i < KERNEL_SIZE; i++)
    for (int j = 0; j < KERNEL_SIZE; j++)
        sum += tile[ty + i][tx + j] * kernel[i][j];
```

**Required for:** Direct convolution, stencil operations, wavelet decomposition.

### Memory Access Patterns

**Coalesced memory access:** Threads in a warp access consecutive memory addresses.
```
Thread 0 -> addr + 0
Thread 1 -> addr + 1
...
Thread 31 -> addr + 31
```

**Bank conflict avoidance:** Shared memory access patterns that avoid same-bank collisions.
- Padding shared memory dimensions
- Using `__ldg` for read-only data (texture cache)

**L2 cache persistence:** For iterative algorithms, use `cudaMallocAsync` with persisting access policy.

**Required for:** All bulk memory operations (sorting, FFT, matrix operations).

## Recommended Implementation Stack

### Libraries to Use (NVIDIA)

| Library | Version | Purpose | Nova Integration |
|---------|---------|---------|------------------|
| cuBLAS | 13.2 | Matrix operations (already used) | Extend with batched ops |
| cuSOLVER | 13.2 | SVD, eigenvalue decomposition | New wrappers |
| cuFFT | 13.2 | FFT, convolution | Already used, extend multi-GPU |
| CUB | Latest (bundled with CUDA) | Sort, reduce, scan, warp primitives | Use for new algorithms |
| cuRAND | 13.2 | Random number generation | Monte Carlo |

### Libraries to NOT Use (Already Covered or Deprecated)

| Library | Why Not | Use Instead |
|---------|---------|-------------|
| cuSolverSP | Deprecated | cuDSS |
| cuSolverRF | Deprecated | cuDSS |
| cuSolverMG | Deprecated | cuSOLVERMp |
| Thrust | Nova uses CUB directly | CUB device primitives |
| cuDNN | Not needed for these algorithms | N/A |
| MAGMA | CPU-GPU hybrid, not pure GPU | cuSOLVERDN |

### New Custom Implementations Needed

| Algorithm | Approach | Complexity |
|-----------|----------|------------|
| Top-K selection | CUB segmented sort or argmax | Medium |
| Binary search | Warp shuffle binary search | Medium |
| Monte Carlo | cuRAND + parallel reduction | Low |
| Numerical integration | Parallel prefix/trapezoidal | Low |
| Root finding | Bisection (simple) or Newton | Low |
| Interpolation | Linear (trivial), cubic (medium) | Low-Medium |
| Wavelet transforms | Direct convolution or FFT | Medium |

## Integration Points with Nova Architecture

### Five-Layer Architecture Alignment

| Layer | Existing | Extension for v2.3 |
|-------|----------|-------------------|
| Memory (L1) | Buffer, unique_ptr, MemoryPool | No changes needed |
| Device (L2) | Device management | No changes needed |
| Algorithm (L3) | reduce, scan, sort, histogram | Add: top-k, binary search, numerical methods |
| API (L4) | Public algorithms | Add: SVD, eigenvalues, wavelets, Monte Carlo |
| Application (L5) | Domain-specific | No changes |

### Code Patterns

**cuSOLVER wrapper pattern:**
```cpp
// Header: include/nova/algo/svd.hpp
namespace nova {
class SVD {
public:
    struct Result { /* U, S, V^T */ };
    static Expected<Result> compute(const Buffer& A, int m, int n, cudaStream_t stream = 0);
};
}
```

**Custom algorithm pattern (Monte Carlo):**
```cpp
// Header: include/nova/algo/monte_carlo.hpp
namespace nova {
Expected<double> pi_estimation(int samples, cudaStream_t stream = 0);
}
```

**Integration with existing algorithms:**
- Sort: `cub::DeviceRadixSort` wrapped in Nova's Buffer API
- FFT: Use existing cuFFT integration for wavelet transforms
- Random: `curand` wrapped for Monte Carlo

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Reimplementing cuSOLVER algorithms | NVIDIA provides production-optimized versions | Wrap cuSOLVERDN |
| Reimplementing CUB sort | Already in Nova | Extend if needed |
| GPU random without cuRAND | Quality matters for Monte Carlo | cuRAND |
| General matrix multiplication | Already in Nova via cuBLAS | Extend with batched |
| Generic FFT | Already in Nova via cuFFT | Extend with multi-GPU |
| Sparse solvers | cuSolverSP deprecated, use cuDSS separately | Separate dependency |
| cuDNN for signal processing | Not needed for target algorithms | N/A |

## Sources

- [cuBLAS 13.2 Documentation](https://docs.nvidia.com/cuda/cublas/) — HIGH
- [cuSOLVER 13.2 Documentation](https://docs.nvidia.com/cuda/cusolver/) — HIGH
- [cuFFT 13.2 Documentation](https://docs.nvidia.com/cuda/cufft/) — HIGH
- [CUB GitHub / Documentation](https://nvlabs.github.io/cub/) — HIGH
- [NVIDIA cuRAND Documentation](https://docs.nvidia.com/cuda/curand/) — HIGH
- [CUDA C++ Programming Guide — Warp Shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions) — HIGH
- [cuDSS (successor to cuSolverSP)](https://developer.nvidia.com/cudss) — MEDIUM

---
*Stack research for: Nova v2.3 Extended Algorithms*
*Researched: 2026-04-28*
