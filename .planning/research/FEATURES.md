# Feature Landscape: Extended CUDA Algorithms

**Domain:** GPU Parallel Algorithms (Sorting, Linear Algebra Extras, Numerical Methods, Signal Processing)
**Researched:** 2026-04-28
**Confidence:** HIGH (based on official NVIDIA documentation)

## Executive Summary

This document catalogs GPU algorithm capabilities across four domains targeted for Nova v2.3. Key findings:

- **Sorting & Searching:** CUB provides production-quality radix sort; top-k requires custom implementation
- **Linear Algebra Extras:** cuSOLVER has comprehensive LAPACK coverage (SVD, Eigendecomposition, Cholesky/LU/QR); matrix square root needs custom implementation
- **Numerical Methods:** No standard library support; must implement Monte Carlo, integration, root finding, interpolation from scratch
- **Signal Processing:** cuFFT covers FFT-based convolution; wavelet transforms and FIR/IIR filters require custom kernels

---

## 1. Sorting & Searching

### Table Stakes (Standard CUDA Library Features)

| Feature | Library | Why Expected | Complexity | Notes |
|---------|---------|--------------|------------|-------|
| **Radix Sort** | CUB `DeviceRadixSort` | GPU sorting workhorse; O(n log n) with high parallelism | LOW | Key-value pairs, ascending/descending, bit range selection |
| **Merge Sort** | Thrust `stable_sort` | Guarantees stability; fallback when radix unsuitable | LOW | Uses merge-based approach, memory-heavy |
| **Block-level Sort** | CUB `BlockRadixSort` | Sorting within thread blocks for shared memory | LOW | Specialized for warp/block-level parallelism |
| **Histogram** | CUB `DeviceHistogram` | Counting-based algorithms foundation | LOW | Atomic operations, multiple bin strategies |

### Differentiators (Competitive Features)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Top-K Selection** | Finding k largest/smallest without full sort | MEDIUM | Custom kernel needed; CUB provides `AgentSelect` but not full top-k |
| **Segment Sort** | Sort disjoint array segments independently | HIGH | Requires segment boundary handling, variable-length keys |
| **Sort with Custom Comparator** | Sort by arbitrary predicates (e.g., struct fields) | MEDIUM | CUB requires comparison via functor; thrust more flexible |
| **Nth Element / Order Statistics** | Find k-th order statistic | MEDIUM | Partial sort variant; custom implementation recommended |
| **Distributed Sort** | Multi-GPU sorting with data redistribution | HIGH | CUB lacks native multi-GPU; build on NCCL collectives |

### Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Bitonic Sort** | O(n log² n), less efficient than radix for most sizes | Use radix sort from CUB |
| **Full GPU quicksort** | Poor GPU utilization due to branching | Hybrid: GPU sort + CPU fallback for small arrays |

### Sources

- CUB DeviceRadixSort documentation (HIGH confidence)
- Thrust sorting algorithms (HIGH confidence)
- NVIDIA/cub GitHub repository

---

## 2. Linear Algebra Extras

### Table Stakes (LAPACK-equivalent on GPU)

| Feature | Library | Function | Complexity | Notes |
|---------|---------|----------|------------|-------|
| **Cholesky Decomposition** | cuSOLVER | `potrf` | LOW | Symmetric positive-definite matrices |
| **LU Decomposition** | cuSOLVER | `getrf` | LOW | General matrices with partial pivoting |
| **QR Decomposition** | cuSOLVER | `geqrf`, `orgqr` | MEDIUM | Orthogonal Q, upper triangular R |
| **LDL Decomposition** | cuSOLVER | `sytrf` | MEDIUM | Symmetric indefinite matrices (Bunch-Kaufman) |
| **Bidiagonalization** | cuSOLVER | `gebrd` | MEDIUM | Prepares matrix for SVD |
| **Tridiagonal Solve** | cuSOLVER | `sytrs` | LOW | Solve using LDL factors |

### Eigenvalue & SVD Features

| Feature | Library | Function | Complexity | Notes |
|---------|---------|----------|------------|-------|
| **Symmetric Eigenvalue Decomposition** | cuSOLVER | `syevd`, `syevdx`, `syevj` | MEDIUM | All eigenvalues or range; j= Jacobi |
| **Generalized Symmetric EVD** | cuSOLVER | `sygvd`, `sygvdx`, `sygvj` | MEDIUM | A*x = lambda*B*x |
| **Singular Value Decomposition (SVD)** | cuSOLVER | `gesvd`, `gesvdj`, `gesvdaStridedBatched` | HIGH | j= Jacobi, StrideBatched= column-wise |
| **Nonsymmetric Eigenvalue** | cuSOLVER | `geev` | HIGH | Left/right eigenvectors, Schur form |
| **Polar Decomposition** | cuSOLVER | `polar` | HIGH | A = UP decomposition |

### SVD Algorithm Variants

| Algorithm | Use Case | Trade-offs |
|-----------|----------|------------|
| `gesvd` (Golub-Reinsch) | Standard SVD | Memory efficient, 2-stage |
| `gesvdj` (Jacobi) | Better accuracy | More iterations, potentially slower |
| `gesvdaStridedBatched` | Batch processing | Efficient for multiple matrices |
| `gesvdp` (Partial SVD) | Top-k singular values | Only computes requested values |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Matrix Square Root** | A^0.5 computation | HIGH | No direct cuSOLVER function; requires iterative method or Schur decomposition |
| **Divide-and-Conquer SVD** | Faster for large matrices | HIGH | Custom implementation based on bidiagonal divide-and-conquer |
| **Generalized SVD** | A = U*Sigma*V^T for rectangular | HIGH | Not in cuSOLVER; requires custom |
| **Hessenberg Reduction** | Preconditioning for eigenvalue | MEDIUM | `gehrd` function exists but rarely needed directly |
| **Sparse Eigensolver** | Eigenvalues of sparse matrices | HIGH | cuSolverSP provides shift-inverse power method |

### Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Eigenvalue for general matrices** | Computationally expensive | Convert to upper Hessenberg first if doing many |
| **Dense symmetric eigensolver batch** | `syevjBatched` has limitations | Process in sequence or use custom batched Jacobi |

### Sources

- NVIDIA cuSOLVER documentation (HIGH confidence)
- LAPACK documentation for algorithm references (HIGH confidence)

---

## 3. Numerical Methods

### Table Stakes (No Standard Library Support)

> **CRITICAL:** Neither NVIDIA nor standard CUDA libraries provide Monte Carlo, numerical integration, root finding, or interpolation algorithms. These must be implemented from scratch or via third-party libraries (e.g., ViennaCL, ArrayFire).

| Feature | Status | Implementation Complexity | Notes |
|---------|--------|---------------------------|-------|
| **Monte Carlo Simulation** | Custom only | HIGH | Parallel RNG, reduction, variance reduction |
| **Numerical Integration** | Custom only | MEDIUM-HIGH | Trapezoidal, Simpson, Monte Carlo quadrature |
| **Root Finding** | Custom only | MEDIUM | Bisection, Newton-Raphson, Brent's method |
| **Interpolation** | Custom only | MEDIUM | Cubic splines, Lagrange, linear |

### Monte Carlo Methods

| Technique | Implementation Notes | Difficulty |
|-----------|---------------------|------------|
| **Basic MC Integration** | Sample N points, average f(x), scale by volume | MEDIUM |
| **Antithetic Variates** | Use x and 1-x pairs to reduce variance | MEDIUM |
| **Importance Sampling** | Sample from non-uniform distribution | HIGH |
| **Control Variates** | Use correlated variable with known mean | MEDIUM |
| **Stratified Sampling** | Divide domain into strata, sample each | MEDIUM |
| **Quasi-Monte Carlo** | Use low-discrepancy sequences (Sobol, Halton) | HIGH |

### Key Algorithm Considerations

#### Monte Carlo

```
GPU Parallelism Strategy:
- Each thread processes independent simulation paths
- RNG state per thread (CUDA(curand)) 
- Atomic reduction for sum/mean
- Variance computed via Welford's online algorithm
```

#### Numerical Integration

| Method | GPU Strategy | Accuracy |
|--------|--------------|----------|
| **Trapezoidal** | Parallel prefix sum of function values | O(1/n²) |
| **Simpson's Rule** | Requires function evaluation, then reduction | O(1/n⁴) |
| **Monte Carlo** | Random sampling, parallel reduction | O(1/√n) |
| **Adaptive Quadrature** | Dynamic subdivision, work queue pattern | Variable |

#### Root Finding

| Method | Parallelization | Convergence |
|--------|-----------------|-------------|
| **Bisection** | Search intervals in parallel | Linear, guaranteed |
| **Newton-Raphson** | Each interval runs Newton step | Quadratic, may diverge |
| **Secant Method** | Similar to Newton, uses difference | Superlinear |
| **Brent's Method** | Combines bisection + secant + inverse interp | Superlinear, robust |

#### Interpolation

| Method | GPU Implementation | Use Case |
|--------|-------------------|----------|
| **Linear** | Simple weighted average | Fast approximation |
| **Lagrange** | Polynomial evaluation at points | Exactly fits points |
| **Cubic Spline** | Thomas algorithm for tridiagonal system | Smooth interpolation |
| **Akima Spline** | Robust to outliers | Noisy data |

### Differentiators

| Feature | Value Proposition | Complexity |
|---------|-------------------|------------|
| **Quasi-Monte Carlo** | Faster convergence than pseudo-MC | HIGH |
| **GPU-based ODE Solvers** | Price options, simulate dynamics | HIGH |
| **Automatic Differentiation** | Gradient computation for optimization | HIGH |
| **Polynomial Evaluation** | Horner's method, parallel evaluation | LOW |
| **FFT-based Convolution** | Fast convolution for integral equations | MEDIUM |

### Sources

- ViennaCL library patterns (MEDIUM confidence)
- ArrayFire numerical methods (MEDIUM confidence)
- Numerical Recipes CUDA implementations (MEDIUM confidence)
- Academic papers on GPU quadrature (LOW confidence)

---

## 4. Signal Processing

### Table Stakes (cuFFT Foundation)

| Feature | Library | Function | Complexity | Notes |
|---------|---------|----------|------------|-------|
| **1D FFT** | cuFFT | `cufftPlan1d`, `cufftExecC2C` | LOW | Single transform |
| **2D/3D FFT** | cuFFT | `cufftPlan2d/3d` | LOW | Multi-dimensional |
| **Batch FFT** | cuFFT | `cufftPlanMany` | LOW | Multiple transforms |
| **Real FFT (R2C/C2R)** | cuFFT | `cufftExecR2C/D2Z` | LOW | Exploits Hermitian symmetry |
| **Multi-GPU FFT** | cuFFT Xt | `cufftXtSetGPUs` | MEDIUM | Up to 16 GPUs |
| **Inverse FFT** | cuFFT | Same plan, inverse direction | LOW | Standard inverse |

### FFT Capabilities

| Feature | Description | Notes |
|---------|-------------|-------|
| **Supported Sizes** | 2^a * 3^b * 5^c * 7^d | Smaller primes = better performance |
| **Precision** | FP16, FP32, FP64, BF16 | BF16 requires SM80+ |
| **Plan Caching** | Reuse plans for same size | Significant speedup |
| **Callback Routines** | LTO load/store callbacks | Custom pre/post processing |
| **CUDA Graphs** | Capture FFT execution | For efficient batching |

### Signal Processing Features (Custom Implementation)

| Feature | Implementation Approach | Complexity |
|---------|------------------------|------------|
| **Wavelet Transform (Haar)** | Recursive downsampling with pair averaging | LOW |
| **Wavelet Transform (Daubechies)** | Convolution with Daubechies filters | MEDIUM |
| **Inverse Wavelet** | Transpose of forward with synthesis filters | MEDIUM |
| **FIR Filters** | Convolution via FFT or direct | LOW-MEDIUM |
| **IIR Filters** | Recursive computation per sample | MEDIUM |
| **FFT-based Convolution** | FFT → multiply → IFFT | LOW (leverages cuFFT) |
| **Winograd Convolution** | Reduced multiplications for small filters | HIGH |

### Wavelet Transform Implementation Patterns

```
Haar Wavelet (Simple):
1. For N elements: compute N/2 averages (low freq) and N/2 differences (high freq)
2. Recursively apply to averages for multi-level decomposition
3. Inverse: reconstruct from averages and differences

Daubechies D4:
1. Convolve with low-pass and high-pass filters
2. Downsample by 2
3. Filters coefficients are predefined (db4)
```

### Filter Design (Custom)

| Filter Type | Design Approach | Implementation |
|-------------|-----------------|----------------|
| **Moving Average** | Simple sliding window | Shared memory reduction |
| **Gaussian Blur** | Convolve with Gaussian kernel | Separable FFT convolution |
| **Butterworth** | IIR design from specs | Transposed direct form II |
| **Chebyshev** | IIR with equiripple | Same as Butterworth |
| **Box Filter** | Uniform averaging | Integral image approach |

### Spectral Analysis

| Feature | Library | Complexity | Notes |
|---------|---------|------------|-------|
| **Power Spectral Density** | Custom (FFT + magnitude²) | LOW | Uses existing cuFFT |
| **Cross-Spectral Density** | Custom | MEDIUM | Two FFTs, complex conjugate multiply |
| **Short-Time FFT (STFT)** | Custom | MEDIUM | Window + FFT + slide |
| **Welch's Method** | Custom | MEDIUM | Overlapping segments, average PSD |
| **Autocorrelation** | Custom | LOW | IFFT of power spectrum |

### Differentiators

| Feature | Value Proposition | Complexity |
|---------|-------------------|------------|
| **2D Wavelet Transform** | Image compression, denoising | MEDIUM |
| **Continuous Wavelet Transform** | Time-frequency analysis | HIGH |
| **Adaptive Filters (LMS, RLS)** | Signal tracking, noise cancellation | HIGH |
| **Filter Bank Design** | Multi-rate signal processing | MEDIUM |
| **STFT with Phase Vocoder** | Time-stretching, pitch-shifting | HIGH |

### Anti-Features

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| **Full wavelet library** | Complexity explosion | Implement Haar first, then Daubechies |
| **Real-time audio processing** | Not in scope for this milestone | Document as future feature |
| **Image convolution without FFT** | Direct convolution O(n²k²) | Use FFT-based for large kernels |

### Sources

- NVIDIA cuFFT documentation (HIGH confidence)
- Wavelet theory: Daubechies "Ten Lectures on Wavelets" (HIGH confidence)
- Filter design: Oppenheim & Schafer, "Discrete-Time Signal Processing" (HIGH confidence)

---

## Feature Dependencies

### Sorting & Searching
```
CUB DeviceRadixSort
    └──requires──> Device memory allocation
    └──enhances──> Top-K (build on sorted segments)
    └──enhances──> Order statistics (partial sort)

Thrust stable_sort
    └──requires──> Device memory, thrust headers
    └──fallback──> When radix sort unsuitable
```

### Linear Algebra
```
cuSOLVER potrf (Cholesky)
    └──requires──> cuBLAS
    └──uses──────> Symmetric positive-definite matrices

cuSOLVER gesvdj (SVD Jacobi)
    └──requires──> cuBLAS
    └──optional──> syevjInfo for convergence tuning
    └──precedes──> Matrix square root (via SVD)

Matrix Square Root (Custom)
    └──requires──> SVD decomposition
    └──algorithm──> A^0.5 = V * diag(sqrt(S)) * V^T
```

### Numerical Methods
```
Monte Carlo Simulation
    └──requires──> cuRAND for RNG
    └──uses──────> Atomic operations for reduction
    └──optional──> Variance reduction techniques

Numerical Integration (Trapezoidal)
    └──requires──> Function evaluation kernel
    └──uses──────> Prefix sum for cumulative sum

Root Finding (Newton-Raphson)
    └──requires──> Derivative evaluation (or secant variant)
    └──uses──────> Convergence test per thread
```

### Signal Processing
```
cuFFT Plan
    └──requires──> cufft.h header
    └──optimizes──> FFT-based convolution
    └──optional──> cuFFT Xt for multi-GPU

Wavelet Transform
    └──requires──> Parallel downsampling
    └──optional──> Multi-level decomposition

FIR Filter
    └──optimizes──> FFT-based for large kernels
    └──uses──────> Direct convolution for small kernels
```

---

## MVP Definition

### Phase 1: Sorting & Searching (Priority: HIGH)

Minimum viable sorting algorithms for v2.3:

- [ ] CUB-based radix sort (key-value pairs)
- [ ] Block-level sort primitive
- [ ] Top-k selection kernel
- [ ] Binary search on sorted arrays

### Phase 2: Linear Algebra Extras (Priority: HIGH)

Minimum viable LAPACK coverage:

- [ ] SVD wrapper (`gesvdj` with tunable parameters)
- [ ] Symmetric eigenvalue decomposition (`syevd`)
- [ ] QR decomposition wrapper (`geqrf` + `ormqr`)
- [ ] Cholesky with batched variants (`potrfBatched`)

### Phase 3: Numerical Methods (Priority: MEDIUM)

Core numerical algorithms:

- [ ] Monte Carlo with cuRAND (antithetic variates)
- [ ] Trapezoidal integration (prefix sum-based)
- [ ] Newton-Raphson root finding
- [ ] Cubic spline interpolation

### Phase 4: Signal Processing (Priority: MEDIUM)

Signal processing primitives:

- [ ] FFT-based convolution (using cuFFT)
- [ ] Haar wavelet transform (forward/inverse)
- [ ] FIR filter implementation (direct and FFT-based)
- [ ] Basic spectral analysis (PSD via FFT)

---

## Complexity Assessment by Feature

| Domain | Table Stakes | Differentiators | Complexity Notes |
|--------|--------------|-----------------|------------------|
| **Sorting** | CUB radix sort | Top-k, segment sort | Low for basics; medium for advanced |
| **Linear Algebra** | cuSOLVER LAPACK | Matrix sqrt, generalized SVD | Low (wrappers); high (custom algos) |
| **Numerical Methods** | Custom only | QMC, adaptive quadrature | High overall; medium per method |
| **Signal Processing** | cuFFT + custom | Wavelets, adaptive filters | Medium for FFT-based; high for wavelets |

---

## Sources

### Primary Sources (HIGH Confidence)

1. **NVIDIA cuBLAS Documentation v13.2** - Dense linear algebra API
2. **NVIDIA cuSOLVER Documentation v13.2** - LAPACK on GPU, SVD, eigenvalue decomposition
3. **NVIDIA cuFFT Documentation v13.2** - Fast Fourier Transform library
4. **NVIDIA CUB GitHub (archived, now in CCCL)** - Device-wide primitives including sort
5. **NVIDIA Thrust GitHub** - Low-level algorithm wrappers

### Secondary Sources (MEDIUM Confidence)

6. **ViennaCL Documentation** - Open-source GPU numerical methods patterns
7. **ArrayFire Documentation** - GPU computing library numerical methods
8. **cuRAND Documentation** - Random number generation for Monte Carlo

### Tertiary Sources (LOW Confidence, Verify)

9. **Wavelet literature**: Daubechies, "Ten Lectures on Wavelets" (1992)
10. **Filter design**: Oppenheim & Schafer reference implementations
11. **Monte Carlo variance reduction**: academic papers on GPU MC integration

---

## Research Gaps & Future Investigation

| Gap | Why Needed | When to Investigate |
|-----|------------|---------------------|
| **Sparse eigensolver** | Scientific computing use cases | Phase 4+ |
| **Multi-GPU sorting** | Large dataset handling | Phase 4+ |
| **cuDSS migration** | cuSolverSP deprecation | cuSolverSP users |
| **cuFFTDx for 1D FFT** | Multi-GPU 1D FFT efficiency | If multi-GPU FFT needed |
| **Matrix function gradients** | Automatic differentiation | Future research |

*Last updated: 2026-04-28 for Nova v2.3 Extended Algorithms*
