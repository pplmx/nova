# Feature Landscape — v2.8 Numerical Computing & Performance

**Domain:** GPU numerical computing and performance analysis
**Researched:** 2026-05-01
**Confidence:** HIGH (primary sources from NVIDIA cuSOLVER/cuSPARSE/cuDSS docs)

## Executive Summary

This milestone adds three feature categories to the Nova CUDA library:
1. **Krylov iterative solvers** built on existing SpMV infrastructure
2. **Roofline performance model** leveraging existing bandwidth measurement
3. **Advanced sparse formats** (ELL/HYB/SELL) extending existing CSR/CSC support

**Key insight:** NVIDIA's cuSolverSP (sparse iterative solvers) is deprecated. The replacement cuDSS is a direct solver. True iterative Krylov methods (CG, GMRES, BiCGSTAB) require custom implementation using SpMV, which Nova already provides from v2.1.

---

## 1. Krylov Solver Features

### Category Overview

**Type:** Iterative linear solvers for sparse/dense systems
**Complexity:** Medium-High
**Dependencies:** SpMV (v2.1), cuBLAS dotproduct, memory pool
**Library Status:** cuSolverSP deprecated; cuDSS is direct (not iterative); custom implementation required

### Table Stakes Features

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| CG solver (Conjugate Gradient) | Standard iterative solver for symmetric positive-definite systems | Medium | Requires SpMV, dot product, vector operations |
| GMRES solver (Generalized Minimal Residual) | Standard solver for non-symmetric systems | High | More complex than CG; requires Arnoldi iteration |
| Convergence criteria | User must know when iteration stops | Low | Relative residual norm, max iterations |
| Preconditioner interface | Without preconditioning, convergence is slow/unpredictable | Medium | ILU0, Jacobi as starting points |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Adaptive convergence tolerance | Match solution accuracy to problem needs | Low | FP64/FP32/FP16 tolerance per solver |
| Solver status reporting | Diagnostic information for debugging | Low | Iteration count, final residual, convergence history |
| SpMV backend selection | CSR/CSC already available; ELL/HYB planned | Medium | Format-aware kernel selection |
| Memory pool integration | Reuse temporaries across solver iterations | Medium | Avoid allocation per iteration |

### Anti-Features (Explicitly NOT Building)

| Anti-Feature | Why Avoid | What To Do Instead |
|--------------|-----------|-------------------|
| Direct sparse solvers (LU/Cholesky) | Use cuDSS for direct solves | Provide cuDSS wrapper in separate module |
| Full preconditioner library | Scope creep; complex to implement/test | Just ILU0/Jacobi as starting point |
| Eigenvalue solvers via Arnoldi | Separate feature; defer to future milestone | Note as future work |
| Matrix-free solvers | Premature abstraction | Focus on explicit sparse matrix formats first |

### Implementation Notes

**CG Algorithm Overview:**
```
r = b - Ax
p = r
while ||r|| > tolerance and iter < max_iter:
    Ap = SpMV(A, p)
    alpha = (r . r) / (p . Ap)
    x = x + alpha * p
    r_new = r - alpha * Ap
    beta = (r_new . r_new) / (r . r)
    p = r_new + beta * p
    r = r_new
```

**Key Operations (all available or planned):**
- SpMV: Nova already has CSR/CSC from v2.1
- Dot product: cuBLAS dot (already available)
- Vector operations: axpy, scale, copy (already in library)
- Norm: cuBLAS nrm2 (already available)

**cuDSS vs Custom Krylov:**
- cuDSS: Direct solver with iterative refinement (GMRES-based refinement in cuSOLVER)
- Custom Krylov: True iterative solvers needed for preconditioned solves
- Nova should implement custom iterative solvers for control over preconditioning

---

## 2. Roofline Model Features

### Category Overview

**Type:** Performance analysis tool
**Complexity:** Low-Medium
**Dependencies:** Bandwidth measurement (v2.7), device properties (existing)
**Mathematical Basis:** Arithmetic intensity vs device bandwidth

### Table Stakes Features

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Device peak FLOP/s lookup | Roofline baseline for each compute capability | Low | NVIDIA provides published peak values |
| Device memory bandwidth | Roofline baseline; Nova already measures this | Low | Leverage v2.7 bandwidth tool |
| Arithmetic intensity calculation | Core of Roofline: FLOPs / bytes accessed | Low | Per-kernel analysis |
| Performance limiter classification | Math-limited vs memory-limited diagnosis | Low | Compare intensity to ops:byte ratio |
| Roofline plot data export | JSON format for external visualization | Low | Integrate with v2.7 dashboard |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-kernel analysis pipeline | Identify bottlenecks across kernel suite | Medium | Use existing kernel stats from v2.7 |
| Device-specific ops:byte table | Pre-computed for supported architectures | Low | SM 6.0-9.0 values |
| AI recommendation engine | Suggest optimizations based on limiter | Medium | Memory-bound → increase arithmetic intensity |
| Multi-precision analysis | FP64/FP32/FP16/INT8 ceilings | Medium | Different peaks per precision |

### Anti-Features (Explicitly NOT Building)

| Anti-Feature | Why Avoid | What To Do Instead |
|--------------|-----------|-------------------|
| Full visualization library | Scope creep | Export data for existing tools (matplotlib, plotly) |
| Automatic optimization suggestions | Complex/fragile | Provide data; let user interpret |
| Real-time monitoring | Over-engineering for v2.8 | Batch analysis mode sufficient |

### Roofline Mathematics

**Key Formula (from NVIDIA GPU Performance Guide):**

```
Arithmetic Intensity (AI) = FLOPs / Bytes Accessed

ops:byte ratio = Peak Math FLOP/s / Peak Memory Bandwidth

If AI > ops:byte ratio → Math-limited
If AI < ops:byte ratio → Memory-limited
```

**Example Device Values (A100):**
| Precision | Peak Math | Memory Bandwidth | ops:byte |
|-----------|-----------|------------------|----------|
| FP64 | 9.7 TFLOPS | 2.0 TB/s | 4.85 |
| FP32 | 19.5 TFLOPS | 2.0 TB/s | 9.75 |
| TF32 | 156 TFLOPS | 2.0 TB/s | 78 |
| FP16 | 312 TFLOPS | 2.0 TB/s | 156 |

**Algorithm Examples:**
| Operation | Typical AI | Limit |
|-----------|------------|-------|
| SpMV (sparse) | 0.5-2 FLOPS/B | Memory |
| GEMM (large) | 100+ FLOPS/B | Math |
| GEMM (small) | <5 FLOPS/B | Memory |
| ReLU | 0.25 FLOPS/B | Memory |

---

## 3. Sparse Format Features

### Category Overview

**Type:** Data structure and operations
**Complexity:** Low-Medium (formats) / Medium-High (conversion)
**Dependencies:** Existing CSR/CSC (v2.1), SpMV (v2.1)
**Library Status:** cuSPARSE supports SELL, Blocked-ELL, BSR; ELL/HYB need custom implementation

### Table Stakes Features

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| ELL format storage | Efficient for regular sparse matrices | Low | Padding overhead for irregular matrices |
| ELL SpMV kernel | Core operation for ELL format | Medium | Coalesced memory access |
| SELL format storage | Better than ELL for irregular matrices | Low | Slice-based organization |
| SELL SpMV kernel | Performance-critical operation | Medium | Slightly more complex than ELL |
| Format conversion: CSR→ELL | Common workflow | Medium | Analyze row-length distribution first |
| Format conversion: CSR→SELL | Common workflow | Medium | Choose slice size parameter |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| HYB format (ELL+COO) | Best of both worlds for mixed patterns | High | Complex conversion and SpMV |
| Format auto-selection | Choose best format based on matrix analysis | Medium | Analyze nnz/row variance |
| BSR format support | Structured sparsity patterns | Medium | cuSPARSE has native support |
| SpMV kernel benchmarks | Characterize format performance | Medium | Different formats optimal for different matrices |

### Anti-Features (Explicitly NOT Building)

| Anti-Feature | Why Avoid | What To Do Instead |
|--------------|-----------|-------------------|
| DIA (Diagonal) format | Rarely optimal; premature | Defer if real need emerges |
| JAD (JDS) format | Complex; limited use cases | Defer |
| All format conversions | Combinatorial explosion | Focus on CSR→ELL, CSR→SELL, CSR→HYB |

### Format Comparison

| Format | Best For | Memory Overhead | SpMV Performance |
|--------|----------|-----------------|------------------|
| CSR | General sparse; current implementation | Minimal | Good |
| CSC | Column-wise operations | Minimal | Good |
| ELL | Regular rows (similar nnz/row) | Padding for irregular rows | Excellent if regular |
| SELL | Semi-irregular matrices | Less padding than ELL | Good balance |
| HYB | Mixed regular/irregular rows | ELL + COO overhead | Good for irregular |
| BSR | Structured 2D block patterns | Padding for partial blocks | Excellent for blocks |

### ELL Format Structure

```
Standard ELL: Each row padded to max_nnz
Row 0: [a00, a01, a02, -, -]  (max=3)
Row 1: [a10, a11, -, -, -]
Row 2: [a20, a21, a22, a23, -]
Row 3: [a30, a31, -, -, -]

Storage: values[rows * max_nnz], col_idx[rows * max_nnz]
```

### SELL Format Structure

```
SELL: Matrix divided into slices of sliceSize rows
Each slice padded to max nnz within that slice

Slice 0 (rows 0-3): max_nnz = 4
Slice 1 (rows 4-7): max_nnz = 2

Better memory balance than full ELL for varying row lengths
```

### HYB Format Structure

```
HYB: Hybrid ELL + COO
- ELL portion: Stores regular rows efficiently
- COO portion: Handles irregular rows that would waste ELL space
- Partition point: Determined by analyzing row nnz distribution

Conversion: Rows with nnz <= threshold → ELL
            Rows with nnz > threshold → COO
```

---

## 4. Complexity Assessment

### By Category

| Category | Implementation | Testing | Total Risk |
|----------|---------------|---------|------------|
| Krylov Solvers | Medium-High | Medium | High |
| Roofline Model | Low-Medium | Low | Low |
| Sparse Formats | Medium | Medium | Medium |

### Krylov Solver Complexity Breakdown

| Component | Complexity | Notes |
|-----------|------------|-------|
| CG solver basic | Medium | ~200 lines kernel + wrapper |
| CG with Jacobi preconditioner | Medium-High | +100 lines |
| GMRES solver | High | Arnoldi iteration, restart logic |
| Convergence testing | Low | Simple norm comparisons |
| Memory management | Medium | Pool integration, temp buffers |

### Sparse Format Complexity Breakdown

| Component | Complexity | Notes |
|-----------|------------|-------|
| ELL storage class | Low | Simple padding structure |
| ELL SpMV kernel | Medium | Warp-based row processing |
| SELL SpMV kernel | Medium | Slice-aware processing |
| CSR→ELL conversion | Medium | Row analysis + padding |
| CSR→SELL conversion | Medium | Slice organization |
| HYB format + SpMV | High | Dual-kernel approach |

---

## 5. Feature Dependencies

```
Existing (v2.1-v2.7):
├── SpMV (CSR/CSC) ──────────────┐
├── cuBLAS dot/scale/axpy ───────┼── Krylov Solver foundation
├── Bandwidth measurement ───────┤
└── Kernel statistics ───────────┘    └──

v2.8 Features:
├── Roofline Model
│   ├── Device properties (existing)
│   ├── Bandwidth measurement (v2.7) ───┐
│   └── Kernel statistics (v2.7) ───────┤── Roofline data sources
│                                       │
├── Krylov Solvers                      │
│   ├── SpMV (v2.1) ────────────────────┼── Solver kernel
│   ├── cuBLAS operations ───────────────┤── BLAS primitives
│   └── ELL/HYB SpMV (planned v2.8) ────┘── Format-specific kernels
│
└── Sparse Formats
    ├── CSR storage (v2.1) ──────────────┘
    ├── ELL storage + SpMV ──────────────┐── Format implementations
    ├── SELL storage + SpMV ─────────────┤
    └── HYB conversion + SpMV ───────────┘
```

---

## 6. Sources

- **cuSOLVER Documentation** (v13.2): https://docs.nvidia.com/cuda/cusolver/index.html
  - cuSolverSP deprecated; cuDSS successor noted
  - Iterative refinement uses GMRES internally
- **cuSPARSE Documentation** (v13.2): https://docs.nvidia.com/cuda/cusparse/index.html
  - SELL, Blocked-ELL, BSR formats documented
  - Format conversion APIs available
- **cuDSS Documentation** (Preview): https://docs.nvidia.com/cuda/cudss/index.html
  - Direct solver replacement for cuSolverSP
  - Multi-GPU, multi-node support
  - Analysis/Symbolic/Numeric/Solve phases
- **NVIDIA GPU Performance Background**: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
  - Arithmetic intensity definition
  - ops:byte ratio concept
  - Performance limiter classification
- **NVIDIA Matrix Multiplication Guide**: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
  - GEMM arithmetic intensity calculation
  - Device-specific peak values

---

*Last updated: 2026-05-01*
*Prepared for: v2.8 Planning*
