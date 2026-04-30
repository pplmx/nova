# Research Summary: v2.8 Numerical Computing & Performance

**Domain:** CUDA numerical computing library extension (Krylov solvers, Roofline model, advanced sparse formats)
**Researched:** 2026-05-01
**Overall Confidence:** HIGH

---

## Executive Summary

The v2.8 milestone adds three production-quality features to Nova's CUDA library. Analysis of the existing five-layer architecture reveals clean integration points:

1. **Krylov Solvers** extend the Algorithm layer, building on existing SpMV primitives from `algo/spmv.h` and memory buffers from `memory/buffer.h`. Three solver variants (CG, GMRES, BiCGSTAB) share common workspace infrastructure.

2. **Roofline Model** extends the Observability layer, integrating with `kernel_stats.h` and `bandwidth_tracker.h`. The model computes operational intensity vs. device peaks to identify compute vs. memory bottlenecks.

3. **ELL/HYB Formats** extend the existing Sparse module, adding format-specific SpMV kernels and conversion utilities from CSR. Pattern analysis enables automatic format selection.

All three features use NVTX domains for profiling and fit within Nova's RAII memory management and stream-based async patterns.

---

## Key Findings

| Feature | Primary Integration | New Components | Dependencies |
|---------|---------------------|----------------|--------------|
| Krylov Solvers | Algorithm layer | 7 new files in `solvers/` | SpMV, memory buffers, reduce |
| Roofline Model | Observability layer | 1 new file + extend existing | kernel_stats, device_info, bandwidth |
| ELL/HYB Formats | Sparse module | 2 new files + extend existing | CSR SpMV, format converter |

### Integration Complexity Assessment
- **Krylov**: LOW complexity — leverages existing SpMV, minimal new interfaces
- **Roofline**: LOW complexity — additive observability, post-hoc analysis
- **ELL/HYB**: MEDIUM complexity — new format classes, conversion utilities, format selection logic

---

## Suggested Phase Structure

### Phase 1: Sparse Format Foundation (Lowest Risk)
- Add ELL/HYB classes to `sparse_matrix.hpp`
- Create `format_converter.hpp` and `format_analyzer.hpp`
- Add ELL/HYB SpMV to `sparse_ops.hpp`
- **Rationale:** Pure extension, no behavioral changes to existing code

### Phase 2: Krylov Solver Core (Medium Risk)
- Create workspace management (`krylov_context.h`)
- Implement CG/GMRES/BiCGSTAB in separate headers
- Create unified `krylov.h` entry point
- **Rationale:** Depends on Phase 1 SpMV, but internally consistent

### Phase 3: Roofline Model (Independent)
- Add `observability/roofline.h`
- Extend `kernel_stats.h` for FLOP/byte tracking
- **Rationale:** Fully independent, can run in parallel with Phase 2

### Phase 4: Integration & Production
- Update `algo_wrapper.h` with solver wrapping
- Integrate roofline into profiler output
- Add E2E tests and benchmarks

---

## Research Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Architecture | HIGH | Clean integration with existing five-layer design |
| Krylov Design | HIGH | Standard numerical patterns, well-understood |
| Roofline Design | HIGH | Existing observability infrastructure, standard model |
| ELL/HYB Design | HIGH | Follows existing sparse module patterns |
| Build Dependencies | MEDIUM | Clear from analysis, but parallel workstreams may enable optimization |
| API Surface | MEDIUM | Follows Nova patterns, may need adjustment based on feedback |

---

## Gaps to Address

1. **Preconditioner design** — ILU/Jacobi preconditioners for GMRES/BiCGSTAB not specified; may need additional research
2. **Format-specific SpMV kernels** — CUDA kernels for ELL/HYB SpMV not detailed; may need CUDA-specific research
3. **Multi-GPU Krylov** — Not scoped for v2.8, but architecture should support future distributed solvers
4. **Roofline visualization** — Post-processing script not specified; may need tooling research

---

*Summary complete: 2026-05-01*
