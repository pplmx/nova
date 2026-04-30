# Domain Pitfalls: v2.8 Numerical Computing & Performance

**Research Date:** 2026-05-01
**Overall Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: GMRES Workspace Accumulation

**What goes wrong:** GMRES restarts create new orthogonal basis vectors, leading to O(k²n) memory growth across restarts if not managed.

**Why it happens:** Each GMRES restart stores the full Krylov basis (size `restart × n`), and naive implementation allocates new vectors per restart.

**Consequences:** 
- OOM on large systems with frequent restarts
- Performance degradation from memory pressure
- Potential memory fragmentation

**Prevention:**
```cpp
class KrylovContext {
    memory::Buffer<T> workspace_;  // Preallocated, fixed size
    size_t restart_;               // Fixed restart parameter
    
    void ensure_capacity(int n) {
        // Allocate once: restart × n vectors
        // Reuse across restarts
    }
};
```

**Detection:** Monitor peak memory during GMRES iterations; unexpected growth indicates reallocation.

---

### Pitfall 2: Roofline Misattribution (Memory-bound vs Compute-bound)

**What goes wrong:** Roofline analysis incorrectly labels kernels as compute-bound when they're actually memory-bound (or vice versa).

**Why it happens:** 
- Incorrect FLOP counting (includes non-compute operations)
- Missing memory traffic (ignores reads-again, redundant loads)
- Wrong peak FLOP/s assumption (using FP64 peak for FP32 kernels)

**Consequences:**
- Misguided optimization efforts
- Performance cliff when actual bottleneck differs
- Wasted engineering time on wrong targets

**Prevention:**
```cpp
RooflinePoint analyze_kernel(const KernelStats& stats, 
                             const DeviceProperties& props) {
    double actual_flops = count_actual_flops(stats.kernel_name);
    double actual_bytes = estimate_memory_bytes(stats);
    
    // Use correct peak for kernel type
    double peak_flops = get_kernel_peak_flops(stats.kernel_name, props);
    
    // ...
}
```

**Detection:** Verify roofline points fall near expected ridge point; outliers indicate counting errors.

---

### Pitfall 3: ELL Format Padding Overhead

**What goes wrong:** ELL format with high max_nnz creates excessive padding, wasting memory and potentially hurting performance.

**Why it happens:** ELL pads all rows to `max_row_nnz`, which can be 10-100× the average for irregular matrices.

**Consequences:**
- Memory usage explosion for skewed distributions
- Register pressure in SpMV kernel
- Worse performance than CSR for highly irregular matrices

**Prevention:**
```cpp
// Only use ELL when matrix is regular
if (variance_coefficient < 0.3 && max_nnz / avg_nnz < 3.0) {
    return SparseFormat::ELL;
}
// Otherwise use HYB or keep CSR
```

**Detection:** Log ELL padding ratio (`max_nnz / avg_nnz`); values > 5.0 indicate poor ELL suitability.

---

## Moderate Pitfalls

### Pitfall 4: Krylov Convergence on Ill-Conditioned Systems

**What goes wrong:** Solvers fail to converge or converge slowly on poorly conditioned matrices.

**Why it happens:** Without preconditioning, condition number directly impacts convergence rate. CG requires SPD matrices; wrong type causes divergence.

**Prevention:**
- Validate matrix properties before solver selection (SPD for CG)
- Add configurable tolerance and iteration limits
- Document required matrix properties in API

### Pitfall 5: SpMV Kernel Selection Mismatch

**What goes wrong:** Wrong SpMV kernel selected for format, causing correctness issues or performance degradation.

**Why it happens:** CSR SpMV kernel fails silently on ELL data, and vice versa.

**Prevention:**
- Use static dispatch based on format enum
- Validate format at construction, not runtime
- Add format tag to kernel launch

### Pitfall 6: NVTX Overhead in Tight Loops

**What goes wrong:** NVTX annotations inside Krylov iteration loops cause measurable overhead.

**Why it happens:** NVTX mark/range calls add kernel launch overhead, which compounds in iterative loops.

**Prevention:**
- Move NVTX ranges to outer loops only
- Use compile-time flag to disable in production
- Profile with and without NVTX to measure impact

---

## Minor Pitfalls

### Pitfall 7: CSR-to-ELL Conversion O(n²) Complexity

**What goes wrong:** Naive CSR-to-ELL conversion uses O(n²) scanning for max_nnz.

**Prevention:** Single pass to find max_nnz during construction.

### Pitfall 8: Roofline Peak Assumption Mismatch

**What goes wrong:** Using aggregate device peak instead of per-kernel achievable peak.

**Prevention:** Use kernel-specific theoretical limits (tensor FLOP/s vs. FP64 FLOP/s).

### Pitfall 9: HYB Format Threshold Sensitivity

**What goes wrong:** Hardcoded ELL threshold in HYB conversion causes suboptimal results.

**Prevention:** Make threshold configurable; expose `format_analyzer` recommendation.

### Pitfall 10: BiCGSTAB Numerical Instability

**What goes wrong:** BiCGSTAB breakdown when (r, r̄) ≈ 0.

**Prevention:** Detect breakdown conditions; fall back to GMRES.

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|----------------|------------|
| **1: Sparse Formats** | ELL padding overhead (Pitfall 3) | Use format_analyzer for selection |
| **1: Sparse Formats** | SpMV kernel mismatch (Pitfall 5) | Static dispatch with format tag |
| **2: Krylov Solvers** | Workspace accumulation (Pitfall 1) | Preallocate in KrylovContext |
| **2: Krylov Solvers** | Convergence on ill-conditioned (Pitfall 4) | Validate matrix, add preconditioning |
| **3: Roofline Model** | Misattribution (Pitfall 2) | Verify FLOP/byte counting |
| **3: Roofline Model** | NVTX overhead (Pitfall 6) | Outer-loop annotation only |
| **4: Integration** | Threshold sensitivity (Pitfall 9) | Expose as configuration |
| **4: Integration** | BiCGSTAB instability (Pitfall 10) | Implement breakdown detection |

---

## Validation Checklist

Before shipping each phase:

- [ ] Memory profiling shows stable allocation (no growth in loops)
- [ ] Roofline analysis validated against known kernels
- [ ] ELL format only used when variance_coefficient < threshold
- [ ] NVTX overhead measured and acceptable
- [ ] Solver convergence tested on ill-conditioned matrices
- [ ] Format conversion tested for round-trip accuracy
- [ ] SpMV results validated against CSR reference

---

*Pitfalls research complete: 2026-05-01*
