---
phase: 41
phase_name: Property-Based Tests
status: planning
created: 2026-04-26
requirements:
  - PROP-01
  - PROP-02
  - PROP-03
  - PROP-04
---

# Phase 41: Property-Based Tests - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning
**Mode:** Autonomous (from ROADMAP.md)

## Phase Boundary

Implement QuickCheck-style tests verifying mathematical and algorithmic properties.

## Implementation Decisions

### Approach
Implement hand-rolled property-based testing framework (avoiding external dependency on rapidcheck). Build on input generation patterns from Phase 40 fuzzing.

### Property Categories

1. **Mathematical Invariants**
   - Matmul identity: A @ I = A
   - FFT inverse: FFT⁻¹(FFT(x)) ≈ x
   - Transpose: (A^T)^T = A

2. **Algorithmic Correctness**
   - Sort produces sorted output
   - Reduce is associative
   - Scan produces correct prefix sums

3. **Numerical Stability**
   - FP16/FP32/FP64 consistency
   - No NaN/Inf without cause

### Seed Reproducibility
Store test seeds in metadata for exact reproduction of failures.

## Specific Ideas

### PROP-01: Mathematical Invariants
- Matmul identity property
- FFT inverse property
- Transpose involution

### PROP-02: Algorithmic Correctness
- Sort correctness
- Reduce associativity
- Scan prefix correctness

### PROP-03: Numerical Stability
- Precision mode tests
- NaN/Inf detection

### PROP-04: Reproducible Seeds
- Seed logging
- Seed-based test replay

---

*Context generated for Phase 41: Property-Based Tests*
