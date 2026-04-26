---
phase: 44
phase_name: Sparse Matrix Support
status: planning
created: 2026-04-26
requirements:
  - SPARSE-01
  - SPARSE-02
  - SPARSE-03
  - SPARSE-04
---

# Phase 44: Sparse Matrix Support - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning
**Mode:** Autonomous (from ROADMAP.md)

## Phase Boundary

Add CSR/CSC sparse matrix formats and sparse operations (SpMV, SpMM).

## Implementation Decisions

### Formats
- CSR (Compressed Sparse Row) as primary format
- CSC (Compressed Sparse Column) for column-wise operations
- cuSPARSE integration when available, fallback otherwise

### Operations
- SpMV: Sparse Matrix-Vector Multiplication
- SpMM: Sparse Matrix-Dense Matrix Multiplication

## Specific Ideas

### SPARSE-01: CSR Format
- SparseMatrixCSR class with values, row_offsets, col_indices
- FromDense() factory method

### SPARSE-02: CSC Format  
- SparseMatrixCSC class with values, col_offsets, row_indices
- ToCSC() conversion from CSR

### SPARSE-03: SpMV
- sparse_mv() function for y = A * x

### SPARSE-04: SpMM
- sparse_mm() function for C = A * B

---

*Context generated for Phase 44: Sparse Matrix Support*
