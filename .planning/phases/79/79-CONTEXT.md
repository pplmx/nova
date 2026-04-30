# Phase 79: Sparse Format Foundation - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can store sparse matrices in ELL/SELL formats, convert from CSR, and perform SpMV operations. This phase builds on existing CSR/CSC infrastructure from Phase 76, adding two additional sparse matrix formats commonly used in iterative solvers and finite element methods.

</domain>

<decisions>
## Implementation Decisions

### ELL Format API
- Follow existing SparseMatrixCSR pattern with values_, col_indices_, padded_row_offsets_ members
- Factory method FromCSR() for conversion from existing CSR matrices
- Constructor accepting pre-padded data for advanced users
- Accessors: values(), col_indices(), row_ptr() (padded offsets), num_rows(), num_cols(), nnz(), padded_nnz()

### ELL Padding Strategy
- Auto-calculate max_nnz_per_row from input CSR (max over all rows)
- Store padded_nnz (total padded storage) alongside actual nnz for memory reporting
- Public accessor max_nnz_per_row() for diagnostics

### SELL Format API
- Slice height C (configurable, default 32 matching warp size)
- Similar storage to ELL but row_ptr stores offsets per slice rather than per row
- Factory method FromCSR() with slice_height parameter
- Public accessor slice_height() for diagnostics

### SpMV Operations
- CPU-only implementation in this phase (GPU kernels can follow in Phase 81 with HYB format)
- Match existing SparseOps::spmv pattern with static method for ELL and SELL
- Numerical accuracy verification against CSR baseline

### the agent's Discretion
All implementation details follow existing codebase patterns and conventions. No user-facing API design decisions required — pure infrastructure phase.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- SparseMatrixCSR<T> class with FromDense factory method
- SparseMatrixCSC<T> class with FromCSR conversion
- SparseOps::spmv, SparseOps::spmm static methods
- sparse_mv(), sparse_mm() free functions

### Established Patterns
- Member storage: std::vector<T> values_, std::vector<int> offsets_, int dimensions
- Factory methods: static std::optional<T> FromX(), static T FromY()
- Getters: num_rows(), num_cols(), nnz()
- Accessors: values(), row_offsets(), etc. (const and mutable)
- Test class: TEST_F(SparseMatrixTest, Name) with create_dense helper

### Integration Points
- Include files: include/cuda/sparse/sparse_matrix.hpp, include/cuda/sparse/sparse_ops.hpp
- Test file: tests/sparse/sparse_matrix_test.cpp
- namespace nova::sparse

</code_context>

<specifics>
## Specific Ideas

No specific requirements — follow standard approaches:
- ELLPACK format: pad each row to max_nnz_per_row
- SELL-C-σ format: group C rows into slices, pad each slice to slice's max
- SpMV returns same results as CSR baseline (numerical verification)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
