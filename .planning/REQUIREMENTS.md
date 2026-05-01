# Requirements — v2.10 Sparse Solver Acceleration

## Preconditioning (PRECOND)

### PRECOND-01
**User can:** Apply Jacobi preconditioner with configurable relaxation parameter

**Details:**
- Extract diagonal from SparseMatrix and compute reciprocal
- Implement weighted Jacobi with parameter ω (default 1.0)
- GPU-accelerated apply using existing SpMV infrastructure
- Validate non-zero diagonal entries with descriptive error

### PRECOND-02
**User can:** Use unified Preconditioner interface

**Details:**
- Abstract base class with virtual setup() and apply() methods
- Concrete implementations: JacobiPreconditioner, ILUPreconditioner
- RAII ownership via unique_ptr

### PRECOND-03
**User can:** Reorder sparse matrices using RCM bandwidth reduction

**Details:**
- BFS-based Reverse Cuthill-McKee algorithm
- Compute permutation vector for matrix reordering
- Support in-place and out-of-place reordering
- Validate ordering quality (bandwidth reduction ratio)

### PRECOND-04
**User can:** Apply ILU(0) incomplete factorization preconditioner

**Details:**
- Use cuSPARSE csrilu0 for GPU-accelerated ILU setup
- Forward solve + backward solve for apply
- Monitor fill-in ratio (nnz(L+U) / nnz(A))
- Handle zero pivot gracefully with descriptive error

---

## Solver Integration (SOLVER)

### SOLVER-01
**User can:** Use CG solver with preconditioner

**Details:**
- Extend ConjugateGradient class with set_preconditioner()
- Apply preconditioner in each iteration (left preconditioning)
- Profile and document expected iteration count reduction

### SOLVER-02
**User can:** Use GMRESGPU solver with preconditioner

**Details:**
- Extend GMRESGPU class with set_preconditioner()
- Apply preconditioner to residual each iteration
- Maintain restart capability with preconditioner

### SOLVER-03
**User can:** Use BiCGSTAB solver with preconditioner

**Details:**
- Extend BiCGSTAB class with set_preconditioner()
- Apply preconditioner to both residual and auxiliary vectors
- Ensure numerical stability for non-SPD matrices

---

## Testing (TEST)

### TEST-01
**User can:** Verify Jacobi preconditioner correctness

**Details:**
- Test on identity, diagonal, and known SPD matrices
- Verify numerical accuracy against CPU reference
- Test weighted variant with various ω values
- Test error handling for zero diagonal

### TEST-02
**User can:** Verify ILU preconditioner correctness

**Details:**
- Test on known matrices with analytical solution
- Verify fill-in ratio within expected bounds
- Test error handling for singular matrices
- Compare with cuSPARSE reference (if available)

### TEST-03
**User can:** Verify RCM ordering correctness

**Details:**
- Test on known graph structures (paths, grids, random)
- Verify bandwidth reduction on test matrices
- Test permutation invertibility (P * P' = I)
- Benchmark ordering time complexity

### TEST-04
**User can:** Verify E2E convergence improvement

**Details:**
- Test on ill-conditioned matrices (condition number > 10^6)
- Compare iteration count: no preconditioner vs Jacobi vs ILU
- Verify final solution meets convergence tolerance

### TEST-05
**User can:** Measure preconditioner performance impact

**Details:**
- Benchmark setup time (Jacobi: O(n), ILU: O(nnz))
- Benchmark apply time per iteration
- Compare total solve time (setup + iterations)
- Document crossover point where preconditioner helps

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRECOND-01 | 88 | — |
| PRECOND-02 | 88 | — |
| PRECOND-03 | 89 | — |
| PRECOND-04 | 90 | — |
| SOLVER-01 | 91 | — |
| SOLVER-02 | 91 | — |
| SOLVER-03 | 91 | — |
| TEST-01 | 92 | — |
| TEST-02 | 92 | — |
| TEST-03 | 92 | — |
| TEST-04 | 92 | — |
| TEST-05 | 92 | — |

---

## Out of Scope

- **AMG (Algebraic Multigrid)** — Too complex, future milestone
- **ILU(k) with k > 0** — ILU(0) only for this milestone
- **Minimum Degree ordering** — RCM only for this milestone
- **Right preconditioning** — Left preconditioning only
- **SuperLU/MUMPS integration** — CPU-based, out of scope
