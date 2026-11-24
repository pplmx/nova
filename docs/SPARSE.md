# Sparse Matrix Operations

Nova provides comprehensive sparse matrix support including multiple storage formats, SpMV operations, and iterative solvers.

## Supported Formats

| Format   | Use Case           | Pros             | Cons                   |
| -------- | ------------------ | ---------------- | ---------------------- |
| **CSR**  | General purpose    | Simple, flexible | Slower SpMV            |
| **ELL**  | Regular sparsity   | Fast SpMV        | Wasteful for irregular |
| **SELL** | Irregular sparsity | Balanced         | More complex           |
| **HYB**  | Mixed patterns     | Adaptive         | Most complex           |

## Quick Start

### Creating a Sparse Matrix

```cpp
#include "cuda/sparse/matrix.hpp"

std::vector<float> dense = /* your data */;
int rows = 1000, cols = 1000;

// Create from dense (automatically detects zeros)
auto matrix = nova::sparse::SparseMatrix<float>::FromDense(
    dense.data(), rows, cols, 1e-6f);

// Or from explicit CSR data
auto csr_matrix = nova::sparse::SparseMatrix<float>::FromHostData(
    values, row_offsets, col_indices, rows, cols);
```

### Sparse Matrix-Vector Multiply

```cpp
#include "cuda/sparse/sparse_ops.hpp"

// Allocate input/output vectors
cuda::memory::Buffer<float> d_x(cols);
cuda::memory::Buffer<float> d_y(rows);

// Copy input data
d_x.copy_from(host_x.data(), cols);

// Perform SpMV: y = A * x
nova::sparse::spmv(*matrix, d_x.data(), d_y.data());

// Copy result back
std::vector<float> host_y(rows);
d_y.copy_to(host_y.data(), rows);
```

### Converting Between Formats

```cpp
#include "cuda/sparse/sparse_matrix.hpp"

// CSR (already have)
// SparseMatrixCSR<float> csr = ...;

// Convert to ELL (good for regular patterns)
auto ell = nova::sparse::SparseMatrixELL<float>::FromCSR(csr);

// Convert to SELL (good for irregular patterns)
auto sell = nova::sparse::SparseMatrixSELL<float>::FromCSR(csr, 32);  // slice_height=32
```

## Iterative Solvers

Nova provides Krylov subspace solvers for sparse linear systems Ax = b.

### Conjugate Gradient (CG)

For symmetric positive definite matrices:

```cpp
#include "cuda/sparse/krylov.hpp"

nova::sparse::SolverConfig<float> config{
    .relative_tolerance = 1e-6f,
    .max_iterations = 1000,
    .verbose = true
};

nova::sparse::ConjugateGradient<float> solver(config);
nova::sparse::SolverResult<float> result = solver.solve(A, b, x);

if (result.converged) {
    std::cout << "Converged in " << result.iterations << " iterations\n";
    std::cout << "Final residual: " << result.relative_residual << "\n";
}
```

### GMRES (Generalized Minimal Residual)

For non-symmetric matrices:

```cpp
auto result = nova::sparse::gmres(A, b, x, {
    .max_iterations = 500,
    .restart = 50,        // Restart every 50 iterations
    .tolerance = 1e-6f
});
```

### Preconditioners

Improve convergence with preconditioners:

```cpp
// Jacobi (diagonal) preconditioner - simplest, good for diagonal-dominant.
// The constructor takes the damping factor (default 1.0); setup() is the call that
// factorises. There is no compute() method.
nova::sparse::JacobiPreconditioner<float> jacobi;
jacobi.setup(A);

auto result = nova::sparse::gmres(A, b, x, config, &jacobi);

// There is no ILU(0) preconditioner to use here. nova::sparse::ILUPreconditioner
// exists, but its setup() and apply() both throw PreconditionerError stating that
// the ILU(0) factorization and triangular solve are not implemented -- the class
// throws rather than returning an unfactorized buffer, which is the honest
// behaviour, but it is not usable as a preconditioner yet. Use
// JacobiPreconditioner above, or drive the cuSPARSE csrsm2 path behind the
// Preconditioner interface directly.
```

## Performance Tips

1. **Choose the right format**: Use ELL for regular sparsity (~30% faster SpMV), SELL for irregular patterns

2. **Reorder for better locality**: RCM reordering reduces bandwidth:

   ```cpp
   nova::sparse::RCMReorderer<float> reorderer;
   auto reorder_result = reorderer.reorder(A);
   // Use reorder_result.permutation to reorder A
   ```

3. **Use preconditioners**: Can reduce iterations by 2-10x

4. **Tune tolerance**: Tight tolerances (1e-8) may not converge faster; 1e-6 is often sufficient

## Error Handling

All operations throw on error with descriptive messages:

```cpp
try {
    nova::sparse::spmv(*matrix, d_x.data(), d_y.data());
} catch (const std::exception& e) {
    std::cerr << "SpMV failed: " << e.what() << "\n";
}
```

## See Also

- [API Documentation](../include/cuda/sparse/) - Detailed API reference
- [Matrix Format Reference](../include/cuda/sparse/sparse_matrix.hpp) - Format details
- [Solver Reference](../include/cuda/sparse/krylov.hpp) - Solver API
