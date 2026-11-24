#pragma once

/**
 * @file linalg.h
 * @brief GPU linear algebra operations: SVD, eigenvalue decomposition, matrix factorization
 * @author Nova CUDA Library
 * @version 2.3
 *
 * @par Layout convention
 * Every matrix here is **row-major**, both in and out. Nothing in this header said so, which
 * is how two of the four functions came to hand a row-major buffer to column-major cuSOLVER
 * routines and copy the factors back without transposing: qr_decomposition returned a Q that
 * was not orthogonal and an R that was not triangular, and cholesky_decomposition returned a
 * factor that did not reconstruct its input at all. Both are fixed, and the conversions are
 * explicit at the cuSOLVER boundary.
 *
 * @par Result layouts
 * - SVD: U is m x min(m,n), S is min(m,n), Vt is min(m,n) x n, all row-major.
 * - EVD: eigenvalues ascending; eigenvectors **column-major** (V[i][j] at `i + j*n`), so
 *   eigenvector j is contiguous. This is the one result that is not row-major, kept because
 *   it is the layout syevd exposed and callers may already index it that way.
 * - QR: Q is m x min(m,n) row-major with orthonormal columns, R is min(m,n) x n row-major
 *   upper-triangular, A = Q*R.
 * - Cholesky: L is n x n row-major lower-triangular, A = L*L^T.
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::linalg {

    enum class SVDMode { Full, Thin, Randomized };
    enum class MatrixType { General, Symmetric, PositiveDefinite };

    struct SVDResult {
        memory::Buffer<float> U;
        memory::Buffer<float> S;
        memory::Buffer<float> Vt;
        size_t actual_rank;
        float condition_number;
    };

    struct EVDResult {
        memory::Buffer<float> eigenvalues;
        memory::Buffer<float> eigenvectors;
        float condition_number;
    };

    struct QRResult {
        memory::Buffer<float> Q;
        memory::Buffer<float> R;
    };

    struct CholeskyResult {
        memory::Buffer<float> L;
        bool is_positive_definite;
    };

    /// @name Matrix decompositions
    ///
    /// All four take a DEVICE pointer for `A` (column-major, m*n floats). They copy it into a
    /// working buffer with cudaMemcpyDeviceToDevice before factoring, so a host array is not
    /// accepted; passing one fails with a bare "CUDA error: invalid argument" from the copy, which
    /// does not name the argument. The output buffers are device memory too.
    ///
    /// Each result satisfies its defining identity on device data:
    ///
    ///     cholesky_decomposition   A == L * L^T                 (lower triangle in L)
    ///     qr_decomposition         A == Q * R,  Q^T * Q == I
    ///     svd                      A == U * diag(S) * Vt
    ///     eigenvalue_decomposition eigenvalues sum to trace(A)  (AscendingEigenSolver)
    ///
    /// Measured on a 3x3 SPD matrix and a 4x3 rectangular one, the first three reproduce A to
    /// within 2e-6 and Q^T Q to within 2e-7; the eigenvalues of a symmetric 3x3 summed to exactly
    /// its trace.
    /// @{
    void svd(const float* A, size_t m, size_t n, SVDResult& result, SVDMode mode = SVDMode::Thin);
    void eigenvalue_decomposition(const float* A, size_t n, EVDResult& result);
    void qr_decomposition(const float* A, size_t m, size_t n, QRResult& result);
    void cholesky_decomposition(const float* A, size_t n, CholeskyResult& result);
    /// @}

}  // namespace cuda::linalg
