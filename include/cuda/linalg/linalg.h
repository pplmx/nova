#pragma once

/**
 * @file linalg.h
 * @brief GPU linear algebra operations: SVD, eigenvalue decomposition, matrix factorization
 * @author Nova CUDA Library
 * @version 2.3
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

    void svd(const float* A, size_t m, size_t n, SVDResult& result, SVDMode mode = SVDMode::Thin);
    void eigenvalue_decomposition(const float* A, size_t n, EVDResult& result);
    void qr_decomposition(const float* A, size_t m, size_t n, QRResult& result);
    void cholesky_decomposition(const float* A, size_t n, CholeskyResult& result);

}  // namespace cuda::linalg
