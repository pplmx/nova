#include "cuda/device/cublas_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo {

// Matrix multiplication C = A * B using cuBLAS.
// Matrix dimensions: A(mxk), B(kxn), C(mxn), all ROW-MAJOR.
//
// cuBLAS is column-major only. Reading row-major A as if it were column-major
// means you are really holding A^T, so C = A*B has to be expressed as
//
//     C^T = B^T * A^T
//
// which is one cuBLAS call with the two operands SWAPPED and the result
// dimensions swapped: (n, m, k) with lda = n, ldb = k, ldc = m.
//
// The previous version passed the operands in order with lda = k, ldb = n,
// ldc = n. For m=2,n=4,k=3 that produced {4,5,0,0,4,5,0,0} where the correct
// rowwise product is {0,6,0,6,0,15,0,15}; and whenever m > n it passed
// ldc = n < m, which cuBLAS rejects outright with "parameter number 10 had an
// illegal value".
void matrixMultiply(const memory::Buffer<float>& a, const memory::Buffer<float>& b, memory::Buffer<float>& c, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    device::CublasContext ctx;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(ctx.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                             /*m=*/n, /*n=*/m, /*k=*/k,
                             &alpha,
                             b.data(), /*lda=*/n,
                             a.data(), /*ldb=*/k,
                             &beta,
                             c.data(), /*ldc=*/n));
}

// Double-precision matrix multiplication - same logic as float version above
// Uses cublasDgemm for double precision arithmetic
void matrixMultiply(const memory::Buffer<double>& a, const memory::Buffer<double>& b, memory::Buffer<double>& c, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    device::CublasContext ctx;

    const double alpha = 1.0;
    const double beta = 0.0;

    // Same row-major/column-major swap as the float overload.
    CUBLAS_CHECK(cublasDgemm(ctx.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                             /*m=*/n, /*n=*/m, /*k=*/k,
                             &alpha,
                             b.data(), /*lda=*/n,
                             a.data(), /*ldb=*/k,
                             &beta,
                             c.data(), /*ldc=*/n));
}

}  // namespace cuda::algo
