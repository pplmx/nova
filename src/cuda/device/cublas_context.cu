#include "cuda/device/cublas_context.h"
#include "cuda/memory/buffer.h"

namespace cuda::algo {

    void matrixMultiply(const memory::Buffer<float>& a, const memory::Buffer<float>& b, memory::Buffer<float>& c, int m, int n, int k) {
        device::CublasContext ctx;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemm(ctx.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.data(), k, b.data(), n, &beta, c.data(), n));
    }

    void matrixMultiply(const memory::Buffer<double>& a, const memory::Buffer<double>& b, memory::Buffer<double>& c, int m, int n, int k) {
        device::CublasContext ctx;
        const double alpha = 1.0;
        const double beta = 0.0;

        CUBLAS_CHECK(cublasDgemm(ctx.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.data(), k, b.data(), n, &beta, c.data(), n));
    }

}  // namespace cuda::algo
