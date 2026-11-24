#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuda::neural {

    struct MatmulOptions {
        cublasHandle_t handle = nullptr;
        /// Stream the GEMM is submitted on. nullptr keeps cuBLAS's default (the null stream).
        ///
        /// cuBLAS is asynchronous: cublasSgemm returns as soon as the work is queued, so this is
        /// the only thing that decides WHICH queue. It was missing until now, and cublasSetStream
        /// was called nowhere in the tree, so every caller ran on the null stream no matter what
        /// stream it passed to a wrapper -- DistributedMatmul::matmul_async() took a stream and
        /// ignored it for exactly this reason.
        cudaStream_t stream = nullptr;
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasOperation_t trans_a = CUBLAS_OP_N;
        cublasOperation_t trans_b = CUBLAS_OP_N;
    };

    void matmul(const float* A, const float* B, float* C, int m, int n, int k, MatmulOptions options = {});

    void matmul_batch(const float* A, const float* B, float* C, int batch_count, int m, int n, int k, MatmulOptions options = {});

    cublasHandle_t get_cublas_handle();

}  // namespace cuda::neural
