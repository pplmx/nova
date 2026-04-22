#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, \
                    static_cast<int>(status));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace cuda::device {

enum class ReduceOp { SUM, MAX, MIN };

constexpr int WARP_SIZE = 32;

template<typename T>
__device__ T warp_reduce(T val, ReduceOp op) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        if (op == ReduceOp::SUM) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        } else if (op == ReduceOp::MAX) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        } else {
            val = min(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

} // namespace cuda::device
