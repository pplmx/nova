#include "cuda/device/reduce_kernels.h"

namespace cuda::device {

template<typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op) {
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) {
        if (op == ReduceOp::SUM) {
            val += input[i + blockDim.x];
        } else if (op == ReduceOp::MAX) {
            val = max(val, input[i + blockDim.x]);
        } else {
            val = min(val, input[i + blockDim.x]);
        }
    }
    sdata[tid] = val;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (op == ReduceOp::SUM) {
                sdata[tid] += sdata[tid + s];
            } else if (op == ReduceOp::MAX) {
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            } else {
                sdata[tid] = min(sdata[tid], sdata[tid + s]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

template<typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op) {
    __shared__ T sdata[32];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) {
        if (op == ReduceOp::SUM) {
            val += input[i + blockDim.x];
        } else if (op == ReduceOp::MAX) {
            val = max(val, input[i + blockDim.x]);
        } else {
            val = min(val, input[i + blockDim.x]);
        }
    }

    val = warp_reduce(val, op);

    if (tid % WARP_SIZE == 0) sdata[tid / WARP_SIZE] = val;
    __syncthreads();

    if (tid < WARP_SIZE) val = (tid < blockDim.x / WARP_SIZE) ? sdata[tid] : T{};
    val = warp_reduce(val, op);

    if (tid == 0) output[blockIdx.x] = val;
}

#define REDUCE_KERNEL_INSTANTIATE(T)                                                       \
    template __global__ void reduce_basic_kernel<T>(const T*, T*, size_t, ReduceOp);       \
    template __global__ void reduce_optimized_kernel<T>(const T*, T*, size_t, ReduceOp);

REDUCE_KERNEL_INSTANTIATE(int)
REDUCE_KERNEL_INSTANTIATE(float)
REDUCE_KERNEL_INSTANTIATE(double)
REDUCE_KERNEL_INSTANTIATE(unsigned int)

} // namespace cuda::device
