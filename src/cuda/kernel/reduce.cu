#include "cuda/kernel/cuda_utils.h"
#include "cuda/algo/reduce.h"
#include "cuda/algo/device_buffer.h"
#include <vector>
#include <algorithm>

namespace cuda::kernel {

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

} // namespace cuda::kernel

namespace cuda::algo {

namespace {
template<typename T>
T execute_reduce(const T* input, size_t size, bool optimized, cuda::kernel::ReduceOp op) {
    if (size == 0) return T{};

    const int blockSize = 256;
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    DeviceBuffer<T> output(gridSize);

    if (optimized) {
        cuda::kernel::reduce_optimized_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
    } else {
        cuda::kernel::reduce_basic_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    output.copy_to(h_output.data(), gridSize);

    if (op == cuda::kernel::ReduceOp::SUM) {
        T result = 0;
        for (T val : h_output) result += val;
        return result;
    } else if (op == cuda::kernel::ReduceOp::MAX) {
        T result = h_output[0];
        for (size_t i = 1; i < h_output.size(); ++i) {
            result = std::max(result, h_output[i]);
        }
        return result;
    } else {
        T result = h_output[0];
        for (size_t i = 1; i < h_output.size(); ++i) {
            result = std::min(result, h_output[i]);
        }
        return result;
    }
}
}

template<typename T>
T reduce_sum(const T* input, size_t size) {
    return execute_reduce(input, size, false, cuda::kernel::ReduceOp::SUM);
}

template<typename T>
T reduce_sum_optimized(const T* input, size_t size) {
    return execute_reduce(input, size, true, cuda::kernel::ReduceOp::SUM);
}

template<typename T>
T reduce_max(const T* input, size_t size) {
    return execute_reduce(input, size, false, cuda::kernel::ReduceOp::MAX);
}

template<typename T>
T reduce_min(const T* input, size_t size) {
    return execute_reduce(input, size, false, cuda::kernel::ReduceOp::MIN);
}

#define REDUCE_ALGO_INSTANTIATE(T)  \
    template T reduce_sum<T>(const T*, size_t); \
    template T reduce_sum_optimized<T>(const T*, size_t); \
    template T reduce_max<T>(const T*, size_t); \
    template T reduce_min<T>(const T*, size_t);

REDUCE_ALGO_INSTANTIATE(int)
REDUCE_ALGO_INSTANTIATE(float)
REDUCE_ALGO_INSTANTIATE(double)
REDUCE_ALGO_INSTANTIATE(unsigned int)

} // namespace cuda::algo
