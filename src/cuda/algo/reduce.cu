#include "cuda/device/reduce_kernels.h"
#include "cuda/algo/reduce.h"
#include "cuda/memory/buffer.h"
#include <vector>
#include <algorithm>

namespace cuda::algo {

namespace {
template<typename T>
T execute_reduce(const T* input, size_t size, bool optimized, cuda::device::ReduceOp op) {
    if (size == 0) return T{};

    constexpr size_t blockSize = cuda::device::REDUCE_BLOCK_SIZE;
    const size_t gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    cuda::memory::Buffer<T> output(gridSize);

    if (optimized) {
        cuda::device::reduce_optimized_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
    } else {
        cuda::device::reduce_basic_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    output.copy_to(h_output.data(), gridSize);

    if (op == cuda::device::ReduceOp::SUM) {
        T result = 0;
        for (T val : h_output) result += val;
        return result;
    } else if (op == cuda::device::ReduceOp::MAX) {
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
    return execute_reduce(input, size, false, cuda::device::ReduceOp::SUM);
}

template<typename T>
T reduce_sum_optimized(const T* input, size_t size) {
    return execute_reduce(input, size, true, cuda::device::ReduceOp::SUM);
}

template<typename T>
T reduce_max(const T* input, size_t size) {
    return execute_reduce(input, size, false, cuda::device::ReduceOp::MAX);
}

template<typename T>
T reduce_min(const T* input, size_t size) {
    return execute_reduce(input, size, false, cuda::device::ReduceOp::MIN);
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
