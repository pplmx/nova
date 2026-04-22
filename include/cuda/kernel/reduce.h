#pragma once

#include "cuda/device/reduce_kernels.h"
#include "cuda/algo/reduce.h"

using cuda::device::ReduceOp;
using cuda::device::warp_reduce;
using cuda::device::WARP_SIZE;

using cuda::algo::reduce_sum;
using cuda::algo::reduce_sum_optimized;
using cuda::algo::reduce_max;
using cuda::algo::reduce_min;

template<typename T>
T reduceSum(const T* d_input, size_t size) {
    return cuda::algo::reduce_sum(d_input, size);
}

template<typename T>
T reduceSumOptimized(const T* d_input, size_t size) {
    return cuda::algo::reduce_sum_optimized(d_input, size);
}

template<typename T>
T reduceMax(const T* d_input, size_t size) {
    return cuda::algo::reduce_max(d_input, size);
}

template<typename T>
T reduceMin(const T* d_input, size_t size) {
    return cuda::algo::reduce_min(d_input, size);
}
