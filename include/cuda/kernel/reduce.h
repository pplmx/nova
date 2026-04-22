#pragma once

#include "cuda_utils.h"
#include <cstddef>

namespace cuda::kernel {

template<typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op);

template<typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op);

} // namespace cuda::kernel
