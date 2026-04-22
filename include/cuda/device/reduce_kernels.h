#pragma once

#include "device_utils.h"
#include <cstddef>

namespace cuda::device {

template<typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op);

template<typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op);

} // namespace cuda::device
