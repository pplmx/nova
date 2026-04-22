#pragma once

#include "device_utils.h"
#include <cstddef>

namespace cuda::device {

constexpr size_t REDUCE_BLOCK_SIZE = 256;
constexpr size_t REDUCE_OPTIMIZED_SHMEM_SIZE = 32;

template<typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op);

template<typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op);

} // namespace cuda::device
