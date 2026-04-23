#pragma once

/**
 * @file reduce_kernels.h
 * @brief CUDA reduce kernels for parallel reduction operations
 */

#include <cstddef>

#include "device_utils.h"

namespace cuda::device {

/**
 * @brief Block size for basic reduction kernel
 */
constexpr size_t REDUCE_BLOCK_SIZE = 256;

/**
 * @brief Shared memory size for optimized reduction (one value per warp)
 */
constexpr size_t REDUCE_OPTIMIZED_SHMEM_SIZE = 32;

/**
 * @brief Basic parallel reduction kernel
 *
 * Performs parallel reduction using shared memory. Each block reduces
 * a portion of the input array, storing the result at output[blockIdx.x].
 *
 * @tparam T Arithmetic type (int, float, double)
 * @param input Pointer to input array on device
 * @param output Pointer to output array on device (size: (n + 2*blockDim.x - 1) / (2*blockDim.x))
 * @param size Total number of elements in input
 * @param op Reduction operation (SUM, MAX, MIN)
 */
template <typename T>
__global__ void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op);

/**
 * @brief Optimized parallel reduction kernel using warp-level primitives
 *
 * Performs parallel reduction with warp-level shuffling for better performance.
 * Each block reduces using warp shuffles before shared memory reduction.
 *
 * @tparam T Arithmetic type (int, float, double)
 * @param input Pointer to input array on device
 * @param output Pointer to output array on device
 * @param size Total number of elements in input
 * @param op Reduction operation (SUM, MAX, MIN)
 */
template <typename T>
__global__ void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op);

}  // namespace cuda::device
