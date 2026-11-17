#pragma once

/**
 * @file all_gather.h
 * @brief Multi-GPU all-gather operation
 *
 * Each GPU contributes its local data, and all GPUs receive the concatenated
 * results from all GPUs. The output buffer has size count * n GPUs.
 *
 * @warning send_data and recv_data must be DEVICE pointers. The single-device path copies
 * with cudaMemcpyDeviceToDevice and the multi-device path hands the pointers to NCCL; a host
 * pointer from a std::vector or a stack array is passed through unchanged and faults. This
 * was measured: calling all_gather with host arrays raises "an illegal memory access was
 * encountered" at the cudaMemcpy in the fallback.
 *
 * @example
 * @code
 * // Each GPU has count elements, after call all GPUs have count * n elements.
 * // send_data and recv_data are device allocations, not host vectors.
 * float* send_data = nullptr;   // cudaMalloc'd, count elements
 * float* recv_data = nullptr;   // cudaMalloc'd, count * n elements
 * DistributedAllGather::all_gather(send_data, recv_data, count);
 * // recv_data layout: [GPU0_data, GPU1_data, ..., GPUn-1_data]
 * @endcode
 */

#include "common.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::distributed {

/**
 * @class DistributedAllGather
 * @brief Multi-GPU all-gather collective operation
 *
 * Concatenates data from all GPUs into a single buffer on each GPU.
 * Each GPU contributes count elements, and receives count * n elements.
 *
 * @note Single-GPU fallback: Copy send_data to recv_data (same size)
 * @note recv_data size must be count * n elements
 */
class DistributedAllGather {
public:
    /**
     * @brief All-gather: gather data from all GPUs
     *
     * @param send_data Input: count elements from this GPU (device pointer)
     * @param recv_data Output: count * n elements, all GPUs' data concatenated
     *                   (device pointer)
     * @param count Number of elements per GPU
     * @param dtype CUDA data type (default: CUDA_R_32F)
     *
     * @note recv_data must have space for count * device_count elements
     */
    static void all_gather(
        const void* send_data,
        void* recv_data,
        size_t count,
        cudaDataType dtype = cudaDataType::CUDA_R_32F);

    /**
     * @brief Async all-gather with explicit stream
     *
     * @param send_data Input data (device pointer)
     * @param recv_data Output buffer (device pointer, count * device_count elements)
     * @param count Elements per GPU
     * @param stream CUDA stream
     * @param dtype CUDA data type
     */
    static void all_gather_async(
        const void* send_data,
        void* recv_data,
        size_t count,
        cudaStream_t stream,
        cudaDataType dtype = cudaDataType::CUDA_R_32F);
};

}  // namespace cuda::distributed
