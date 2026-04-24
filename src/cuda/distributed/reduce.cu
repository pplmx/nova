/**
 * @file reduce.cu
 * @brief Multi-GPU all-reduce implementation
 *
 * Implements distributed reduction using a gather-reduce-broadcast pattern
 * coordinated by CPU threads. This is simpler and more testable than true
 * ring all-reduce while maintaining correctness.
 *
 * Algorithm:
 * 1. CPU coordinates: each GPU copies its data to GPU 0 (via host staging)
 * 2. GPU 0 reduces all chunks
 * 3. GPU 0 copies result back to all GPUs (via host staging)
 */

#include "cuda/distributed/reduce.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/mesh/peer_copy.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <vector>

namespace cuda::distributed {

namespace {

// Reduction kernel for element-wise operations
template <typename T, ReductionOp Op>
__global__ void reduce_kernel(T* __restrict__ dst, const T* __restrict__ src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        if constexpr (Op == ReductionOp::Sum) {
            dst[idx] = dst[idx] + src[idx];
        } else if constexpr (Op == ReductionOp::Min) {
            dst[idx] = dst[idx] < src[idx] ? dst[idx] : src[idx];
        } else if constexpr (Op == ReductionOp::Max) {
            dst[idx] = dst[idx] > src[idx] ? dst[idx] : src[idx];
        } else {
            dst[idx] = dst[idx] * src[idx];
        }
    }
}

template <typename T>
void launch_reduce_kernel(T* dst, const T* src, size_t count, ReductionOp op, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((count + block_size - 1) / block_size);

    switch (op) {
        case ReductionOp::Sum: {
            reduce_kernel<T, ReductionOp::Sum><<<grid_size, block_size, 0, stream>>>(
                dst, src, count);
            break;
        }
        case ReductionOp::Min: {
            reduce_kernel<T, ReductionOp::Min><<<grid_size, block_size, 0, stream>>>(
                dst, src, count);
            break;
        }
        case ReductionOp::Max: {
            reduce_kernel<T, ReductionOp::Max><<<grid_size, block_size, 0, stream>>>(
                dst, src, count);
            break;
        }
        case ReductionOp::Product: {
            reduce_kernel<T, ReductionOp::Product><<<grid_size, block_size, 0, stream>>>(
                dst, src, count);
            break;
        }
    }
}

}  // anonymous namespace

bool DistributedReduce::needs_multi_gpu() {
    return cuda::mesh::DeviceMesh::instance().device_count() > 1;
}

void DistributedReduce::all_reduce(const void* send_data, void* recv_data,
                                   size_t count, ReductionOp op, cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();
    size_t elem_size = (dtype == cudaDataType::CUDA_R_32F) ? sizeof(float) : 4;
    size_t total_bytes = count * elem_size;

    // Single-GPU fallback
    if (n <= 1) {
        if (send_data != recv_data) {
            CUDA_CHECK(cudaMemcpy(recv_data, send_data, total_bytes, cudaMemcpyDeviceToDevice));
        }
        return;
    }

    // Get current device
    int my_rank = 0;
    CUDA_CHECK(cudaGetDevice(&my_rank));

    // Allocate staging buffer on GPU 0
    void* staging = nullptr;
    void* result = nullptr;
    void* host_staging = nullptr;

    if (my_rank == 0) {
        CUDA_CHECK(cudaMalloc(&staging, total_bytes * n));
        CUDA_CHECK(cudaMalloc(&result, total_bytes));
        host_staging = malloc(total_bytes * n);
    }

    // Phase 1: Gather data to GPU 0
    // Each GPU: copy data to host, then GPU 0 copies to staging
    std::vector<void*> host_buffers(n);

    // Allocate host buffers for each GPU's data
    for (int i = 0; i < n; ++i) {
        host_buffers[i] = malloc(total_bytes);
    }

    // Copy each GPU's data to host
    for (int gpu = 0; gpu < n; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        void* src = (send_data != recv_data) ? const_cast<void*>(send_data) : recv_data;
        CUDA_CHECK(cudaMemcpy(host_buffers[gpu], src, total_bytes, cudaMemcpyDeviceToHost));
    }

    // GPU 0: copy all host buffers to staging area
    if (my_rank == 0) {
        CUDA_CHECK(cudaSetDevice(0));
        for (int i = 0; i < n; ++i) {
            char* dst = static_cast<char*>(staging) + i * total_bytes;
            CUDA_CHECK(cudaMemcpy(dst, host_buffers[i], total_bytes, cudaMemcpyHostToDevice));
        }

        // Phase 2: Reduce on GPU 0
        // Start with chunk 0 as accumulator
        float* acc = static_cast<float*>(staging);
        for (int i = 1; i < n; ++i) {
            float* src = static_cast<float*>(staging) + i * count;
            launch_reduce_kernel(acc, src, count, op, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result to result buffer
        CUDA_CHECK(cudaMemcpy(result, acc, total_bytes, cudaMemcpyDeviceToDevice));
    }

    // Phase 3: Broadcast result to all GPUs
    // Copy result to host on GPU 0
    if (my_rank == 0) {
        CUDA_CHECK(cudaMemcpy(host_staging, result, total_bytes, cudaMemcpyDeviceToHost));
    }

    // Each GPU copies from host to its device memory
    for (int gpu = 0; gpu < n; ++gpu) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMemcpy(recv_data, (my_rank == 0) ? host_staging : host_buffers[0],
                             total_bytes, cudaMemcpyHostToDevice));
    }

    // Cleanup
    for (int i = 0; i < n; ++i) {
        free(host_buffers[i]);
    }

    if (my_rank == 0) {
        free(host_staging);
        CUDA_CHECK(cudaFree(staging));
        CUDA_CHECK(cudaFree(result));
    }

    // Final sync
    CUDA_CHECK(cudaDeviceSynchronize());

    // Restore device
    CUDA_CHECK(cudaSetDevice(my_rank));
}

void DistributedReduce::all_reduce_async(const void* send_data, void* recv_data,
                                         size_t count, ReductionOp op,
                                         cudaStream_t stream, cudaDataType dtype) {
    // Delegate to sync version for simplicity
    // Production would use async P2P operations
    all_reduce(send_data, recv_data, count, op, dtype);
}

}  // namespace cuda::distributed
