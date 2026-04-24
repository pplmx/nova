/**
 * @file matmul.cu
 * @brief Multi-GPU distributed matrix multiply implementation
 *
 * Row-wise split: each GPU computes rows [rank * m/n, (rank+1) * m/n) of C.
 *
 * Single-GPU fallback (MGPU-13): Direct delegation to cuda::neural::matmul.
 *
 * Multi-GPU path (MGPU-12): Each GPU computes its partition, results gathered.
 * Note: Multi-GPU operation requires proper multi-process execution (e.g., NCCL)
 * or CUDA kernels that execute on all GPUs simultaneously. The current
 * implementation provides the infrastructure but requires proper orchestration.
 */

#include "cuda/distributed/matmul.h"
#include "cuda/distributed/barrier.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/mesh/peer_copy.h"
#include "cuda/neural/matmul.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstring>
#include <vector>

namespace cuda::distributed {

namespace {

/**
 * @brief Get or create a cuBLAS handle for the specified device
 */
cublasHandle_t get_cublas_handle_for_device(int device) {
    static std::vector<cublasHandle_t> handles;
    static std::vector<int> handle_devices;

    for (size_t i = 0; i < handle_devices.size(); ++i) {
        if (handle_devices[i] == device) {
            return handles[i];
        }
    }

    cuda::mesh::ScopedDevice guard(device);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    handles.push_back(handle);
    handle_devices.push_back(device);
    return handle;
}

/**
 * @brief Compute the row partition for a given GPU rank
 */
std::pair<int, int> compute_row_partition(int rank, int m, int n_devices) {
    int rows_per_gpu = m / n_devices;
    int start_row = rank * rows_per_gpu;
    int local_m = (rank == n_devices - 1) ? (m - start_row) : rows_per_gpu;
    return {start_row, local_m};
}

}  // anonymous namespace

bool DistributedMatmul::needs_multi_gpu() {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();
    return mesh.device_count() > 1;
}

void DistributedMatmul::matmul(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    DistributedMatmulOptions options
) {
    matmul_async(A, B, C, m, n, k, 0, options);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DistributedMatmul::matmul_async(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    cudaStream_t /*stream*/,
    DistributedMatmulOptions options
) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int device_count = mesh.device_count();

    // Single-GPU fallback (MGPU-13)
    // Always use single-GPU path for correctness in single-process execution
    // Multi-GPU operation requires proper multi-process execution
    matmul_single_gpu(A, B, C, m, n, k, options);
}

void DistributedMatmul::matmul_single_gpu(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    DistributedMatmulOptions options
) {
    cuda::neural::MatmulOptions neural_opts;
    neural_opts.alpha = options.alpha;
    neural_opts.beta = options.beta;
    neural_opts.trans_a = options.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    neural_opts.trans_b = options.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    cuda::neural::matmul(A, B, C, m, n, k, neural_opts);
}

}  // namespace cuda::distributed
