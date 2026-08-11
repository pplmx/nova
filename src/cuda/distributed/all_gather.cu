/**
 * @file all_gather.cu
 * @brief Multi-GPU all-gather implementation
 *
 * Single-GPU fallback (MGPU-13) is a plain copy. The multi-GPU path is a
 * genuine per-rank NCCL all-gather: each rank contributes its local block on
 * its own communicator, and every rank ends with recv = [rank 0][rank 1]...
 * in rank order (recv must hold count * device_count elements).
 *
 * Converged onto the verified NCCL layer in v2.17 (decision-v17-dist-ops-real-
 * multigpu). The pre-v2.17 P2P implementation read the caller's single
 * send_data as if every source had its own block, so each block came back as a
 * copy of the caller's own data — broken on real multi-GPU (see
 * issue-v17-dist-ops-multigpu-untested / ev-v17-p1-dist-ops-failures).
 */

#include "cuda/distributed/all_gather.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"
#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_all_reduce.h"  // shared static to_nccl_dtype()
#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_error.h"

#include <cuda_runtime.h>

namespace cuda::distributed {

namespace {

size_t element_size_bytes(cudaDataType dtype) {
    switch (dtype) {
        case CUDA_R_8I:
        case CUDA_R_8U:    return 1;
        case CUDA_R_16F:
        case CUDA_R_16BF:  return 2;
        case CUDA_R_32F:
        case CUDA_R_32I:
        case CUDA_R_32U:   return 4;
        case CUDA_R_64F:
        case CUDA_R_64I:
        case CUDA_R_64U:   return 8;
        default:           return 4;
    }
}

int current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void require_nccl_initialized(const char* fn) {
    auto& ctx = cuda::nccl::NcclContext::instance();
    if (!ctx.has_nccl()) {
        throw cuda::nccl::NcclException(
            "multi-GPU all-gather requires an initialized NCCL context (call "
            "NcclContext::instance().initialize() from one thread per rank); "
            "the single-GPU fallback applies only when device_count <= 1",
            ncclInternalError, fn, __FILE__, __LINE__);
    }
}

}  // anonymous namespace

void DistributedAllGather::all_gather(const void* send_data, void* recv_data,
                                      size_t count, cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU fallback: gathering one contribution is a copy.
    if (n <= 1) {
        CUDA_CHECK(cudaMemcpy(recv_data, send_data,
                              count * element_size_bytes(dtype),
                              cudaMemcpyDeviceToDevice));
        return;
    }

    // Per-rank NCCL all-gather (blocking). Caller runs one thread per device,
    // each pinned with cudaSetDevice, holding its own block in `send_data`.
    int device = current_device();
    require_nccl_initialized("DistributedAllGather::all_gather");
    auto& ctx = cuda::nccl::NcclContext::instance();

    cuda::nccl::NcclAllGather op(ctx);
    cuda::nccl::NcclResult result = op.all_gather(
        send_data, recv_data, count,
        cuda::nccl::NcclAllReduce::to_nccl_dtype(dtype),
        ctx.get_stream(device));
    if (!result.ok()) {
        throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                        "DistributedAllGather::all_gather (NCCL)",
                                        __FILE__, __LINE__);
    }
}

void DistributedAllGather::all_gather_async(const void* send_data, void* recv_data,
                                            size_t count, cudaStream_t stream,
                                            cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU fallback (async, on the caller's stream).
    if (n <= 1) {
        CUDA_CHECK(cudaMemcpyAsync(recv_data, send_data,
                                   count * element_size_bytes(dtype),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }

    // Per-rank NCCL all-gather on the caller's stream.
    require_nccl_initialized("DistributedAllGather::all_gather_async");
    auto& ctx = cuda::nccl::NcclContext::instance();

    cuda::nccl::NcclAllGather op(ctx);
    cuda::nccl::NcclResult result = op.all_gather_async(
        send_data, recv_data, count,
        cuda::nccl::NcclAllReduce::to_nccl_dtype(dtype),
        stream);
    if (!result.ok()) {
        throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                        "DistributedAllGather::all_gather_async (NCCL)",
                                        __FILE__, __LINE__);
    }
}

}  // namespace cuda::distributed
