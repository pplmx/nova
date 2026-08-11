/**
 * @file reduce.cu
 * @brief Multi-GPU all-reduce implementation
 *
 * Single-GPU fallback (MGPU-13) is a plain copy. The multi-GPU path is a
 * genuine per-rank NCCL all-reduce: each rank contributes its local data on
 * its own communicator (`current_comm()`, resolved from the calling thread's
 * current device) and every rank ends with the reduced result.
 *
 * This converges the high-level distributed ops onto the verified NCCL layer
 * (see decision-nccl-current-device-routing / decision-v17-dist-ops-real-
 * multigpu). The pre-v2.17 CPU-coordinated host-staging implementation was
 * broken on real multi-GPU (it read the caller's single send_data while
 * cudaSetDevice-swapping across all GPUs and could never gather distinct
 * per-rank contributions) and is not retained — see
 * issue-v17-dist-ops-multigpu-untested and ev-v17-p1-dist-ops-failures.
 */

#include "cuda/distributed/reduce.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_error.h"

#include <cuda_runtime.h>

namespace cuda::distributed {

namespace {

// Byte size of one element for the given dtype (used by the single-GPU
// fallback copy). Full map, not float-or-4: CUDA_R_16F is 2 bytes (a 4-byte
// copy would overrun the buffer) and the 64-bit dtypes are 8.
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

// Fail fast (throw) when a multi-GPU group has no initialized NCCL context:
// the single-GPU fallback only applies to device_count <= 1, and there is no
// working non-NCCL multi-GPU implementation to fall back to.
void require_nccl_initialized() {
    auto& ctx = cuda::nccl::NcclContext::instance();
    if (!ctx.has_nccl()) {
        throw cuda::nccl::NcclException(
            "multi-GPU all-reduce requires an initialized NCCL context (call "
            "NcclContext::instance().initialize() from one thread per rank); "
            "the single-GPU fallback applies only when device_count <= 1",
            ncclInternalError, "DistributedReduce",
            __FILE__, __LINE__);
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

    // Single-GPU fallback: all-reduce of one contribution is the identity.
    if (n <= 1) {
        if (send_data != recv_data) {
            CUDA_CHECK(cudaMemcpy(recv_data, send_data,
                                  count * element_size_bytes(dtype),
                                  cudaMemcpyDeviceToDevice));
        }
        return;
    }

    // Per-rank NCCL all-reduce (blocking). Every rank thread must enter; the
    // caller must run one thread per device pinned with cudaSetDevice (see the
    // collective contract used by the NCCL suite / distributed matmul).
    //
    // Ordering contract: the collective posts and blocks on the context's
    // per-device stream (ctx.get_stream(device)) — send_data must be readable
    // from that stream (produce on the same stream, the default stream, or a
    // prior host-visible barrier such as a blocking cudaMemcpy). There is no
    // implicit edge from the caller's arbitrary private stream.
    int device = current_device();
    require_nccl_initialized();
    auto& ctx = cuda::nccl::NcclContext::instance();

    cuda::nccl::NcclAllReduce reduce(ctx);
    cuda::nccl::NcclResult result = reduce.all_reduce(
        send_data, recv_data, count,
        cuda::nccl::NcclAllReduce::to_nccl_dtype(dtype),
        cuda::nccl::NcclAllReduce::to_nccl_op(op),
        ctx.get_stream(device));
    if (!result.ok()) {
        throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                        "DistributedReduce::all_reduce (NCCL)",
                                        __FILE__, __LINE__);
    }
}

void DistributedReduce::all_reduce_async(const void* send_data, void* recv_data,
                                         size_t count, ReductionOp op,
                                         cudaStream_t stream, cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU fallback (async, on the caller's stream).
    if (n <= 1) {
        if (send_data != recv_data) {
            CUDA_CHECK(cudaMemcpyAsync(recv_data, send_data,
                                       count * element_size_bytes(dtype),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }

    // Per-rank NCCL all-reduce on the caller's stream (the pre-v2.17 async
    // ignored its stream and delegated to a blocking host-staged sync op).
    require_nccl_initialized();
    auto& ctx = cuda::nccl::NcclContext::instance();

    cuda::nccl::NcclAllReduce reduce(ctx);
    cuda::nccl::NcclResult result = reduce.all_reduce_async(
        send_data, recv_data, count,
        cuda::nccl::NcclAllReduce::to_nccl_dtype(dtype),
        cuda::nccl::NcclAllReduce::to_nccl_op(op),
        stream);
    if (!result.ok()) {
        throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                        "DistributedReduce::all_reduce_async (NCCL)",
                                        __FILE__, __LINE__);
    }
}

}  // namespace cuda::distributed
