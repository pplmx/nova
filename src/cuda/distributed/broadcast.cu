/**
 * @file broadcast.cu
 * @brief Multi-GPU broadcast implementation
 *
 * Single-GPU broadcast (MGPU-13) is a no-op (n <= 1 has nothing to receive).
 * The multi-GPU path is a genuine per-rank NCCL broadcast: the root rank's
 * `data` is copied to every rank's `data` in-place (each rank passes its own
 * buffer; `root` names the group rank broadcasting).
 *
 * Converged onto the verified NCCL layer in v2.17 (decision-v17-dist-ops-real-
 * multigpu). The pre-v2.17 P2P/event implementation dereferenced the root's
 * data address on destination devices, so non-root ranks never received the
 * root's value — broken on real multi-GPU (see issue-v17-dist-ops-multigpu-
 * untested / ev-v17-p1-dist-ops-failures).
 */

#include "cuda/distributed/broadcast.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"
#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_error.h"

#include <cuda_runtime.h>

namespace cuda::distributed {

namespace {

int current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void require_nccl_initialized(const char* fn) {
    auto& ctx = cuda::nccl::NcclContext::instance();
    if (!ctx.has_nccl()) {
        throw cuda::nccl::NcclException(
            "multi-GPU broadcast requires an initialized NCCL context (call "
            "NcclContext::instance().initialize() from one thread per rank); "
            "the single-GPU fallback applies only when device_count <= 1",
            ncclInternalError, fn, __FILE__, __LINE__);
    }
}

}  // anonymous namespace

void DistributedBroadcast::broadcast(void* data, size_t count, int root, cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU broadcast: nothing to receive.
    if (n <= 1) {
        return;
    }

    // Per-rank NCCL broadcast (blocking), in-place (data is both the root's
    // source and every rank's destination). @p root is the broadcasting RANK
    // within the NCCL group; for the default {0..n-1} group rank == device, but
    // callers of a custom NcclContextConfig must pass
    // NcclContext::rank_of_device(root_device), not the raw device index.
    int device = current_device();
    require_nccl_initialized("DistributedBroadcast::broadcast");
    auto& ctx = cuda::nccl::NcclContext::instance();

    cuda::nccl::NcclBroadcast op(ctx);
    cuda::nccl::NcclResult result = op.broadcast(
        data, data, count,
        cuda::nccl::NcclBroadcast::to_nccl_dtype(dtype),
        root, ctx.get_stream(device));
    if (!result.ok()) {
        throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                        "DistributedBroadcast::broadcast (NCCL)",
                                        __FILE__, __LINE__);
    }
}

void DistributedBroadcast::broadcast_async(void* data, size_t count,
                                           int root, cudaStream_t stream,
                                           cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU broadcast: nothing to receive.
    if (n <= 1) {
        return;
    }

    // Per-rank NCCL broadcast on the caller's stream (the pre-v2.17 async
    // ignored its stream).
    require_nccl_initialized("DistributedBroadcast::broadcast_async");
    auto& ctx = cuda::nccl::NcclContext::instance();

    cuda::nccl::NcclBroadcast op(ctx);
    cuda::nccl::NcclResult result = op.broadcast_async(
        data, data, count,
        cuda::nccl::NcclBroadcast::to_nccl_dtype(dtype),
        root, stream);
    if (!result.ok()) {
        throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                        "DistributedBroadcast::broadcast_async (NCCL)",
                                        __FILE__, __LINE__);
    }
}

}  // namespace cuda::distributed
