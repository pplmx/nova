/**
 * @file barrier.cu
 * @brief MeshBarrier implementation converged onto the verified NCCL layer
 *
 * Multi-GPU barrier via a genuine per-rank `NcclBarrier` collective: each rank
 * enters from its own thread pinned with cudaSetDevice, and the collective
 * completes only when every rank has arrived (true rendezvous semantics).
 * Single-GPU fallback is a plain cudaDeviceSynchronize.
 *
 * This converges the last unconverged high-level distributed op onto the
 * verified NCCL layer (see DEC-001 / issue-v17-meshbarrier-multigpu). The
 * pre-v2.18 implementation was a per-instance host event-poll: it recorded an
 * event per device on internally-created (empty) streams and host-polled for
 * completion — those events fire immediately, so a rank was released without
 * any cross-rank arrival signal (wrong barrier semantics; proven by the
 * MultiGpu_*Rendezvous tests — HYP-001 / EV-001).
 */

#include "cuda/distributed/barrier.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"
#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_error.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace cuda::distributed {

namespace {

// Fail fast (throw) when a multi-GPU group has no initialized NCCL context:
// the single-GPU fallback only applies to device_count <= 1, and the
// pre-v2.18 non-NCCL event-poll path performed no real barrier at all.
void require_nccl_initialized() {
    auto& ctx = cuda::nccl::NcclContext::instance();
    if (!ctx.has_nccl()) {
        throw cuda::nccl::NcclException(
            "multi-GPU barrier requires an initialized NCCL context (call "
            "NcclContext::instance().initialize() from one thread per rank); "
            "the single-GPU fallback applies only when device_count <= 1",
            ncclInternalError, "MeshBarrier", __FILE__, __LINE__);
    }
}

int current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void throw_nccl(const cuda::nccl::NcclResult& result, const char* expr) {
    throw cuda::nccl::NcclException(result.error_message.c_str(), result.code,
                                    expr, __FILE__, __LINE__);
}

}  // anonymous namespace

MeshBarrier::MeshBarrier()
    : device_count_(0),
      barriering_(false),
      initialized_(false) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();
    device_count_ = mesh.device_count();
    initialized_ = device_count_ > 1;
}

MeshBarrier::~MeshBarrier() = default;

void MeshBarrier::synchronize() {
    // Single-GPU fallback: a barrier over one device is a device sync.
    if (!initialized_) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    require_nccl_initialized();
    auto& ctx = cuda::nccl::NcclContext::instance();

    barriering_.store(true, std::memory_order_release);
    // Per-rank blocking NCCL barrier. Completes only when every rank has
    // entered the same collective (true rendezvous). The stream is the
    // context's per-device stream, matching the other distributed ops'
    // ordering contract.
    try {
        cuda::nccl::NcclBarrier barrier(ctx);
        cuda::nccl::NcclResult result =
            barrier.barrier(ctx.get_stream(current_device()));
        if (!result.ok()) {
            throw_nccl(result, "MeshBarrier::synchronize (NCCL)");
        }
    } catch (...) {
        barriering_.store(false, std::memory_order_release);
        throw;
    }
    barriering_.store(false, std::memory_order_release);
}

void MeshBarrier::synchronize_async(cudaStream_t stream) {
    // Single-GPU fallback: nothing to synchronize cross-device.
    if (!initialized_) {
        return;
    }

    require_nccl_initialized();
    auto& ctx = cuda::nccl::NcclContext::instance();

    barriering_.store(true, std::memory_order_release);
    // Per-rank NCCL barrier on the caller's stream (pre-v2.18 ignored the
    // stream argument entirely). The underlying NcclBarrier::barrier_async
    // waits for cross-rank completion before returning, so on success the
    // rendezvous is genuinely done and barriering_ resets — matching
    // synchronize(). Completes only when every rank has entered.
    try {
        cuda::nccl::NcclBarrier barrier(ctx);
        cuda::nccl::NcclResult result = barrier.barrier_async(stream);
        if (!result.ok()) {
            throw_nccl(result, "MeshBarrier::synchronize_async (NCCL)");
        }
    } catch (...) {
        barriering_.store(false, std::memory_order_release);
        throw;
    }
    barriering_.store(false, std::memory_order_release);
}

void MeshBarrier::synchronize_devices(const std::vector<int>& devices) {
    const int n = static_cast<int>(devices.size());

    // True single-GPU fallback (device_count <= 1): device sync.
    if (!initialized_) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    require_nccl_initialized();
    auto& ctx = cuda::nccl::NcclContext::instance();

    // Only the full NCCL group can use the existing group communicator. A
    // proper subset — including a 1-element request on a multi-GPU box —
    // would need a sub-communicator (not exposed by NcclContext), and any
    // group member not entering the collective would stall it; fail fast with
    // a clear explanation instead.
    const std::vector<int> group = ctx.device_ids();
    std::vector<int> requested = devices;
    std::sort(requested.begin(), requested.end());
    std::vector<int> sorted_group = group;
    std::sort(sorted_group.begin(), sorted_group.end());
    if (requested != sorted_group) {
        throw cuda::nccl::NcclException(
            "synchronize_devices: only the full NCCL group is supported (a "
            "proper subset requires a sub-communicator, which NcclContext does "
            "not expose)",
            ncclInvalidArgument, "MeshBarrier::synchronize_devices",
            __FILE__, __LINE__);
    }

    barriering_.store(true, std::memory_order_release);
    try {
        cuda::nccl::NcclBarrier barrier(ctx);
        cuda::nccl::NcclResult result =
            barrier.barrier(ctx.get_stream(current_device()));
        if (!result.ok()) {
            throw_nccl(result, "MeshBarrier::synchronize_devices (NCCL)");
        }
    } catch (...) {
        barriering_.store(false, std::memory_order_release);
        throw;
    }
    barriering_.store(false, std::memory_order_release);
}

}  // namespace cuda::distributed
