/**
 * @file broadcast.cu
 * @brief Broadcast implementation using P2P peer copy
 */

#include "cuda/distributed/broadcast.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/mesh/peer_copy.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <memory>

namespace cuda::distributed {

void DistributedBroadcast::broadcast(void* data, size_t count, int root, cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU or root: nothing to do
    if (n <= 1) {
        return;
    }

    broadcast_async(data, count, root, 0, dtype);

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DistributedBroadcast::broadcast_async(void* data, size_t count,
                                           int root, cudaStream_t /*stream*/,
                                           cudaDataType dtype) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();

    // Single-GPU or root: nothing to do
    if (n <= 1) {
        return;
    }

    // Get my rank
    int rank = 0;
    CUDA_CHECK(cudaGetDevice(&rank));

    // Root broadcasts to all other GPUs
    if (rank == root) {
        auto& streams = MeshStreams::instance();
        if (!streams.initialized()) {
            streams.initialize(n);
        }

        size_t elem_size = (dtype == cudaDataType::CUDA_R_32F) ? sizeof(float) : 4;
        size_t total_bytes = count * elem_size;

        // Copy to each destination GPU via P2P
        cuda::mesh::PeerCopy copier;

        for (int dst = 0; dst < n; ++dst) {
            if (dst == root) continue;

            // Enable peer access if needed
            if (mesh.can_access_peer(root, dst)) {
                copier.enable_peer_access(root, dst);
            }

            // Record event on root after data is ready
            CUDA_CHECK(cudaEventRecord(streams.get_event(root), streams.get_stream(root)));

            // Destination waits for root's data
            CUDA_CHECK(cudaSetDevice(dst));
            CUDA_CHECK(cudaStreamWaitEvent(streams.get_stream(dst), streams.get_event(root), 0));

            // Async copy from root to destination
            if (mesh.can_access_peer(root, dst)) {
                copier.copy_async(data, data, total_bytes, dst, root,
                                 streams.get_stream(dst));
            } else {
                // Host-mediated fallback for non-P2P capable pairs
                CUDA_CHECK(cudaSetDevice(root));
                void* host_tmp = malloc(total_bytes);
                CUDA_CHECK(cudaMemcpyAsync(host_tmp, data, total_bytes,
                                           cudaMemcpyDeviceToHost, streams.get_stream(root)));
                CUDA_CHECK(cudaEventRecord(streams.get_event(root), streams.get_stream(root)));

                CUDA_CHECK(cudaSetDevice(dst));
                CUDA_CHECK(cudaStreamWaitEvent(streams.get_stream(dst), streams.get_event(root), 0));
                CUDA_CHECK(cudaMemcpyAsync(data, host_tmp, total_bytes,
                                           cudaMemcpyHostToDevice, streams.get_stream(dst)));
                free(host_tmp);
            }

            // Record completion event on destination
            CUDA_CHECK(cudaEventRecord(streams.get_event(dst), streams.get_stream(dst)));
        }
    } else {
        // Non-root: just wait for data to arrive
        // This is handled by the stream synchronization on each destination
    }
}

}  // namespace cuda::distributed
