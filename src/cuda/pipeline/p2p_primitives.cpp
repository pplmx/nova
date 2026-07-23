/**
 * @file p2p_primitives.cpp
 * @brief P2P primitives implementation
 */

#include "cuda/pipeline/p2p_primitives.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <vector>

namespace cuda::pipeline {

P2PSendRecv::P2PSendRecv(int src_device, int dst_device)
    : src_device_(src_device),
      dst_device_(dst_device),
      peer_access_(false) {
    CUDA_CHECK(cudaEventCreate(&send_event_));
    CUDA_CHECK(cudaEventCreate(&recv_event_));

    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src_device_, dst_device_));
    peer_access_ = (can_access != 0);

    if (peer_access_) {
        CUDA_CHECK(cudaDeviceEnablePeerAccess(dst_device_, src_device_));
    }
}

P2PSendRecv::~P2PSendRecv() {
    // Not using CUDA_CHECK here: throwing in a destructor is undefined
    // behavior (C++11 destructors default to noexcept).
    cudaEventDestroy(send_event_);
    cudaEventDestroy(recv_event_);
}

void P2PSendRecv::send_async(const void* data, size_t bytes, cudaStream_t stream) {
    if (!peer_access_ || dst_ptr_ == nullptr) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(src_device_));
    CUDA_CHECK(cudaMemcpyPeerAsync(
        dst_ptr_, dst_device_,
        data, src_device_,
        bytes,
        stream));
    CUDA_CHECK(cudaEventRecord(send_event_, stream));
}

void P2PSendRecv::recv_async(void* data, size_t bytes, cudaStream_t stream) {
    if (!peer_access_) {
        return;
    }

    dst_ptr_ = data;
    bytes_ = bytes;

    CUDA_CHECK(cudaSetDevice(dst_device_));
    CUDA_CHECK(cudaEventRecord(recv_event_, stream));
}

void P2PSendRecv::wait_send() {
    if (!peer_access_) {
        return;
    }
    CUDA_CHECK(cudaEventSynchronize(send_event_));
}

void P2PSendRecv::wait_recv() {
    if (!peer_access_) {
        return;
    }
    CUDA_CHECK(cudaEventSynchronize(recv_event_));
}

bool P2PSendRecv::has_peer_access() const {
    return peer_access_;
}

int P2PSendRecv::src_device() const {
    return src_device_;
}

int P2PSendRecv::dst_device() const {
    return dst_device_;
}

std::vector<std::unique_ptr<P2PSendRecv>> create_p2p_connections(
    const ::cuda::mesh::DeviceMesh& mesh) {
    std::vector<std::unique_ptr<P2PSendRecv>> connections;

    int device_count = mesh.device_count();
    for (int i = 0; i < device_count; ++i) {
        for (int j = i + 1; j < device_count; ++j) {
            if (mesh.can_access_peer(i, j)) {
                connections.push_back(std::make_unique<P2PSendRecv>(i, j));
            }
        }
    }

    return connections;
}

}  // namespace cuda::pipeline
