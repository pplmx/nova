/**
 * @file p2p_primitives.cpp
 * @brief P2P primitives implementation
 */

#include "cuda/pipeline/p2p_primitives.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::pipeline {

    P2PSendRecv::P2PSendRecv(int src_device, int dst_device)
        : src_device_(src_device),
          dst_device_(dst_device),
          peer_access_(false) {
        // cudaEventCreate and cudaDeviceEnablePeerAccess both act on the CURRENT
        // device, while the arguments name a specific pair. Without selecting
        // src_device_ first, enabling peer access either failed with
        // "invalid argument" (the pair has nothing to do with whatever device the
        // thread happened to be on) or silently configured the wrong one.
        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        CUDA_CHECK(cudaSetDevice(src_device_));

        CUDA_CHECK(cudaEventCreate(&send_event_));

        // send_event_ records on src_device_ (send_async) and recv_event_ records on
        // dst_device_ (recv_async). A cudaEvent_t belongs to the device that created
        // it, so both events on one device made the receive-side record fail with
        // "invalid resource handle".
        CUDA_CHECK(cudaSetDevice(dst_device_));
        CUDA_CHECK(cudaEventCreate(&recv_event_));
        CUDA_CHECK(cudaSetDevice(src_device_));

        int can_access = 0;
        const cudaError_t can_err = cudaDeviceCanAccessPeer(&can_access, src_device_, dst_device_);
        if (can_err != cudaSuccess) {
            // An unsupported pair is a property of the topology, not an error the
            // caller can act on here: leave peer_access_ false so the operations
            // become explicit no-ops rather than throwing out of a constructor.
            can_access = 0;
            (void)cudaGetLastError();
        }
        peer_access_ = (can_access != 0);

        if (peer_access_) {
            const cudaError_t enable_err = cudaDeviceEnablePeerAccess(dst_device_, 0);
            if (enable_err == cudaErrorPeerAccessAlreadyEnabled) {
                // Another P2PSendRecv (or the caller) already enabled this
                // direction. That is the desired state, not a failure — and the
                // previous code made constructing a second object for the same
                // device pair throw.
                (void)cudaGetLastError();
            } else if (enable_err != cudaSuccess) {
                peer_access_ = false;
                (void)cudaGetLastError();
            }
        }

        if (had_previous) {
            cudaSetDevice(previous_device);
        }
    }

    P2PSendRecv::~P2PSendRecv() {
        // Not using CUDA_CHECK here: throwing in a destructor is undefined
        // behavior (C++11 destructors default to noexcept).
        //
        // Events belong to the device that was current when they were created
        // (src_device_), so select it before destroying them and put the caller's
        // device back afterwards.
        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        // Each event must be destroyed on the device that created it.
        cudaSetDevice(src_device_);
        cudaEventDestroy(send_event_);
        cudaSetDevice(dst_device_);
        cudaEventDestroy(recv_event_);
        if (had_previous) {
            cudaSetDevice(previous_device);
        }
    }

    void P2PSendRecv::send_async(const void* data, size_t bytes, cudaStream_t stream) {
        // A send with no destination is a caller error, not a successful no-op:
        // returning silently dropped the data and the receiver waited forever for
        // bytes that were never queued.
        if (!peer_access_) {
            throw std::runtime_error("P2PSendRecv::send_async: no peer access between device " + std::to_string(src_device_) + " and " + std::to_string(dst_device_));
        }
        if (dst_ptr_ == nullptr) {
            throw std::runtime_error(
                "P2PSendRecv::send_async: no destination registered; call "
                "recv_async() with the destination buffer before sending");
        }

        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        CUDA_CHECK(cudaSetDevice(src_device_));
        CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr_, dst_device_, data, src_device_, bytes, stream));
        CUDA_CHECK(cudaEventRecord(send_event_, stream));
        if (had_previous) {
            cudaSetDevice(previous_device);
        }
    }

    void P2PSendRecv::recv_async(void* data, size_t bytes, cudaStream_t stream) {
        if (!peer_access_) {
            throw std::runtime_error("P2PSendRecv::recv_async: no peer access between device " + std::to_string(src_device_) + " and " + std::to_string(dst_device_));
        }

        dst_ptr_ = data;
        bytes_ = bytes;

        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        CUDA_CHECK(cudaSetDevice(dst_device_));
        CUDA_CHECK(cudaEventRecord(recv_event_, stream));
        if (had_previous) {
            cudaSetDevice(previous_device);
        }
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

    std::vector<std::unique_ptr<P2PSendRecv>> create_p2p_connections(const ::cuda::mesh::DeviceMesh& mesh) {
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
