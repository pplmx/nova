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

        // Everything from here to the end of the constructor acquires something, and a throwing
        // constructor never runs the destructor -- so the events are created without throwing and
        // released here if a later step fails. Measured before this guard: failing the second
        // cudaEventCreate left send_event_ owned by nobody.
        CUDA_CHECK(cudaEventCreate(&send_event_));

        // send_event_ records on src_device_ (send_async) and recv_event_ records on
        // dst_device_ (recv_async). A cudaEvent_t belongs to the device that created
        // it, so both events on one device made the receive-side record fail with
        // "invalid resource handle".
        auto release_events_and_rethrow = [this](cudaError_t err) {
            // Both events belong to the device that created them; send_event_ was made on
            // src_device_, which is the device current when this runs.
            if (recv_event_) {
                cudaEventDestroy(recv_event_);
                recv_event_ = nullptr;
            }
            if (send_event_) {
                cudaEventDestroy(send_event_);
                send_event_ = nullptr;
            }
            CUDA_CHECK(err);
        };

        cudaError_t err = cudaSetDevice(dst_device_);
        if (err == cudaSuccess) {
            err = cudaEventCreate(&recv_event_);
        }
        if (err != cudaSuccess) {
            release_events_and_rethrow(err);
        }

        err = cudaSetDevice(src_device_);
        if (err != cudaSuccess) {
            release_events_and_rethrow(err);
        }

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
        // recv_async() records the size of the buffer it was given. Enforce it: the
        // copy below writes `bytes` into that buffer, so a larger send runs past the
        // end of the receiver's allocation. bytes_ was written and never read, so the
        // limit the receiver declared was recorded and then ignored.
        if (bytes_ != 0 && bytes > bytes_) {
            throw std::runtime_error("P2PSendRecv::send_async: send size " + std::to_string(bytes) + " exceeds the registered receive buffer of " + std::to_string(bytes_)
                                     + " bytes");
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
