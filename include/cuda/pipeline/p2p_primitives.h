#pragma once

/**
 * @file p2p_primitives.h
 * @brief Peer-to-peer communication primitives
 *
 * Provides send and receive primitives for inter-GPU communication
 * in pipeline parallelism.
 */

#include "cuda/mesh/device_mesh.h"

#include <cuda_runtime.h>

#include <memory>

namespace cuda::pipeline {

/**
 * @class P2PSendRecv
 * @brief Peer-to-peer send/receive primitives
 *
 * Handles point-to-point communication between two GPUs.
 * Uses cudaMemcpyAsync with peer access when available.
 *
 * @example
 * @code
 * P2PSendRecv p2p(0, 1);
 * p2p.send_async(data, size, stream);
 * // On destination GPU:
 * p2p.recv_async(buffer, size, stream);
 * @endcode
 */
class P2PSendRecv {
public:
    /**
     * @brief Construct P2P send/receive
     * @param src_device Source device index
     * @param dst_device Destination device index
     */
    P2PSendRecv(int src_device, int dst_device);

    ~P2PSendRecv();

    // Non-copyable
    P2PSendRecv(const P2PSendRecv&) = delete;
    P2PSendRecv& operator=(const P2PSendRecv&) = delete;

    /**
     * @brief Send data asynchronously to peer GPU
     * @param data Source data pointer
     * @param bytes Size in bytes
     * @param stream CUDA stream
     */
    void send_async(const void* data, size_t bytes, cudaStream_t stream);

    /**
     * @brief Receive data asynchronously from peer GPU
     * @param data Destination data pointer
     * @param bytes Size in bytes
     * @param stream CUDA stream
     */
    void recv_async(void* data, size_t bytes, cudaStream_t stream);

    /**
     * @brief Wait for send to complete
     */
    void wait_send();

    /**
     * @brief Wait for receive to complete
     */
    void wait_recv();

    /**
     * @brief Check if peer access is available
     * @return true if GPUs can access each other's memory directly
     */
    [[nodiscard]] bool has_peer_access() const;

    /**
     * @brief Get source device
     */
    [[nodiscard]] int src_device() const;

    /**
     * @brief Get destination device
     */
    [[nodiscard]] int dst_device() const;

private:
    int src_device_;
    int dst_device_;
    bool peer_access_;
    void* dst_ptr_ = nullptr;
    size_t bytes_ = 0;
    cudaEvent_t send_event_;
    cudaEvent_t recv_event_;
};

/**
 * @brief Create P2P connections for all device pairs in mesh
 * @param mesh Device mesh
 * @return Vector of P2P send/receive pairs
 */
[[nodiscard]]
std::vector<std::unique_ptr<P2PSendRecv>> create_p2p_connections(
    const ::cuda::mesh::DeviceMesh& mesh);

}  // namespace cuda::pipeline
