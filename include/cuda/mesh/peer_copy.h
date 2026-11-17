#pragma once

/**
 * @file peer_copy.h
 * @brief Async peer-to-peer memory copy primitives
 *
 * Provides asynchronous P2P copy between GPUs with peer access validation
 * and integration with CUDA streams.
 *
 * @example
 * @code
 * cuda::mesh::PeerCopy copier;
 *
 * // Check peer access before copy
 * if (copier.peer_access_available(0, 1)) {
 *     copier.enable_peer_access(0, 1);
 * }
 *
 * // Async copy between GPUs
 * copier.copy_async(dst_ptr, src_ptr, size, 1, 0, stream);
 *
 * // Or synchronous copy
 * copier.copy(dst_ptr, src_ptr, size, 1, 0);
 * @endcode
 */

#include <cuda_runtime.h>

#include <set>
#include <utility>

namespace cuda::mesh {

/**
 * @class PeerCopy
 * @brief Async peer-to-peer memory copy with validation and caching
 *
 * Validates peer access before enabling and tracks enabled pairs in a std::set so a repeated
 * enable is a no-op.
 *
 * @warning NOT thread-safe, though this note previously said it was. The enabled-pairs set
 * (enabled_peer_pairs_, a std::set<std::pair<int,int>>) is read and written by
 * enable_peer_access() with no lock, and that method is public and demonstrably called
 * directly — the example above does so. Two threads enabling peer access at once race on the
 * set; copy_async() alone does not touch it, so read-only use from several threads is fine.
 *
 * A caller that needs concurrent enabling should serialize calls to enable_peer_access(), or
 * enable the pairs up front from one thread.
 *
 * @note Single-GPU fallback: If src_device == dst_device, uses normal
 *       cudaMemcpyAsync without requiring peer access.
 */
class PeerCopy {
public:
    /**
     * @brief Default constructor
     */
    PeerCopy() = default;

    /**
     * @brief Perform async peer-to-peer copy
     *
     * @param dst Destination pointer (device memory)
     * @param src Source pointer (device memory)
     * @param bytes Number of bytes to copy
     * @param dst_device Destination device index
     * @param src_device Source device index
     * @param stream CUDA stream for async operation (default: 0)
     * @throws CudaException if peer access not available (and devices differ)
     *
     * @note If dst_device == src_device, uses cudaMemcpyDeviceToDevice without
     *       requiring peer access enablement.
     */
    void copy_async(void* dst, const void* src, size_t bytes,
                    int dst_device, int src_device,
                    cudaStream_t stream = 0);

    /**
     * @brief Perform synchronous peer-to-peer copy
     *
     * @param dst Destination pointer (device memory)
     * @param src Source pointer (device memory)
     * @param bytes Number of bytes to copy
     * @param dst_device Destination device index
     * @param src_device Source device index
     * @throws CudaException if copy fails
     */
    void copy(void* dst, const void* src, size_t bytes,
              int dst_device, int src_device);

    /**
     * @brief Check if peer access is available between devices
     *
     * @param src_device Source device index
     * @param dst_device Destination device index
     * @return true if peer access is available (cached from DeviceMesh)
     * @note Self-access (src == dst) always returns true
     */
    bool peer_access_available(int src_device, int dst_device) const;

    /**
     * @brief Enable peer access between devices if not already enabled
     *
     * @param src_device Source device index
     * @param dst_device Destination device index
     * @throws CudaException if peer access is not available
     * @note Idempotent: safe to call multiple times
     */
    void enable_peer_access(int src_device, int dst_device);

    // Non-copyable, non-movable
    PeerCopy(const PeerCopy&) = delete;
    PeerCopy& operator=(const PeerCopy&) = delete;

private:
    std::set<std::pair<int, int>> enabled_peer_pairs_;
};

}  // namespace cuda::mesh
