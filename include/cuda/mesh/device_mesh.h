#pragma once

/**
 * @file device_mesh.h
 * @brief Device mesh detection and peer capability management
 *
 * Provides device enumeration, peer access capability queries, and cached
 * peer access matrix for O(1) topology lookups.
 *
 * @example
 * @code
 * auto& mesh = cuda::mesh::DeviceMesh::instance();
 * mesh.initialize();
 *
 * // Query device count
 * int count = mesh.device_count();
 *
 * // Check peer access capability
 * if (mesh.can_access_peer(0, 1)) {
 *     // Peer-to-peer access is available
 * }
 *
 * // Get cached peer capability matrix
 * const auto& capabilities = mesh.peer_capabilities();
 * bool can_access = capabilities.can_access(0, 1);
 * @endcode
 */

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace cuda::mesh {

/**
 * @struct PeerInfo
 * @brief Information about a single CUDA device
 */
struct PeerInfo {
    int device_id = -1;
    size_t global_memory_bytes = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    char name[256] = {0};
};

/**
 * @class PeerCapabilityMap
 * @brief Cached peer access capability matrix for O(1) topology lookups
 *
 * Stores the results of cudaDeviceCanAccessPeer queries in a 2D matrix
 * for fast capability lookups without repeated CUDA API calls.
 *
 * @example
 * @code
 * PeerCapabilityMap capabilities;
 * capabilities.initialize(4);  // Initialize for 4 GPUs
 *
 * // O(1) lookup for peer access
 * if (capabilities.can_access(0, 1)) {
 *     // Can access
 * }
 * @endcode
 */
class PeerCapabilityMap {
public:
    /**
     * @brief Initialize the peer capability matrix for the given device count
     * @param device_count Number of CUDA devices
     */
    void initialize(int device_count);

    /**
     * @brief Check if peer access is available between two devices
     * @param src Source device index
     * @param dst Destination device index
     * @return true if src can access dst's memory via P2P
     */
    bool can_access(int src, int dst) const;

    /**
     * @brief Returns the number of devices in the matrix
     * @return Device count
     */
    int device_count() const { return static_cast<int>(matrix_.size()); }

private:
    std::vector<std::vector<bool>> matrix_;
};

/**
 * @class DeviceMesh
 * @brief Singleton for device topology discovery and peer access management
 *
 * Provides lazy-initialized device enumeration and peer capability queries.
 * Uses the Meyer's singleton pattern for thread-safe lazy initialization.
 *
 * @note All operations are safe on single-GPU systems. The peer capability
 *       matrix will have size 1x1 with can_access(0, 0) = true.
 *
 * @example
 * @code
 * auto& mesh = cuda::mesh::DeviceMesh::instance();
 * mesh.initialize();
 *
 * // Get all mesh devices
 * auto devices = mesh.get_mesh_devices();
 * for (const auto& info : devices) {
 *     printf("GPU %d: %s (%.1f GB)\n",
 *            info.device_id, info.name,
 *            info.global_memory_bytes / 1e9);
 * }
 * @endcode
 */
class DeviceMesh {
public:
    /**
     * @brief Get the singleton instance (Meyer's singleton)
     * @return Reference to the DeviceMesh instance
     */
    static DeviceMesh& instance();

    /**
     * @brief Get the number of CUDA devices
     * @return Device count
     */
    int device_count() const { return device_count_; }

    /**
     * @brief Get information about all devices in the mesh
     * @return Vector of PeerInfo for each device
     */
    std::vector<PeerInfo> get_mesh_devices() const;

    /**
     * @brief Check if peer-to-peer access is available between two devices
     * @param src Source device index
     * @param dst Destination device index
     * @return true if peer access is available
     * @note Self-access (src == dst) always returns true
     */
    bool can_access_peer(int src, int dst) const;

    /**
     * @brief Get the cached peer capability map
     * @return Reference to the PeerCapabilityMap
     */
    const PeerCapabilityMap& peer_capabilities() const { return peer_capabilities_; }

    /**
     * @brief Initialize the device mesh (lazy initialization)
     *
     * Queries device count and builds peer capability matrix.
     * Safe to call multiple times (idempotent).
     */
    void initialize();

    // Non-copyable, non-movable
    DeviceMesh(const DeviceMesh&) = delete;
    DeviceMesh& operator=(const DeviceMesh&) = delete;
    DeviceMesh(DeviceMesh&&) = delete;
    DeviceMesh& operator=(DeviceMesh&&) = delete;

private:
    DeviceMesh();
    ~DeviceMesh() = default;

    int device_count_ = 0;
    PeerCapabilityMap peer_capabilities_;
    std::atomic<bool> initialized_{false};
};

/**
 * @class ScopedDevice
 * @brief RAII guard for device switching
 *
 * Saves the current device on construction and restores it on destruction.
 * Provides exception-safe device switching.
 *
 * @example
 * @code
 * void do_work_on_device(int target_device) {
 *     cuda::mesh::ScopedDevice guard(target_device);
 *     // All CUDA operations here use target_device
 *     // On scope exit, original device is restored
 * }
 * @endcode
 */
class ScopedDevice {
public:
    /**
     * @brief Construct and switch to the specified device
     * @param device Device index to switch to
     * @throws CudaException if device switch fails
     */
    explicit ScopedDevice(int device);

    /**
     * @brief Destructor restores the original device
     */
    ~ScopedDevice();

    // Non-copyable, non-movable
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    ScopedDevice(ScopedDevice&&) = delete;
    ScopedDevice& operator=(ScopedDevice&&) = delete;

private:
    int saved_device_ = -1;
    int target_device_ = -1;
    bool valid_ = false;
};

}  // namespace cuda::mesh
