/**
 * @file device_mesh.cu
 * @brief DeviceMesh and PeerCapabilityMap implementation
 */

#include "cuda/mesh/device_mesh.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace cuda::mesh {

// Helper macro for CUDA error checking (without context overhead)
#define CUDA_CHECK_MESH(call)                                                    \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            throw ::cuda::device::CudaException(err, __FILE__, __LINE__);       \
        }                                                                         \
    } while (0)

// ============================================================================
// PeerCapabilityMap Implementation
// ============================================================================

void PeerCapabilityMap::initialize(int device_count) {
    if (device_count < 0) {
        throw std::invalid_argument("Device count must be non-negative");
    }

    matrix_.clear();
    matrix_.resize(device_count);

    for (int i = 0; i < device_count; ++i) {
        matrix_[i].resize(device_count, false);
        matrix_[i][i] = true;  // Self-access is always true
    }

    // Query peer access capabilities for all pairs
    for (int i = 0; i < device_count; ++i) {
        for (int j = 0; j < device_count; ++j) {
            if (i != j) {
                int can_access = 0;
                cudaError_t err = cudaDeviceCanAccessPeer(&can_access, i, j);
                if (err == cudaSuccess) {
                    matrix_[i][j] = (can_access != 0);
                } else {
                    // If query fails, assume no access
                    matrix_[i][j] = false;
                }
            }
        }
    }
}

bool PeerCapabilityMap::can_access(int src, int dst) const {
    if (src < 0 || dst < 0 || src >= device_count() || dst >= device_count()) {
        return false;
    }
    return matrix_[src][dst];
}

// ============================================================================
// DeviceMesh Implementation
// ============================================================================

DeviceMesh& DeviceMesh::instance() {
    static DeviceMesh instance;
    return instance;
}

DeviceMesh::DeviceMesh()
    : device_count_(0),
      initialized_(false) {
}

void DeviceMesh::initialize() {
    if (initialized_) {
        return;  // Idempotent
    }

    // Query device count
    cudaError_t err = cudaGetDeviceCount(&device_count_);
    if (err != cudaSuccess) {
        // Single-GPU or no GPU systems may have issues
        device_count_ = 0;
    }

    // Handle single-GPU case gracefully
    if (device_count_ <= 0) {
        device_count_ = 1;  // Assume at least one device for operations
    }

    // Initialize peer capability map
    peer_capabilities_.initialize(device_count_);

    initialized_ = true;
}

std::vector<PeerInfo> DeviceMesh::get_mesh_devices() const {
    std::vector<PeerInfo> devices;
    devices.reserve(device_count_);

    for (int i = 0; i < device_count_; ++i) {
        PeerInfo info;
        info.device_id = i;

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        if (err == cudaSuccess) {
            info.global_memory_bytes = prop.totalGlobalMem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            std::strncpy(info.name, prop.name, sizeof(info.name) - 1);
            info.name[sizeof(info.name) - 1] = '\0';
        } else {
            // Fallback for failed queries
            info.global_memory_bytes = 0;
            info.compute_capability_major = 0;
            info.compute_capability_minor = 0;
            info.name[0] = '\0';
        }

        devices.push_back(info);
    }

    return devices;
}

bool DeviceMesh::can_access_peer(int src, int dst) const {
    // Self-access is always true
    if (src == dst) {
        return true;
    }

    // Bounds check
    if (src < 0 || dst < 0 || src >= device_count_ || dst >= device_count_) {
        return false;
    }

    // Use cached peer capability map
    return peer_capabilities_.can_access(src, dst);
}

// ============================================================================
// ScopedDevice Implementation
// ============================================================================

ScopedDevice::ScopedDevice(int device)
    : saved_device_(-1),
      target_device_(device),
      valid_(false) {

    cudaError_t err = cudaGetDevice(&saved_device_);
    if (err != cudaSuccess) {
        saved_device_ = 0;  // Fallback
    }

    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        // Restore and throw
        cudaSetDevice(saved_device_);
        throw cuda::device::CudaException(err, __FILE__, __LINE__);
    }

    valid_ = true;
}

ScopedDevice::~ScopedDevice() {
    if (valid_ && saved_device_ >= 0) {
        cudaSetDevice(saved_device_);
    }
}

}  // namespace cuda::mesh
