/**
 * @file nccl_context.cpp
 * @brief NcclContext implementation
 *
 * Provides NCCL communicator management with dependency injection pattern.
 * Per-device communicators are cached and reused for efficiency (D-04).
 */

#include "cuda/nccl/nccl_context.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace cuda::nccl {

// ============================================================================
// NcclException Implementation
// ============================================================================

std::string NcclException::format_message(const char* msg, const char* expr,
                                          const char* file, int line) {
    return std::string(file) + ":" + std::to_string(line) +
           " - NCCL error: " + msg + "\n  Expression: " + expr;
}

// ============================================================================
// NcclContext Implementation
// ============================================================================

NcclContext& NcclContext::instance() {
    static NcclContext ctx;
    return ctx;
}

NcclContext::NcclContext(const NcclContextConfig& config) {
    initialize(config);
}

NcclContext::NcclContext(NcclContext&& other) noexcept
    : device_count_(other.device_count_),
      communicators_(std::move(other.communicators_)),
      streams_(std::move(other.streams_)),
      device_ids_(std::move(other.device_ids_)),
      initialized_(other.initialized_) {
    other.initialized_ = false;
    other.device_count_ = 0;
}

NcclContext& NcclContext::operator=(NcclContext&& other) noexcept {
    if (this != &other) {
        destroy();
        device_count_ = other.device_count_;
        communicators_ = std::move(other.communicators_);
        streams_ = std::move(other.streams_);
        device_ids_ = std::move(other.device_ids_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
        other.device_count_ = 0;
    }
    return *this;
}

NcclContext::~NcclContext() {
    destroy();
}

void NcclContext::initialize() {
    initialize_from_mesh();
}

void NcclContext::initialize(const NcclContextConfig& config) {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (initialized_) {
        return;  // Idempotent
    }

#ifdef NOVA_NCCL_ENABLED
    // Use provided device IDs or detect from mesh
    if (!config.device_ids.empty()) {
        device_ids_ = config.device_ids;
    } else {
        initialize_from_mesh();
        return;
    }

    device_count_ = static_cast<int>(device_ids_.size());
    communicators_.resize(device_count_);
    streams_.resize(device_count_);

    // Generate NCCL unique ID for this communicator group
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));

    // Initialize communicators for each device
    for (int i = 0; i < device_count_; ++i) {
        int device = device_ids_[i];

        // Set current device
        CUDA_CHECK(cudaSetDevice(device));

        // Create stream for this device
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));

        // Initialize NCCL communicator
        NCCL_CHECK(ncclCommInitRank(
            &communicators_[i],
            device_count_,
            unique_id,
            i  // rank within NCCL group
        ));
    }

    initialized_ = true;
#else
    // Without NCCL, just use DeviceMesh for device discovery
    initialize_from_mesh();
#endif
}

void NcclContext::initialize_from_mesh() {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    device_count_ = mesh.device_count();
    device_ids_.resize(device_count_);
    for (int i = 0; i < device_count_; ++i) {
        device_ids_[i] = i;
    }

    // Create streams for each device even without NCCL
    streams_.resize(device_count_);
    for (int i = 0; i < device_count_; ++i) {
        int device = device_ids_[i];
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    communicators_.resize(device_count_, nullptr);

    initialized_ = true;
}

ncclComm_t NcclContext::get_comm(int device) const {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!initialized_) {
#ifdef NOVA_NCCL_ENABLED
        throw NcclException("NcclContext not initialized", ncclInvalidArgument,
                            "get_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("NcclContext not initialized");
#endif
    }

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#ifdef NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "get_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return communicators_[std::distance(device_ids_.begin(), it)];
}

cudaStream_t NcclContext::get_stream(int device) const {
    std::lock_guard<std::mutex> lock(init_mutex_);

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#ifdef NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "get_stream", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return streams_[std::distance(device_ids_.begin(), it)];
}

void NcclContext::destroy() {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!initialized_) {
        return;
    }

    // Destroy streams
    for (auto& stream : streams_) {
        if (stream != cudaStreamDefault && stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
    streams_.clear();

#ifdef NOVA_NCCL_ENABLED
    // Destroy NCCL communicators
    for (auto& comm : communicators_) {
        if (comm != nullptr) {
            ncclCommDestroy(comm);
        }
    }
#endif

    communicators_.clear();
    device_count_ = 0;
    device_ids_.clear();
    initialized_ = false;
}

}  // namespace cuda::nccl
