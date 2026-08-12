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
    // Heap-allocated singleton that is intentionally never destroyed.
    //
    // A function-local static's destructor runs at process exit, and this
    // destructor tears down per-device CUDA streams / NCCL communicators.
    // Calling into the CUDA driver from a static destructor after an
    // earlier error (e.g. an OOM that left the driver context in a bad
    // state) makes cudaStreamDestroy SEGV inside the driver — this was
    // observed as an intermittent "end-of-run SIGSEGV" with no failing
    // test nearby, and it discards a 1GB core per occurrence.
    //
    // The CUDA driver reclaims all GPU resources when the process exits, so
    // orderly teardown at exit is unnecessary; callers that want to release
    // resources while the process is still alive use destroy() explicitly
    // (already idempotent). Note this deliberately deviates from the DeviceMesh
    // singleton, whose destructor does run at exit — harmless there only because
    // it makes no CUDA calls; here the destructor would call the driver.
    static NcclContext* ctx = new NcclContext();
    return *ctx;
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
    initialize(NcclContextConfig{});
}

void NcclContext::initialize(const NcclContextConfig& config) {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (initialized_) {
        return;  // Idempotent
    }

    // Use provided device IDs or detect from mesh.
    if (!config.device_ids.empty()) {
        device_ids_ = config.device_ids;
        device_count_ = static_cast<int>(device_ids_.size());
    } else {
        initialize_from_mesh();
    }

    create_streams_and_comms();

    initialized_ = true;
}

void NcclContext::initialize_from_mesh() {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();

    device_count_ = mesh.device_count();
    device_ids_.resize(device_count_);
    for (int i = 0; i < device_count_; ++i) {
        device_ids_[i] = i;
    }
}

void NcclContext::create_streams_and_comms() {
    streams_.resize(device_count_);
    communicators_.resize(device_count_);

#if NOVA_NCCL_ENABLED
    // Generate NCCL unique ID for this communicator group.
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));

    // One stream per device (sequential; cudaStreamCreate is thread-hostile).
    for (int i = 0; i < device_count_; ++i) {
        int device = device_ids_[i];
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    // ncclCommInitRank is itself a per-rank collective: rank 0's call blocks
    // until every other rank posts its own init. Initializing four ranks
    // sequentially from one thread would therefore deadlock at rank 0, so
    // each rank's communicator is initialized from its own thread. cudaSetDevice
    // is per-thread state, hence set again here inside each thread.
    std::vector<std::thread> init_threads;
    init_threads.reserve(device_count_);
    std::mutex error_mutex;
    std::exception_ptr first_error;

    for (int i = 0; i < device_count_; ++i) {
        init_threads.emplace_back([&, i]() {
            try {
                CUDA_CHECK(cudaSetDevice(device_ids_[i]));
                NCCL_CHECK(ncclCommInitRank(
                    &communicators_[i],
                    device_count_,
                    unique_id,
                    i  // rank within NCCL group
                ));
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }

    for (auto& t : init_threads) {
        t.join();
    }

    if (first_error) {
        // Best-effort cleanup of partially-created resources so initialize()
        // can be retried on a clean slate. All init threads have joined, so a
        // non-null communicator is fully created and safe to destroy.
        for (auto& comm : communicators_) {
            if (comm != nullptr) {
                ncclCommDestroy(comm);
            }
        }
        for (auto& s : streams_) {
            if (s != nullptr && s != cudaStreamDefault) {
                cudaStreamDestroy(s);
            }
        }
        communicators_.clear();
        streams_.clear();
        std::rethrow_exception(first_error);
    }
#else
    // Without NCCL, create per-device streams so callers can use get_stream();
    // communicators_ stays nullptr and NCCL paths are guarded out.
    for (int i = 0; i < device_count_; ++i) {
        int device = device_ids_[i];
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
#endif
}

ncclComm_t NcclContext::current_comm() const {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!initialized_) {
#if NOVA_NCCL_ENABLED
        throw NcclException("NcclContext not initialized", ncclInvalidArgument,
                            "current_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("NcclContext not initialized");
#endif
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#if NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "current_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return communicators_[std::distance(device_ids_.begin(), it)];
}

ncclComm_t NcclContext::get_comm(int device) const {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!initialized_) {
#if NOVA_NCCL_ENABLED
        throw NcclException("NcclContext not initialized", ncclInvalidArgument,
                            "get_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("NcclContext not initialized");
#endif
    }

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#if NOVA_NCCL_ENABLED
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
#if NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "get_stream", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return streams_[std::distance(device_ids_.begin(), it)];
}

int NcclContext::rank_of_device(int device) const {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!initialized_) {
#if NOVA_NCCL_ENABLED
        throw NcclException("NcclContext not initialized", ncclInvalidArgument,
                            "rank_of_device", __FILE__, __LINE__);
#else
        throw std::runtime_error("NcclContext not initialized");
#endif
    }

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#if NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "rank_of_device", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return static_cast<int>(std::distance(device_ids_.begin(), it));
}

bool NcclContext::mark_comm_aborted(ncclComm_t comm) noexcept {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (comm == nullptr) {
        return false;
    }

    // Null the aborted communicator out of the vector so destroy() never
    // calls ncclCommDestroy on it (UB on an already-aborted comm) — the error
    // layer already released it. Mark broken: has_nccl() goes false and
    // subsequent collectives fail fast via NcclException.
    bool found = false;
    for (auto& c : communicators_) {
        if (c == comm) {
            c = nullptr;
            found = true;
        }
    }
    if (found) {
        broken_.store(true, std::memory_order_release);
    }
    return found;
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

#if NOVA_NCCL_ENABLED
    // Destroy NCCL communicators (aborted ones were nulled by the error layer
    // via mark_comm_aborted, so this only touches live comms).
    for (auto& comm : communicators_) {
        if (comm != nullptr) {
            ncclCommDestroy(comm);
        }
    }
#endif

    communicators_.clear();
    device_count_ = 0;
    device_ids_.clear();
    broken_.store(false, std::memory_order_release);
    initialized_ = false;
}

}  // namespace cuda::nccl
