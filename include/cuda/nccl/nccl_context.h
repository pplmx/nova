#pragma once

/**
 * @file nccl_context.h
 * @brief NCCL context with dependency injection pattern
 *
 * Provides centralized NCCL communicator management with dependency injection
 * for testability and singleton fallback for simple use cases.
 *
 * @example Dependency injection:
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 * ncclComm_t comm = ctx.get_comm(0);
 * cudaStream_t stream = ctx.get_stream(0);
 * @endcode
 *
 * @example Singleton fallback:
 * @code
 * auto& ctx = NcclContext::instance();
 * ctx.initialize();
 * @endcode
 */

#include "cuda/nccl/nccl_types.h"
#include "cuda/device/error.h"
#include "cuda/mesh/device_mesh.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

/**
 * @def NCCL_CHECK(call)
 * @brief Check NCCL call result and throw on error
 *
 * Follows the CUDA_CHECK pattern from cuda/device/error.h.
 * Throws NcclException on failure with file, line, and error details.
 */
#ifdef NOVA_NCCL_ENABLED
#define NCCL_CHECK(call)                                                          \
    do {                                                                          \
        ncclResult_t err = call;                                                  \
        if (err != ncclSuccess) {                                                 \
            throw cuda::nccl::NcclException(                                      \
                ncclGetErrorString(err), err, #call, __FILE__, __LINE__);         \
        }                                                                         \
    } while (0)
#else
#define NCCL_CHECK(call) ((void)0)
#endif

/**
 * @class NcclException
 * @brief Exception thrown on NCCL errors
 *
 * Provides detailed error information including:
 * - Error code (ncclResult_t)
 * - Error message (from ncclGetErrorString)
 * - Failed expression
 * - Source location (file and line)
 */
class NcclException : public std::runtime_error {
public:
    /**
     * @brief Construct NcclException with full error details
     * @param msg Error message from NCCL
     * @param code NCCL error code
     * @param expr The expression that failed
     * @param file Source file
     * @param line Line number
     */
    NcclException(const char* msg, ncclResult_t code,
                  const char* expr, const char* file, int line)
        : std::runtime_error(format_message(msg, expr, file, line)),
          error_code_(code), expression_(expr), file_(file), line_(line) {}

    /**
     * @brief Get the NCCL error code
     * @return ncclResult_t error code
     */
    [[nodiscard]] ncclResult_t error_code() const noexcept { return error_code_; }

    /**
     * @brief Get the failed expression string
     * @return Expression as string
     */
    [[nodiscard]] const char* expression() const noexcept { return expression_; }

    /**
     * @brief Get the source file name
     * @return File name
     */
    [[nodiscard]] const char* file() const noexcept { return file_; }

    /**
     * @brief Get the source line number
     * @return Line number
     */
    [[nodiscard]] int line() const noexcept { return line_; }

private:
    ncclResult_t error_code_;
    const char* expression_;
    const char* file_;
    int line_;

    static std::string format_message(const char* msg, const char* expr,
                                      const char* file, int line);
};

/**
 * @struct NcclContextConfig
 * @brief Configuration options for NcclContext initialization
 */
struct NcclContextConfig {
    /** Specific device IDs to include (empty = all available) */
    std::vector<int> device_ids;

    /** Enable NCCL debug output */
    bool debug = false;

    /** Timeout for NCCL operations in milliseconds */
    int timeout_ms = 30000;
};

/**
 * @class NcclContext
 * @brief NCCL communicator pool with lazy initialization
 *
 * Manages per-device NCCL communicators and provides thread-safe access.
 * Uses dependency injection pattern for testability with singleton fallback
 * for simple cases (per D-01).
 *
 * Thread-safety: All public methods are thread-safe for concurrent access
 * from multiple threads, except during initialization/destruction.
 *
 * @example Dependency injection:
 * @code
 * NcclContext ctx;
 * ctx.initialize();
 * NcclContext::NcclResult result = ctx.get_comm(0);
 * @endcode
 *
 * @example Singleton (simple cases):
 * @code
 * auto& ctx = NcclContext::instance();
 * ctx.initialize();
 * ncclComm_t comm = ctx.get_comm(0);
 * @endcode
 */
class NcclContext {
public:
    /**
     * @brief Get singleton instance for simple cases (per D-01)
     * @return Reference to global NcclContext
     */
    static NcclContext& instance();

    /**
     * @brief Default constructor (creates uninitialized context)
     */
    NcclContext() = default;

    /**
     * @brief Construct with configuration
     * @param config Configuration options
     */
    explicit NcclContext(const NcclContextConfig& config);

    // Non-copyable, movable
    NcclContext(const NcclContext&) = delete;
    NcclContext& operator=(const NcclContext&) = delete;

    /**
     * @brief Move constructor
     */
    NcclContext(NcclContext&& other) noexcept;

    /**
     * @brief Move assignment
     */
    NcclContext& operator=(NcclContext&& other) noexcept;

    /**
     * @brief Destructor cleans up all communicators and streams
     */
    ~NcclContext();

    /**
     * @brief Initialize with default configuration (all devices)
     *
     * Uses DeviceMesh::instance() to discover available devices.
     */
    void initialize();

    /**
     * @brief Initialize with custom configuration
     * @param config Configuration options including specific device IDs
     */
    void initialize(const NcclContextConfig& config);

    /**
     * @brief Get NCCL communicator for a specific device
     * @param device Device index
     * @return NCCL communicator handle
     * @throws NcclException if context not initialized or device not in group
     */
    ncclComm_t get_comm(int device) const;

    /**
     * @brief Get CUDA stream for a specific device
     * @param device Device index
     * @return CUDA stream for NCCL operations on this device
     * @throws NcclException if device not in group
     */
    cudaStream_t get_stream(int device) const;

    /**
     * @brief Check if NCCL context is initialized
     * @return true if initialized
     */
    [[nodiscard]] bool initialized() const { return initialized_; }

    /**
     * @brief Get device count
     * @return Number of devices in NCCL group
     */
    [[nodiscard]] int device_count() const { return device_count_; }

    /**
     * @brief Check if NCCL is available and functional
     * @return true if context has valid communicators
     */
    [[nodiscard]] bool has_nccl() const { return initialized_ && !communicators_.empty(); }

    /**
     * @brief Destroy all communicators and release resources
     *
     * Safe to call multiple times (idempotent after first call).
     */
    void destroy();

    /**
     * @brief Get device IDs in the NCCL group
     * @return Vector of device IDs
     */
    [[nodiscard]] const std::vector<int>& device_ids() const { return device_ids_; }

private:
    void initialize_from_mesh();
    void cleanup();

    int device_count_ = 0;
    std::vector<ncclComm_t> communicators_;
    std::vector<cudaStream_t> streams_;
    std::vector<int> device_ids_;
    bool initialized_ = false;
    mutable std::mutex init_mutex_;
};

}  // namespace cuda::nccl
