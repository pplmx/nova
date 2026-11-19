/**
 * @file cuda_error.hpp
 * @brief CUDA error handling with descriptive messages and recovery hints
 * @defgroup error Error Handling
 * @ingroup device
 *
 * This module provides structured CUDA error handling with:
 * - Descriptive error messages including operation name, file, line, and device context
 * - Recovery hints for common CUDA errors
 * - std::error_code integration for idiomatic C++ error handling
 *
 * Example usage:
 * @code
 * NOVA_CHECK(cudaMalloc(&ptr, size));
 * @endcode
 *
 * @note Include this header to enable enhanced CUDA error reporting
 */

#pragma once

#include <cuda_runtime.h>
#include <system_error>
#include <string_view>
#include <cstdint>

namespace nova::error {

    /**
     * @brief Information about a CUDA error including context
     * @struct cuda_error_info
     * @ingroup error
     *
     * Contains all relevant information about a CUDA error for debugging:
     * - Error code (cudaError_t)
     * - Operation that failed (e.g., "cudaMalloc")
     * - Source file and line number
     * - Device ID where error occurred
     * - Stream (if applicable)
     */
    struct cuda_error_info {
        /** @brief The CUDA error code */
        cudaError_t code;

        /** @brief Name of the operation that failed (e.g., "cudaMalloc") */
        const char* operation{nullptr};

        /** @brief Source file where error occurred */
        const char* file{nullptr};

        /** @brief Line number in source file */
        int line{0};

        /** @brief Device ID where error occurred (-1 if unknown) */
        int device_id{-1};

        /** @brief Stream where error occurred (nullptr if not applicable) */
        void* stream{nullptr};

        /**
         * @brief Get formatted error message
         * @return Human-readable error message including operation, error string, file, and line
         *
         * Example output: "cudaMalloc failed: out of memory at memory.cu:42 (device 0)"
         */
        [[nodiscard]] std::string message() const noexcept;

        /**
         * @brief Get recovery hint for this error
         * @return Actionable suggestion for resolving the error
         *
         * Example: For out-of-memory errors, returns "Try reducing batch size, freeing memory"
         */
        [[nodiscard]] std::string_view recovery_hint() const noexcept;
    };

    /**
     * @brief CUDA error category for std::error_code integration
     * @class cuda_error_category
     * @ingroup error
     *
     * Implements std::error_category to provide CUDA-specific error messages
     * and integrates with standard C++ error handling mechanisms.
     *
     * @note Use cuda_category() to get the singleton instance
     */
    class cuda_error_category : public std::error_category {
    public:
        /**
         * @brief Returns the name of the category
         * @return "cuda"
         */
        [[nodiscard]] const char* name() const noexcept override { return "cuda"; }

        /**
         * @brief Returns the error message for a given error code
         * @param ev Numeric error code (cudaError_t value)
         * @return Human-readable error string from CUDA runtime
         */
        [[nodiscard]] std::string message(int ev) const override;

        /**
         * @brief Returns a recovery hint for a given error
         * @param ev Numeric error code (cudaError_t value)
         * @return Actionable suggestion for resolving the error
         */
        [[nodiscard]] std::string_view recovery_hint(int ev) const noexcept;
    };

    /**
     * @brief Get the CUDA error category instance
     * @return Reference to the singleton cuda_error_category
     * @ingroup error
     *
     * Use this to create std::error_code from cudaError_t:
     * @code
     * std::error_code ec = std::error_code(err, cuda_category());
     * @endcode
     */
    const std::error_category& cuda_category() noexcept;

    /**
     * @brief Create an error_code from a CUDA error with context
     * @param err CUDA error code
     * @param operation Name of the operation (optional)
     * @param file Source file name (optional)
     * @param line Line number (optional)
     * @param device Device ID (optional)
     * @param stream Stream pointer (optional)
     * @return std::error_code for the CUDA error
     * @ingroup error
     *
     * @note This is a convenience function; the context parameters are
     * primarily used by cuda_error_guard for detailed error reporting
     */
    inline std::error_code make_error_code(
        cudaError_t err, const char* operation = nullptr, const char* file = nullptr, int line = 0, int device = -1, void* stream = nullptr) noexcept {
        return std::error_code(static_cast<int>(err), cuda_category());
    }

    /**
     * @brief Exception thrown when CUDA operations fail
     * @class cuda_exception
     * @ingroup error
     *
     * Extends std::system_error with CUDA-specific context including:
     * - Operation name
     * - Source location
     * - Device ID
     * - Stream
     *
     * Example catch block:
     * @code
     * try {
     *     NOVA_CHECK(cudaMalloc(&ptr, size));
     * } catch (const cuda_exception& e) {
     *     std::cerr << e.info().message() << "\n";
     *     std::cerr << "Hint: " << e.info().recovery_hint() << "\n";
     * }
     * @endcode
     */
    class cuda_exception : public std::system_error {
    public:
        /**
         * @brief Construct a cuda_exception
         * @param err CUDA error code
         * @param operation Name of the operation that failed
         * @param file Source file name
         * @param line Line number
         * @param device Device ID
         * @param stream Stream pointer
         */
        explicit cuda_exception(cudaError_t err, const char* operation = nullptr, const char* file = nullptr, int line = 0, int device = -1, void* stream = nullptr);

        /**
         * @brief Get error information
         * @return cuda_error_info struct with full error context
         */
        [[nodiscard]] cuda_error_info info() const noexcept { return info_; }

    private:
        cuda_error_info info_;
    };

    /**
     * @brief RAII guard for checking CUDA errors
     * @class cuda_error_guard
     * @ingroup error
     *
     * Automatically captures operation context and throws cuda_exception on error.
     * Use this in functions that call CUDA APIs to get detailed error reporting.
     *
     * Example:
     * @code
     * void myFunction() {
     *     cuda_error_guard guard("myFunction", current_device);
     *     guard.check(cudaMalloc(&ptr, size));
     *     guard.check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
     *     // If we get here, no CUDA errors occurred
     * }
     * @endcode
     *
     * @note The guard automatically captures the current device ID if not provided
     */
    class cuda_error_guard {
    public:
        /**
         * @brief Construct a cuda_error_guard
         * @param operation Name of the operation being guarded
         * @param device Device ID (-1 to auto-detect)
         * @param stream Stream pointer (nullptr for default stream)
         * @param file Source file name (typically __FILE__)
         * @param line Line number (typically __LINE__)
         */
        cuda_error_guard(const char* operation, int device = -1, void* stream = nullptr, const char* file = nullptr, int line = 0) noexcept;
        ~cuda_error_guard();

        cuda_error_guard(const cuda_error_guard&) = delete;
        cuda_error_guard& operator=(const cuda_error_guard&) = delete;

        /**
         * @brief Check a CUDA return value
         * @param err CUDA error code to check
         * @throws cuda_exception if err != cudaSuccess
         *
         * @note If an error has already been recorded, subsequent checks are ignored
         */
        void check(cudaError_t err);

        /**
         * @brief Check if no error occurred
         * @return true if all checks passed, false if an error was recorded
         */
        [[nodiscard]] bool ok() const noexcept { return ok_; }

        /**
         * @brief Get error information
         * @return cuda_error_info struct with error context
         */
        [[nodiscard]] const cuda_error_info& info() const noexcept { return info_; }

    private:
        cuda_error_info info_;
        bool ok_{true};
    };

/**
 * @brief Check a CUDA call and throw on error
 * @param call CUDA API call to check
 *
 * Example:
 * @code
 * NOVA_CHECK(cudaMalloc(&ptr, size));
 * NOVA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
 * @endcode
 *
 * @note This is the recommended way to check CUDA errors in Nova
 *
 * @def NOVA_CHECK
 * @ingroup error
 */
#define NOVA_CHECK(call)                                                                      \
    do {                                                                                      \
        static_assert(sizeof(#call) > 0, "NOVA_CHECK requires a CUDA call");                  \
        nova::error::cuda_error_guard nova_err_guard{#call, -1, nullptr, __FILE__, __LINE__}; \
        nova_err_guard.check(call);                                                           \
    } while (0)

/**
 * @brief Check a CUDA call with explicit stream context
 * @param call CUDA API call to check
 * @param stream CUDA stream pointer
 *
 * Use this when operating on a specific stream to include stream
 * context in error messages.
 *
 * @def NOVA_CHECK_WITH_STREAM
 * @ingroup error
 */
#define NOVA_CHECK_WITH_STREAM(call, stream)                                                 \
    do {                                                                                     \
        nova::error::cuda_error_guard nova_err_guard{#call, -1, stream, __FILE__, __LINE__}; \
        nova_err_guard.check(call);                                                          \
    } while (0)

/**
 * @brief Alias for NOVA_CHECK for backward compatibility
 * @param call CUDA API call to check
 * @def CUDA_CHECK
 * @ingroup error
 */
#define CUDA_CHECK(call) NOVA_CHECK(call)

}  // namespace nova::error
