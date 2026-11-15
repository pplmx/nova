/**
 * @file cublas_error.hpp
 * @brief cuBLAS error handling with structured diagnostics
 * @defgroup cublas_error cuBLAS Error Handling
 * @ingroup error
 *
 * This module provides structured cuBLAS error handling with:
 * - Descriptive error messages including operation name, file, and line
 * - Recovery hints for common cuBLAS errors
 * - std::error_code integration for idiomatic C++ error handling
 *
 * Example usage:
 * @code
 * CUBLAS_CHECK(cublasGemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
 *                            m, n, k, &alpha, d_A, lda, d_B, ldb,
 *                            &beta, d_C, ldc));
 * @endcode
 *
 * @see cuda_error.hpp For CUDA runtime error handling
 * @see retry.hpp For automatic retry logic
 */

#pragma once

#include <cublas_v2.h>
#include <system_error>
#include <string_view>

namespace nova::error {

/**
 * @brief Get the name string for a cuBLAS status code
 * @param status The cuBLAS status code
 * @return Human-readable name string (e.g., "CUBLAS_STATUS_SUCCESS")
 * @ingroup cublas_error
 */
const char* cublas_status_name(cublasStatus_t status) noexcept;

/**
 * @brief Information about a cuBLAS error including context
 * @struct cublas_error_info
 * @ingroup cublas_error
 *
 * Contains all relevant information about a cuBLAS error for debugging:
 * - Status code (cublasStatus_t)
 * - Operation that failed (e.g., "cublasGemm_v2")
 * - Source file and line number
 */
struct cublas_error_info {
    /** @brief The cuBLAS status code */
    cublasStatus_t status;

    /** @brief Name of the operation that failed (e.g., "cublasGemm_v2") */
    const char* operation{nullptr};

    /** @brief Source file where error occurred */
    const char* file{nullptr};

    /** @brief Line number in source file */
    int line{0};

    /**
     * @brief Get formatted error message
     * @return Human-readable error message including operation and error string
     * @ingroup cublas_error
     */
    [[nodiscard]] std::string_view message() const noexcept;

    /**
     * @brief Get recovery hint for this error
     * @return Actionable suggestion for resolving the error
     * @ingroup cublas_error
     */
    [[nodiscard]] std::string_view recovery_hint() const noexcept;
};

/**
 * @brief cuBLAS error category for std::error_code integration
 * @class cublas_error_category
 * @ingroup cublas_error
 *
 * Implements std::error_category to provide cuBLAS-specific error messages
 * and integrates with standard C++ error handling mechanisms.
 *
 * @note Use cublas_category() to get the singleton instance
 * @see cuda_error_category For CUDA runtime errors
 */
class cublas_error_category : public std::error_category {
public:
    /**
     * @brief Returns the name of the category
     * @return "cublas"
     */
    [[nodiscard]] const char* name() const noexcept override { return "cublas"; }

    /**
     * @brief Returns the error message for a given status code
     * @param ev Numeric status code (cublasStatus_t value)
     * @return Human-readable error string
     * @ingroup cublas_error
     */
    [[nodiscard]] std::string message(int ev) const override;

    /**
     * @brief Returns a recovery hint for a given status
     * @param ev Numeric status code (cublasStatus_t value)
     * @return Actionable suggestion for resolving the error
     * @ingroup cublas_error
     */
    [[nodiscard]] std::string_view recovery_hint(int ev) const noexcept;
};

/**
 * @brief Get the cuBLAS error category instance
 * @return Reference to the singleton cublas_error_category
 * @ingroup cublas_error
 *
 * Use this to create std::error_code from cublasStatus_t:
 * @code
 * std::error_code ec = std::error_code(status, cublas_category());
 * @endcode
 *
 * @see cuda_category() For CUDA runtime errors
 */
const std::error_category& cublas_category() noexcept;

/**
 * @brief Create an error_code from a cuBLAS status with context
 * @param status cuBLAS status code
 * @param operation Name of the operation (optional)
 * @param file Source file name (optional)
 * @param line Line number (optional)
 * @return std::error_code for the cuBLAS status
 * @ingroup cublas_error
 *
 * @note This is a convenience function; the context parameters are
 * primarily used by cublas_error_guard for detailed error reporting
 */
inline std::error_code make_error_code(cublasStatus_t status,
                                       const char* operation = nullptr,
                                       const char* file = nullptr,
                                       int line = 0) noexcept {
    return std::error_code(static_cast<int>(status), cublas_category());
}

/**
 * @brief Exception thrown when cuBLAS operations fail
 * @class cublas_exception
 * @ingroup cublas_error
 *
 * Extends std::system_error with cuBLAS-specific context including:
 * - Status code
 * - Operation name
 * - Source location
 *
 * Example catch block:
 * @code
 * try {
 *     CUBLAS_CHECK(cublasGemm_v2(handle, ...));
 * } catch (const cublas_exception& e) {
 *     std::cerr << e.info().message() << "\n";
 *     std::cerr << "Hint: " << e.info().recovery_hint() << "\n";
 * }
 * @endcode
 */
class cublas_exception : public std::system_error {
public:
    /**
     * @brief Construct a cublas_exception
     * @param status cuBLAS status code
     * @param operation Name of the operation that failed
     * @param file Source file name
     * @param line Line number
     */
    explicit cublas_exception(cublasStatus_t status, const char* operation = nullptr,
                              const char* file = nullptr, int line = 0);

    /**
     * @brief Get error information
     * @return cublas_error_info struct with full error context
     */
    [[nodiscard]] cublas_error_info info() const noexcept { return info_; }

private:
    cublas_error_info info_;
};

/**
 * @brief RAII guard for checking cuBLAS errors
 * @class cublas_error_guard
 * @ingroup cublas_error
 *
 * Automatically captures operation context and throws cublas_exception on error.
 * Use this in functions that call cuBLAS APIs to get detailed error reporting.
 *
 * Example:
 * @code
 * void computeGemm() {
 *     cublas_error_guard guard("cublasGemm_v2", __FILE__, __LINE__);
 *     guard.check(cublasGemm_v2(handle, ...));
 * }
 * @endcode
 *
 * @note The guard automatically throws on any error status
 * @see CUBLAS_CHECK Macro wrapper for common use case
 */
class cublas_error_guard {
public:
    /**
     * @brief Construct a cublas_error_guard
     * @param operation Name of the operation being guarded
     * @param file Source file name (typically __FILE__)
     * @param line Line number (typically __LINE__)
     */
    cublas_error_guard(const char* operation, const char* file = nullptr, int line = 0) noexcept;
    ~cublas_error_guard();

    cublas_error_guard(const cublas_error_guard&) = delete;
    cublas_error_guard& operator=(const cublas_error_guard&) = delete;

    /**
     * @brief Check a cuBLAS return value
     * @param status cuBLAS status code to check
     * @throws cublas_exception if status != CUBLAS_STATUS_SUCCESS
     *
     * @note If an error has already been recorded, subsequent checks are ignored
     */
    void check(cublasStatus_t status);

    /**
     * @brief Check if no error occurred
     * @return true if all checks passed, false if an error was recorded
     */
    [[nodiscard]] bool ok() const noexcept { return ok_; }

    /**
     * @brief Get error information
     * @return cublas_error_info struct with error context
     */
    [[nodiscard]] const cublas_error_info& info() const noexcept { return info_; }

private:
    cublas_error_info info_;
    bool ok_{true};
};

/**
 * @brief Check a cuBLAS call and throw on error
 * @param call cuBLAS API call to check
 *
 * Example:
 * @code
 * CUBLAS_CHECK(cublasGemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
 *                            m, n, k, &alpha, d_A, lda, d_B, ldb,
 *                            &beta, d_C, ldc));
 * @endcode
 *
 * @note This is the recommended way to check cuBLAS errors in Nova
 *
 * @def CUBLAS_CHECK
 * @ingroup cublas_error
 */
// device/error.h also defines CUBLAS_CHECK, with different semantics (it throws
// CublasException from the call site). Both used to be unconditional, so a
// translation unit including both headers picked whichever came second — the
// include order silently chose the exception type and message format. This
// definition now yields to the other one when both are present.
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) \
    do { \
        static_assert(sizeof(#call) > 0, "CUBLAS_CHECK requires a cuBLAS call"); \
        nova::error::cublas_error_guard nova_cublas_err_guard{#call, __FILE__, __LINE__}; \
        nova_cublas_err_guard.check(call); \
    } while (0)
#endif

} // namespace nova::error
