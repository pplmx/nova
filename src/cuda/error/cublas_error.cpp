#include <cuda/error/cublas_error.hpp>
#include <cstdio>

namespace nova::error {

    const char* cublas_status_name(cublasStatus_t status) noexcept {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            // These two were missing, so the most common runtime failures both
            // reported "CUBLAS_STATUS_UNKNOWN" in every message.
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                return "CUBLAS_STATUS_UNKNOWN";
        }
    }

    std::string cublas_error_category::message(int ev) const {
        return cublas_status_name(static_cast<cublasStatus_t>(ev));
    }

    std::string_view cublas_error_category::recovery_hint(int ev) const noexcept {
        // Keyed on the cublasStatus_t enumerators. The literal table this replaces
        // was wrong for three of them: 11 is MAPPING_ERROR (not INTERNAL_ERROR),
        // 13 is EXECUTION_FAILED (not INVALID_HANDLE) and 14 is INTERNAL_ERROR
        // (not INVALID_POINTER). CUBLAS_STATUS_INVALID_HANDLE and
        // CUBLAS_STATUS_INVALID_POINTER do not exist in the modern cuBLAS API at
        // all, so those hints could never have been reachable.
        switch (static_cast<cublasStatus_t>(ev)) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "Call cublasInit() or create a handle with cublasCreate()";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "GPU memory allocation failed - try reducing matrix sizes";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "Invalid parameter value - check transposed, side, uplo values";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "cuBLAS not supported on this GPU architecture - check compute capability";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "Failed to access GPU memory - verify pointers are device-allocated, "
                       "not host or page-locked memory that cuBLAS cannot map";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "Kernel execution failed - this usually follows an earlier CUDA error; "
                       "check the last error from your own kernels before this call";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "cuBLAS internal error - try recreating the handle";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "The requested operation or dtype is not supported by this cuBLAS version";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "cuBLAS license check failed - the toolkit installation may be incomplete";
            default:
                return "Consult cuBLAS documentation for this status code";
        }
    }

    std::string_view cublas_error_info::message() const noexcept {
        static thread_local char buffer[512];
        const char* name = cublas_status_name(status);
        if (operation && file) {
            snprintf(buffer, sizeof(buffer), "%s failed: %s at %s:%d", operation, name, file, line);
        } else if (operation) {
            snprintf(buffer, sizeof(buffer), "%s failed: %s", operation, name);
        } else {
            snprintf(buffer, sizeof(buffer), "cuBLAS error: %s", name);
        }
        return buffer;
    }

    std::string_view cublas_error_info::recovery_hint() const noexcept {
        const auto& cat = static_cast<const cublas_error_category&>(cublas_category());
        return cat.recovery_hint(static_cast<int>(status));
    }

    const std::error_category& cublas_category() noexcept {
        static const cublas_error_category instance;
        return instance;
    }

    cublas_exception::cublas_exception(cublasStatus_t status, const char* operation, const char* file, int line)
        : std::system_error(make_error_code(status, operation, file, line), cublas_status_name(status)),
          info_{} {
        info_.status = status;
        info_.operation = operation;
        info_.file = file;
        info_.line = line;
    }

    cublas_error_guard::cublas_error_guard(const char* operation, const char* file, int line) noexcept
        : info_{} {
        info_.operation = operation;
        info_.file = file;
        info_.line = line;
    }

    cublas_error_guard::~cublas_error_guard() = default;

    void cublas_error_guard::check(cublasStatus_t status) {
        info_.status = status;
        if (status != CUBLAS_STATUS_SUCCESS) {
            ok_ = false;
            throw cublas_exception(status, info_.operation, info_.file, info_.line);
        }
    }

}  // namespace nova::error
