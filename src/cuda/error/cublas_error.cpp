#include <cuda/error/cublas_error.hpp>
#include <cstdio>

namespace nova::error {

const char* cublas_status_name(cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:         return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:    return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:   return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:   return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_INTERNAL_ERROR:  return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:   return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:   return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                            return "CUBLAS_STATUS_UNKNOWN";
    }
}

std::string cublas_error_category::message(int ev) const {
    return cublas_status_name(static_cast<cublasStatus_t>(ev));
}

std::string_view cublas_error_category::recovery_hint(int ev) const noexcept {
    switch (ev) {
        case 1:  // CUBLAS_STATUS_NOT_INITIALIZED
            return "Call cublasInit() or create a handle with cublasCreate()";
        case 3:  // CUBLAS_STATUS_ALLOC_FAILED
            return "GPU memory allocation failed - try reducing matrix sizes";
        case 7:  // CUBLAS_STATUS_INVALID_VALUE
            return "Invalid parameter value - check transposed, side, uplo values";
        case 8:  // CUBLAS_STATUS_ARCH_MISMATCH
            return "cuBLAS not supported on this GPU architecture - check compute capability";
        case 11: // CUBLAS_STATUS_INTERNAL_ERROR
            return "cuBLAS internal error - try recreating the handle";
        case 13: // CUBLAS_STATUS_INVALID_HANDLE
            return "Invalid cuBLAS handle - call cublasCreate() first";
        case 14: // CUBLAS_STATUS_INVALID_POINTER
            return "Invalid pointer - verify pointers are not null and device-allocated";
        default:
            return "Consult cuBLAS documentation for this status code";
    }
}

std::string_view cublas_error_info::message() const noexcept {
    static thread_local char buffer[512];
    const char* name = cublas_status_name(status);
    if (operation && file) {
        snprintf(buffer, sizeof(buffer), "%s failed: %s at %s:%d",
                 operation, name, file, line);
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

cublas_exception::cublas_exception(cublasStatus_t status, const char* operation,
                                   const char* file, int line)
    : std::system_error(make_error_code(status, operation, file, line),
                        cublas_status_name(status)),
      info_{}
{
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

} // namespace nova::error
