#include <cuda/error/cuda_error.hpp>
#include <cstdio>

namespace nova::error {

std::string cuda_error_category::message(int ev) const {
    const cudaError_t err = static_cast<cudaError_t>(ev);
    const char* str = cudaGetErrorString(err);
    return str ? std::string(str) : "Unknown CUDA error";
}

std::string_view cuda_error_category::recovery_hint(int ev) const noexcept {
    // Keyed on the cudaError_t enumerators, never on numeric literals. The
    // literal version had drifted: cudaErrorNotReady is 600 (the hint sat on
    // 500, so it never matched), 201 is cudaErrorDeviceUninitialized (not
    // InvalidContext, so the wrong advice was given), 303 does not exist
    // (shared-object symbol-not-found is 302), and 3 has no enumerator at all.
    switch (static_cast<cudaError_t>(ev)) {
        case cudaErrorMemoryAllocation:
            return "Try reducing batch size, freeing memory, or using a smaller tensor";
        case cudaErrorInvalidValue:
            return "Check parameter values are within valid ranges";
        case cudaErrorInvalidDevice:
            return "Verify device ID is valid: nvidia-smi --query-gpu=index,name --format=csv";
        case cudaErrorNoDevice:
            return "No CUDA device found - check CUDA installation and driver";
        case cudaErrorDeviceUninitialized:
            return "The device was not initialized - call cudaFree(0) or cudaSetDevice() "
                   "before allocating, and check for an earlier fatal error";
        case cudaErrorFileNotFound:
            return "Check file paths and permissions";
        case cudaErrorSharedObjectSymbolNotFound:
            return "Shared library initialization failed - check library dependencies";
        case cudaErrorNotReady:
            return "Operation not yet complete - wait or check if async operation finished";
        case cudaErrorIllegalAddress:
            return "Check array indices and memory addresses are valid";
        default:
            return "Consult CUDA documentation for this error code";
    }
}

std::string cuda_error_info::message() const noexcept {
    const char* err_str = cudaGetErrorString(code);
    std::string result;

    if (operation) {
        result = operation;
        result += " failed: ";
    }

    result += err_str ? err_str : "Unknown error";

    if (file && line > 0) {
        result += " at ";
        result += file;
        result += ":";
        result += std::to_string(line);
    }

    if (device_id >= 0) {
        result += " (device ";
        result += std::to_string(device_id);
        result += ")";
    }

    return result;
}

std::string_view cuda_error_info::recovery_hint() const noexcept {
    const auto& cat = static_cast<const cuda_error_category&>(cuda_category());
    return cat.recovery_hint(static_cast<int>(code));
}

const std::error_category& cuda_category() noexcept {
    static const cuda_error_category instance;
    return instance;
}

cuda_exception::cuda_exception(cudaError_t err, const char* operation,
                               const char* file, int line,
                               int device, void* stream)
    : std::system_error(make_error_code(err, operation, file, line, device, stream),
                        "CUDA error"),
      info_{}
{
    info_.code = err;
    info_.operation = operation;
    info_.file = file;
    info_.line = line;
    info_.device_id = device;
    info_.stream = stream;
}

cuda_error_guard::cuda_error_guard(const char* operation, int device,
                                   void* stream, const char* file, int line) noexcept
    : info_{} {
    info_.operation = operation;
    info_.file = file;
    info_.line = line;
    info_.device_id = device;
    info_.stream = stream;
    if (device < 0) {
        cudaGetDevice(&info_.device_id);
    }
}

cuda_error_guard::~cuda_error_guard() {
    if (!ok_) {
        [[maybe_unused]] cudaError_t err = cudaGetLastError();
    }
}

void cuda_error_guard::check(cudaError_t err) {
    info_.code = err;
    if (err != cudaSuccess) {
        ok_ = false;
        throw cuda_exception(err, info_.operation, info_.file, info_.line,
                            info_.device_id, info_.stream);
    }
}

} // namespace nova::error
