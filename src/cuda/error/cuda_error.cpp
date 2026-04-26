#include <cuda/error/cuda_error.hpp>
#include <cstdio>

namespace nova::error {

std::string cuda_error_category::message(int ev) const {
    const cudaError_t err = static_cast<cudaError_t>(ev);
    const char* str = cudaGetErrorString(err);
    return str ? std::string(str) : "Unknown CUDA error";
}

std::string_view cuda_error_category::recovery_hint(int ev) const noexcept {
    switch (ev) {
        case 2:  // cudaErrorOutOfMemory
            return "Try reducing batch size, freeing memory, or using a smaller tensor";
        case 4:  // cudaErrorDeinitialized
            return "CUDA driver was shut down - restart CUDA or reinitialize";
        case 1:  // cudaErrorInvalidValue
            return "Check parameter values are within valid ranges";
        case 101:  // cudaErrorInvalidDevice
            return "Verify device ID is valid: nvidia-smi --query-gpu=index,name --format=csv";
        case 100:  // cudaErrorNoDevice
            return "No CUDA device found - check CUDA installation and driver";
        case 3:  // cudaErrorNotInitialized
            return "Call cudaInit() or any CUDA function first to initialize";
        case 201:  // cudaErrorInvalidContext
            return "Invalid device context - call cudaSetDevice() to set a valid context";
        case 301:  // file not found
            return "Check file paths and permissions";
        case 303:  // shared object init failed
            return "Shared library initialization failed - check library dependencies";
        case 500:  // cudaErrorNotReady
            return "Operation not yet complete - wait or check if async operation finished";
        case 700:  // cudaErrorIllegalAddress
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
