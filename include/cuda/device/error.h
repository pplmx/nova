#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace cuda::device {

    class CudaException : public ::std::runtime_error {
    public:
        explicit CudaException(cudaError_t err, const char* file, int line)
            : ::std::runtime_error(format_error(err, file, line)),
              error_(err) {}

        [[nodiscard]] auto error() const noexcept -> cudaError_t { return error_; }

    private:
        cudaError_t error_;

        static auto format_error(cudaError_t err, const char* file, int line) -> ::std::string {
            return ::std::string(file) + ":" + ::std::to_string(line) + " - CUDA error: " + ::std::string(cudaGetErrorString(err));
        }
    };

    class CublasException : public ::std::runtime_error {
    public:
        explicit CublasException(cublasStatus_t status, const char* file, int line)
            : ::std::runtime_error(format_error(status, file, line)),
              status_(status) {}

        [[nodiscard]] auto error() const noexcept -> cublasStatus_t { return status_; }

    private:
        cublasStatus_t status_;

        static auto format_error(cublasStatus_t status, const char* file, int line) -> ::std::string {
            return ::std::string(file) + ":" + ::std::to_string(line) + " - cuBLAS error: " + ::std::to_string(static_cast<int>(status));
        }
    };

    struct OperationContext {
        const char* operation_name = nullptr;
        ::std::variant<size_t, ::std::pair<size_t, size_t>> dimensions;
        int device_id = -1;
        ::std::string extra;
    };

    class CudaExceptionWithContext : public CudaException {
    public:
        explicit CudaExceptionWithContext(cudaError_t err, const char* file, int line,
                                          const OperationContext& ctx)
            : CudaException(err, file, line),
              context_(ctx) {}

        [[nodiscard]] const OperationContext& context() const { return context_; }

    private:
        OperationContext context_;
    };

#define CUDA_CONTEXT(op, dims, device) \
    ::cuda::device::OperationContext{#op, dims, device}

#define CUDA_VALIDATE_SIZE(size, max_size, operation)                                     \
    do {                                                                                \
        if ((size) > (max_size)) {                                                      \
            throw ::cuda::device::CudaExceptionWithContext(                                \
                cudaErrorLaunchFailure, __FILE__, __LINE__,                              \
                ::cuda::device::OperationContext{#operation, static_cast<size_t>(size), 0}); \
        }                                                                               \
    } while (0)

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            throw ::cuda::device::CudaException(err, __FILE__, __LINE__);     \
        }                                                                       \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            throw ::cuda::device::CublasException(status, __FILE__, __LINE__); \
        }                                                                    \
    } while (0)
#endif

}  // namespace cuda::device
