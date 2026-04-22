#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace cuda::device {

class CudaException : public std::runtime_error {
public:
    explicit CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(format_error(err, file, line)), error_(err) {}

    cudaError_t error() const noexcept { return error_; }

private:
    cudaError_t error_;

    static std::string format_error(cudaError_t err, const char* file, int line) {
        return std::string(file) + ":" + std::to_string(line) + " - CUDA error: " +
               std::string(cudaGetErrorString(err));
    }
};

class CublasException : public std::runtime_error {
public:
    explicit CublasException(cublasStatus_t status, const char* file, int line)
        : std::runtime_error(format_error(status, file, line)), status_(status) {}

    cublasStatus_t error() const noexcept { return status_; }

private:
    cublasStatus_t status_;

    static std::string format_error(cublasStatus_t status, const char* file, int line) {
        return std::string(file) + ":" + std::to_string(line) + " - cuBLAS error: " +
               std::to_string(static_cast<int>(status));
    }
};

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            throw cuda::device::CudaException(err, __FILE__, __LINE__);          \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            throw cuda::device::CublasException(status, __FILE__, __LINE__);     \
        }                                                                        \
    } while (0)

} // namespace cuda::device
