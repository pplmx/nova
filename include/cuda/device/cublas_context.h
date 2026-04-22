#pragma once

#include <cublas_v2.h>

#include "cuda/memory/buffer.h"

namespace cuda::device {

    class CublasContext {
    public:
        CublasContext() { CUBLAS_CHECK(cublasCreate(&handle_)); }

        ~CublasContext() { cublasDestroy(handle_); }

        CublasContext(const CublasContext&) = delete;
        CublasContext& operator=(const CublasContext&) = delete;

        CublasContext(CublasContext&& other) noexcept
            : handle_(other.handle_) {
            other.handle_ = nullptr;
        }

        CublasContext& operator=(CublasContext&& other) noexcept {
            if (this != &other) {
                if (handle_) {
                    cublasDestroy(handle_);
                }
                handle_ = other.handle_;
                other.handle_ = nullptr;
            }
            return *this;
        }

        [[nodiscard]] cublasHandle_t get() const { return handle_; }

    private:
        cublasHandle_t handle_{nullptr};
    };

}  // namespace cuda::device

namespace cuda::algo {

    void matrixMultiply(const memory::Buffer<float>& a, const memory::Buffer<float>& b, memory::Buffer<float>& c, int m, int n, int k);

    void matrixMultiply(const memory::Buffer<double>& a, const memory::Buffer<double>& b, memory::Buffer<double>& c, int m, int n, int k);

}  // namespace cuda::algo
