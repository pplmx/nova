#pragma once

/**
 * @file buffer-inl.h
 * @brief Buffer template implementation - include AFTER extern "C" context
 */

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::memory {
namespace detail {
    inline void cuda_check_impl(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            // Align with the canonical CUDA error contract: callers and tests
            // expect cuda::device::CudaException (which derives from
            // std::runtime_error, so existing catch(std::runtime_error) sites
            // remain source-compatible) and can inspect .error().
            throw ::cuda::device::CudaException(err, file, line);
        }
    }
}
}

// Deliberately named NOVA_CUDA_MEM_CHECK (not CUDA_CHECK) to avoid colliding
// with the canonical CUDA_CHECK in cuda/device/error.h (which throws
// CudaException) on the same exception contract, while keeping the macro name
// independent of include order and silencing redefinition warnings.
#define NOVA_CUDA_MEM_CHECK(call) cuda::memory::detail::cuda_check_impl(call, __FILE__, __LINE__)

namespace cuda::memory {

template <typename T>
Buffer<T>::Buffer() = default;

template <typename T>
Buffer<T>::Buffer(size_t count)
    : size_(count) {
    NOVA_CUDA_MEM_CHECK(cudaMalloc(&data_, count * sizeof(T)));
}

template <typename T>
Buffer<T>::~Buffer() {
    if (data_) {
        cudaFree(data_);
    }
}

template <typename T>
Buffer<T>::Buffer(Buffer&& other) noexcept
    : data_(other.data_),
      size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

template <typename T>
Buffer<T>& Buffer<T>::operator=(Buffer<T>&& other) noexcept {
    if (this != &other) {
        if (data_) {
            cudaFree(data_);
        }
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template <typename T>
void Buffer<T>::copy_from(const T* host_data, size_t count) {
    NOVA_CUDA_MEM_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void Buffer<T>::copy_to(T* host_data, size_t count) const {
    NOVA_CUDA_MEM_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
T* Buffer<T>::release() {
    T* ptr = data_;
    data_ = nullptr;
    size_ = 0;
    return ptr;
}

template <typename T>
void Buffer<T>::fill(const T& value) {
    std::vector<T> temp(size_, value);
    copy_from(temp.data(), size_);
}

template <typename T>
void Buffer<T>::resize(size_t new_size) {
    if (new_size == size_) return;
    if (data_) {
        cudaFree(data_);
    }
    size_ = new_size;
    if (size_ > 0) {
        NOVA_CUDA_MEM_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
    } else {
        data_ = nullptr;
    }
}

template <>
class Buffer<void> {
public:
    explicit Buffer(size_t size)
        : size_(size) {
        NOVA_CUDA_MEM_CHECK(cudaMalloc(&data_, size_));
    }

    Buffer()
        : data_(nullptr),
          size_(0) {}

    ~Buffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : data_(other.data_),
          size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFree(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }

    size_t size() const { return size_; }

    void* release() {
        void* ptr = data_;
        data_ = nullptr;
        size_ = 0;
        return ptr;
    }

    void copy_from(const void* host_data, size_t bytes) { NOVA_CUDA_MEM_CHECK(cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice)); }
    void copy_to(void* host_data, size_t bytes) const { NOVA_CUDA_MEM_CHECK(cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost)); }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace cuda::memory
