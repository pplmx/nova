#pragma once

#include <cuda_runtime.h>
#include <cuda/kernel/cuda_utils.h>
#include <utility>

namespace cuda::memory {

template<typename T>
class unique_ptr {
public:
    unique_ptr() noexcept = default;

    explicit unique_ptr(size_t count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    ~unique_ptr() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;

    unique_ptr(unique_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    unique_ptr& operator=(unique_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    T* release() { T* p = ptr_; ptr_ = nullptr; return p; }
    explicit operator bool() const { return ptr_ != nullptr; }

    void reset(T* ptr = nullptr) {
        if (ptr_) cudaFree(ptr_);
        ptr_ = ptr;
    }

    void swap(unique_ptr& other) noexcept {
        T* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
    }

private:
    T* ptr_ = nullptr;
};

template<typename T>
void swap(unique_ptr<T>& a, unique_ptr<T>& b) noexcept {
    a.swap(b);
}

} // namespace cuda::memory
