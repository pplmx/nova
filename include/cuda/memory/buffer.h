#pragma once

#include "cuda/kernel/cuda_utils.h"
#include <cstddef>

namespace cuda::memory {

template<typename T>
class Buffer {
public:
    explicit Buffer(size_t count);

    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    T* release();

    void copy_from(const T* host_data, size_t count);
    void copy_to(T* host_data, size_t count) const;

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

template<>
class Buffer<void> {
public:
    explicit Buffer(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&data_, size_));
    }

    Buffer() : data_(nullptr), size_(0) {}

    ~Buffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
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

    void copy_from(const void* host_data, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice));
    }

    void copy_to(void* host_data, size_t bytes) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost));
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

template<typename T>
Buffer<T>::Buffer(size_t count) : size_(count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
}

template<typename T>
Buffer<T>::~Buffer() {
    if (data_) {
        cudaFree(data_);
    }
}

template<typename T>
Buffer<T>::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

template<typename T>
Buffer<T>& Buffer<T>::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (data_) cudaFree(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template<typename T>
void Buffer<T>::copy_from(const T* host_data, size_t count) {
    CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::copy_to(T* host_data, size_t count) const {
    CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
T* Buffer<T>::release() {
    T* ptr = data_;
    data_ = nullptr;
    size_ = 0;
    return ptr;
}

} // namespace cuda::memory
