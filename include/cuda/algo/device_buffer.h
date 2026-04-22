#pragma once

#include "cuda_utils.h"
#include <cstddef>

namespace cuda::algo {

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : size_(count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

    void copy_from(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace cuda::algo
