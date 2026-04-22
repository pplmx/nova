#pragma once

#include "cuda/algo/device_buffer.h"
#include <vector>
#include <algorithm>

namespace cuda::api {

template<typename T>
class DeviceVector {
public:
    explicit DeviceVector(size_t size = 0) : buffer_(size) {}

    size_t size() const { return buffer_.size(); }
    T* data() { return buffer_.data(); }
    const T* data() const { return buffer_.data(); }

    void resize(size_t new_size) {
        buffer_ = cuda::algo::DeviceBuffer<T>(new_size);
    }

    void copy_from(const std::vector<T>& host_data) {
        buffer_.copy_from(host_data.data(), host_data.size());
    }

    void copy_to(std::vector<T>& host_data) const {
        host_data.resize(size());
        buffer_.copy_to(host_data.data(), size());
    }

    cuda::algo::DeviceBuffer<T>& buffer() { return buffer_; }
    const cuda::algo::DeviceBuffer<T>& buffer() const { return buffer_; }

private:
    cuda::algo::DeviceBuffer<T> buffer_;
};

} // namespace cuda::api
