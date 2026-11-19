#pragma once

/**
 * @file buffer.h
 * @brief RAII wrapper for CUDA device memory
 */

#include <cstddef>

namespace cuda::memory {

    template <typename T>
    class Buffer {
    public:
        Buffer();
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
        void fill(const T& value);
        void resize(size_t new_size);

    private:
        T* data_ = nullptr;
        size_t size_ = 0;
    };

    template <>
    class Buffer<void>;

}  // namespace cuda::memory
