#pragma once

/**
 * @file buffer.h
 * @brief RAII wrapper for CUDA device memory
 */

#include <cstddef>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::memory {

/**
 * @class Buffer
 * @brief RAII wrapper for CUDA device memory with automatic memory management.
 *
 * Buffer<T> provides RAII semantics for GPU memory allocation and deallocation.
 * Memory is automatically freed when the Buffer goes out of scope.
 *
 * @tparam T The element type stored in the buffer
 *
 * @example
 * @code
 * // Create a buffer of 1024 integers
 * cuda::memory::Buffer<int> buf(1024);
 *
 * // Copy data from host to device
 * std::vector<int> host_data(1024, 42);
 * buf.copy_from(host_data.data(), 1024);
 *
 * // Use buffer data pointer directly
 * kernel_launcher.launch(my_kernel, buf.data(), buf.size());
 * @endcode
 */
template <typename T>
class Buffer {
public:
    /**
     * @brief Default constructor creates an empty buffer
     */
    Buffer()
        : data_(nullptr),
          size_(0) {}

    /**
     * @brief Allocates GPU memory for the specified number of elements
     * @param count Number of elements to allocate
     * @throws CudaException if allocation fails
     */
    explicit Buffer(size_t count);

    /**
     * @brief Destructor automatically frees GPU memory
     */
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /**
     * @brief Move constructor transfers ownership
     * @param other Buffer to move from
     */
    Buffer(Buffer&& other) noexcept;
    /**
     * @brief Move assignment transfers ownership
     * @param other Buffer to move from
     * @return Reference to this buffer
     */
    Buffer& operator=(Buffer&& other) noexcept;

    /**
     * @brief Returns pointer to device memory
     * @return Raw pointer to GPU memory
     */
    T* data() { return data_; }

    /**
     * @brief Returns const pointer to device memory
     * @return Const raw pointer to GPU memory
     */
    const T* data() const { return data_; }

    /**
     * @brief Returns the number of elements in the buffer
     * @return Element count
     */
    size_t size() const { return size_; }

    /**
     * @brief Releases ownership of memory without freeing
     * @return Raw pointer to the GPU memory (caller takes ownership)
     */
    T* release();

    /**
     * @brief Copies data from host to device
     * @param host_data Pointer to host memory
     * @param count Number of elements to copy
     * @throws CudaException if copy fails
     */
    void copy_from(const T* host_data, size_t count);

    /**
     * @brief Copies data from device to host
     * @param host_data Pointer to host memory (destination)
     * @param count Number of elements to copy
     * @throws CudaException if copy fails
     */
    void copy_to(T* host_data, size_t count) const;

    /**
     * @brief Fills buffer with a constant value
     * @param value Value to fill with
     * @throws CudaException if operation fails
     */
    void fill(const T& value);

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

/**
 * @class Buffer<void>
 * @brief Partial specialization for void pointer handling
 */
template <>
class Buffer<void> {
public:
    /**
     * @brief Allocates GPU memory with specified byte size
     * @param size Number of bytes to allocate
     * @throws CudaException if allocation fails
     */
    explicit Buffer(size_t size)
        : size_(size) {
        CUDA_CHECK(cudaMalloc(&data_, size_));
    }

    /**
     * @brief Default constructor creates empty buffer
     */
    Buffer()
        : data_(nullptr),
          size_(0) {}

    /**
     * @brief Destructor frees GPU memory
     */
    ~Buffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /**
     * @brief Move constructor transfers ownership
     */
    Buffer(Buffer&& other) noexcept
        : data_(other.data_),
          size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment transfers ownership
     */
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

    /**
     * @brief Returns pointer to device memory
     */
    void* data() { return data_; }
    const void* data() const { return data_; }

    /**
     * @brief Returns size in bytes
     */
    size_t size() const { return size_; }

    /**
     * @brief Releases ownership without freeing
     */
    void* release() {
        void* ptr = data_;
        data_ = nullptr;
        size_ = 0;
        return ptr;
    }

    /**
     * @brief Copies data from host to device
     * @param host_data Pointer to host memory
     * @param bytes Number of bytes to copy
     */
    void copy_from(const void* host_data, size_t bytes) { CUDA_CHECK(cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice)); }

    /**
     * @brief Copies data from device to host
     * @param host_data Pointer to host memory (destination)
     * @param bytes Number of bytes to copy
     */
    void copy_to(void* host_data, size_t bytes) const { CUDA_CHECK(cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost)); }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
Buffer<T>::Buffer(size_t count)
    : size_(count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
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
    CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void Buffer<T>::copy_to(T* host_data, size_t count) const {
    CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
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

}  // namespace cuda::memory
