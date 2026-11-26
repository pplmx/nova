#pragma once

/**
 * @file pinned_memory.h
 * @brief RAII wrapper for page-locked host memory
 */

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::async {

    class PinnedMemoryAllocationError : public std::runtime_error {
    public:
        explicit PinnedMemoryAllocationError(size_t size)
            : std::runtime_error("Failed to allocate " + std::to_string(size) + " bytes of pinned memory"),
              size_(size) {}

        size_t size() const { return size_; }

    private:
        size_t size_;
    };

    template <typename T>
    class PinnedMemory {
    public:
        PinnedMemory() = default;

        explicit PinnedMemory(size_t count)
            : data_(nullptr),
              size_(count) {
            if (count > 0) {
                CUDA_CHECK(cudaMallocHost(&data_, count * sizeof(T)));
            }
        }

        ~PinnedMemory() {
            if (data_) {
                cudaFreeHost(data_);
                data_ = nullptr;
            }
        }

        PinnedMemory(const PinnedMemory&) = delete;
        PinnedMemory& operator=(const PinnedMemory&) = delete;

        PinnedMemory(PinnedMemory&& other) noexcept
            : data_(other.data_),
              size_(other.size_) {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        PinnedMemory& operator=(PinnedMemory&& other) noexcept {
            if (this != &other) {
                if (data_) {
                    cudaFreeHost(data_);
                }
                data_ = other.data_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        T* get() { return data_; }
        const T* get() const { return data_; }

        T* data() { return data_; }
        const T* data() const { return data_; }

        size_t size() const { return size_; }
        size_t size_bytes() const { return size_ * sizeof(T); }

        explicit operator bool() const { return data_ != nullptr; }

        /// Reallocates, leaving the object empty if the new allocation fails.
        ///
        /// The previous form freed the current buffer and then allocated without clearing the
        /// handle between the two:
        ///
        ///     if (data_) { cudaFreeHost(data_); }        // data_ still names the freed block
        ///     size_ = count;
        ///     if (count > 0) { CUDA_CHECK(cudaMallocHost(&data_, ...)); }   // throws here
        ///
        /// A failure therefore left data_ pointing at released memory and size_ already updated, so
        /// the destructor freed the stale handle a second time and the object reported a size it did
        /// not hold. The new allocation is taken first and only then swapped in, so the object is
        /// either fully moved to the new buffer or left empty.
        void reset(size_t count) {
            if (count == 0) {
                if (data_) {
                    cudaFreeHost(data_);
                }
                data_ = nullptr;
                size_ = 0;
                return;
            }

            T* fresh = nullptr;
            CUDA_CHECK(cudaMallocHost(&fresh, count * sizeof(T)));

            if (data_) {
                cudaFreeHost(data_);
            }
            data_ = fresh;
            size_ = count;
        }

        T* release() {
            T* ptr = data_;
            data_ = nullptr;
            size_ = 0;
            return ptr;
        }

    private:
        T* data_ = nullptr;
        size_t size_ = 0;
    };

    template <typename T>
    [[nodiscard]] PinnedMemory<T> make_pinned(size_t count) {
        return PinnedMemory<T>(count);
    }

    template <typename T>
    class PinnedBuffer : public PinnedMemory<T> {
    public:
        using PinnedMemory<T>::PinnedMemory;

        void copy_from(const std::vector<T>& host_data) {
            size_t count = std::min(host_data.size(), this->size());
            if (count > 0 && this->data()) {
                std::copy(host_data.begin(), host_data.begin() + count, this->data());
            }
        }

        void copy_to(std::vector<T>& host_data) const {
            if (!this->data()) {
                return;
            }
            host_data.resize(this->size());
            std::copy(this->data(), this->data() + this->size(), host_data.begin());
        }

        template <typename StreamT>
        void copy_to_device(T* device_ptr, StreamT stream) {
            if (this->data() && device_ptr && this->size_bytes() > 0) {
                CUDA_CHECK(cudaMemcpyAsync(device_ptr, this->data(), this->size_bytes(), cudaMemcpyHostToDevice, stream));
            }
        }

        template <typename StreamT>
        void copy_from_device(const T* device_ptr, StreamT stream) {
            if (this->data() && device_ptr && this->size_bytes() > 0) {
                CUDA_CHECK(cudaMemcpyAsync(this->data(), device_ptr, this->size_bytes(), cudaMemcpyDeviceToHost, stream));
            }
        }
    };

}  // namespace cuda::async
