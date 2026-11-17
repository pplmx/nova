#pragma once

#include <cuda/device/error.h>
#include <cuda_runtime.h>

#include <utility>

namespace cuda::memory {

    /**
     * @class unique_ptr
     * @brief Sole owner of a cudaMalloc'd device buffer
     *
     * Frees with cudaFree in its destructor and is move-only, matching std::unique_ptr. This
     * header had no documenation at all, which matters most for release(): dropping the
     * returned pointer leaks the buffer, because nothing else holds it and cudaFree is the
     * only way back. A test in this tree did exactly that and leaked 4 MB per run until it
     * was found.
     *
     * Constructing with a count of zero leaves the pointer null and does not call
     * cudaMalloc — cudaMalloc(0) returns a null pointer anyway, so both paths behave the
     * same way, but skipping the call keeps the allocator quiet.
     */
    template <typename T>
    class unique_ptr {
    public:
        /// Empty owner: get() is null and nothing is freed.
        unique_ptr() noexcept = default;

        /// Allocates count elements of T on the current device. Throws via CUDA_CHECK if the
        /// allocation fails; a count of zero leaves the pointer null.
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

        unique_ptr(unique_ptr&& other) noexcept
            : ptr_(other.ptr_) {
            other.ptr_ = nullptr;
        }

        unique_ptr& operator=(unique_ptr&& other) noexcept {
            if (this != &other) {
                if (ptr_) {
                    cudaFree(ptr_);
                }
                ptr_ = other.ptr_;
                other.ptr_ = nullptr;
            }
            return *this;
        }

        /// The owned pointer, without transferring ownership.
        T* get() const { return ptr_; }

        /// Relinquishes ownership and returns the pointer. The caller must cudaFree it.
        /// Unlike a Buffer going out of scope, nothing else will: an ignored return value
        /// leaks the allocation outright.
        T* release() {
            T* p = ptr_;
            ptr_ = nullptr;
            return p;
        }
        explicit operator bool() const { return ptr_ != nullptr; }

        /// Frees the current buffer, if any, and takes ownership of ptr (which may be null).
        void reset(T* ptr = nullptr) {
            if (ptr_) {
                cudaFree(ptr_);
            }
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

    template <typename T>
    void swap(unique_ptr<T>& a, unique_ptr<T>& b) noexcept {
        a.swap(b);
    }

}  // namespace cuda::memory
