#pragma once

#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo {

    /**
     * @warning input must be a DEVICE pointer. These functions hand it straight to a CUDA
     * kernel; the caller's host pointer is never staged. Passing a std::vector's data()
     * faults inside the kernel instead of raising an error at the call. This was measured:
     * a fuzz target in this tree called reduce_sum with a host vector and faulted.
     *
     * To reduce host data, copy it into a cuda::memory::Buffer first and pass buffer.data().
     */

    template <typename T>
    auto reduce_sum(const T* input, size_t size) -> T;

    template <typename T>
    auto reduce_sum_optimized(const T* input, size_t size) -> T;

    template <typename T>
    auto reduce_max(const T* input, size_t size) -> T;

    template <typename T>
    auto reduce_min(const T* input, size_t size) -> T;

}  // namespace cuda::algo
