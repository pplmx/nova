#pragma once

#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo {

    /**
     * @note NaN handling differs between the three, following IEEE 754 rather than a house
     * rule:
     *
     *   reduce_sum  propagates NaN (0 + NaN = NaN), so any NaN in the input yields NaN.
     *   reduce_max  IGNORES NaN, because it combines with max(), which returns the non-NaN
     *               operand. An input of nothing but NaN therefore yields the identity
     *               [-infinity], not NaN.
     *   reduce_min  the same, yielding [+infinity].
     *
     * Measured on a two-element all-NaN device buffer: sum = nan, max = -inf, min = inf.
     * The max and min results look like real values and are not; a caller that needs to know
     * whether NaN was present must check for it separately.
     *
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
