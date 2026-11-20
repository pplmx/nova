#pragma once

/**
 * @file sort.h
 * @brief Device sorting primitives (bitonic and odd-even transposition).
 */

#include "cuda/memory/buffer.h"
#include <cstddef>

namespace cuda::parallel {

    /**
     * @brief Sort by the odd-even transposition method
     *
     * O(n) sequential phases over the whole array; a demonstration and reference
     * path rather than a production sort.
     *
     * @param input  Device buffer holding `size` elements
     * @param output Device buffer that receives the sorted elements
     * @param size   Number of elements to sort
     *
     * @note Unlike bitonicSort, this one has no upper size bound: it launches
     *       fixed-size blocks over the whole array.
     */
    template <typename T>
    void oddEvenSort(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

    /**
     * @brief Sort by the bitonic method
     *
     * Requires no particular input order. The input is padded to the next power of
     * two internally, so any size is accepted; this is the price of a bitonic
     * network.
     *
     * @param input  Device buffer holding `size` elements
     * @param output Device buffer that receives the sorted elements
     * @param size   Number of elements to sort
     *
     * @note `output` must hold at least `size` elements and is not resized.
     *       The temporary padded buffer is `next_power_of_two(size)` elements.
     */
    template <typename T>
    void bitonicSort(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

}  // namespace cuda::parallel
