#pragma once

#include "cuda/memory/buffer.h"
#include <cstddef>

namespace cuda::parallel {

    template <typename T>
    void oddEvenSort(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

    template <typename T>
    void bitonicSort(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

}  // namespace cuda::parallel
