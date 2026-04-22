#pragma once

#include "cuda/memory/buffer.h"
#include <cstddef>

namespace cuda::algo {

template<typename T>
T reduce_sum(const T* input, size_t size);

template<typename T>
T reduce_sum_optimized(const T* input, size_t size);

template<typename T>
T reduce_max(const T* input, size_t size);

template<typename T>
T reduce_min(const T* input, size_t size);

} // namespace cuda::algo
