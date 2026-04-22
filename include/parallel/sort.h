#pragma once

#include <cstddef>

namespace cuda::parallel {

template<typename T>
void oddEvenSort(const T* d_input, T* d_output, size_t size);

template<typename T>
void bitonicSort(const T* d_input, T* d_output, size_t size);

} // namespace cuda::parallel
