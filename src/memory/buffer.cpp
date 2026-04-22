#include "cuda/memory/buffer.h"

namespace cuda::memory {

template class Buffer<char>;
template class Buffer<unsigned char>;
template class Buffer<int>;
template class Buffer<unsigned int>;
template class Buffer<float>;
template class Buffer<double>;

} // namespace cuda::memory
