#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::memory {

    template class Buffer<char>;
    template class Buffer<unsigned char>;
    template class Buffer<int>;
    template class Buffer<unsigned int>;
    template class Buffer<float>;
    template class Buffer<double>;

    // No explicit instantiation of Buffer<void>: that name is already an explicit specialization in
    // buffer-inl.h, so `template class Buffer<void>;` here was silently a no-op ("explicit
    // instantiation ... after an explicit specialization has no effect"). The specialization is what
    // every Buffer<void> resolves to, and it does not need instantiating.

}  // namespace cuda::memory
