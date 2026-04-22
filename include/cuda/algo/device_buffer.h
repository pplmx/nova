#pragma once

#include "cuda/memory/buffer.h"

namespace cuda::algo {
template<typename T>
using DeviceBuffer = ::cuda::memory::Buffer<T>;
}
