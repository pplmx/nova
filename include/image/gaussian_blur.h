#pragma once
#include <cstdint>

#include "cuda/memory/buffer.h"
#include <cstddef>

namespace cuda::algo {

    void gaussianBlur(const memory::Buffer<uint8_t>& input, memory::Buffer<uint8_t>& output, size_t width, size_t height, float sigma = 1.0f, int kernel_size = 5);

}
