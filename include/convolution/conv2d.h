#pragma once

#include "cuda/memory/buffer.h"
#include <cstddef>

namespace cuda::algo {

void convolve2D(const memory::Buffer<float>& input,
                memory::Buffer<float>& output,
                const memory::Buffer<float>& kernel,
                size_t width, size_t height,
                int kernel_size);

void createGaussianKernel(memory::Buffer<float>& kernel, int size, float sigma);

void createBoxKernel(memory::Buffer<float>& kernel, int size);

void createSobelKernelX(memory::Buffer<float>& kernel);

void createSobelKernelY(memory::Buffer<float>& kernel);

}
