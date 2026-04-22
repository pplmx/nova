#include <cmath>
#include <vector>

#include "convolution/conv2d.h"
#include "cuda/algo/kernel_launcher.h"
#include "cuda/memory/buffer.h"

namespace {

    template <typename T>
    __global__ void convolve2DKernel(const T* input, T* output, const T* kernel, size_t width, size_t height, int kernel_size, int half) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        T sum = 0;

        for (int ky = -half; ky <= half; ++ky) {
            for (int kx = -half; kx <= half; ++kx) {
                size_t nx = min(max(static_cast<int>(x) + kx, 0), static_cast<int>(width) - 1);
                size_t ny = min(max(static_cast<int>(y) + ky, 0), static_cast<int>(height) - 1);

                int kidx = (ky + half) * kernel_size + (kx + half);
                sum += input[ny * width + nx] * kernel[kidx];
            }
        }

        output[y * width + x] = sum;
    }

}  // namespace

namespace cuda::algo {

    void convolve2D(const memory::Buffer<float>& input, memory::Buffer<float>& output, const memory::Buffer<float>& kernel, size_t width, size_t height, int kernel_size) {
        const int half = kernel_size / 2;

        detail::KernelLauncher launcher;
        launcher.block({16, 16, 1});
        launcher.grid(detail::calc_grid_2d(width, height, {16, 16, 1}));

        launcher.launch(convolve2DKernel<float>, input.data(), output.data(), kernel.data(), width, height, kernel_size, half);

        launcher.synchronize();
    }

    void createGaussianKernel(memory::Buffer<float>& kernel, int size, float sigma) {
        kernel = memory::Buffer<float>(size * size);

        int half = size / 2;
        float sum = 0.0f;

        std::vector<float> h_kernel(size * size);
        for (int y = -half; y <= half; ++y) {
            for (int x = -half; x <= half; ++x) {
                float value = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
                h_kernel[(y + half) * size + (x + half)] = value;
                sum += value;
            }
        }

        for (int i = 0; i < size * size; ++i) {
            h_kernel[i] /= sum;
        }

        kernel.copy_from(h_kernel.data(), size * size);
    }

    void createBoxKernel(memory::Buffer<float>& kernel, int size) {
        kernel = memory::Buffer<float>(size * size);

        std::vector<float> h_kernel(size * size, 1.0f / (size * size));
        kernel.copy_from(h_kernel.data(), size * size);
    }

    void createSobelKernelX(memory::Buffer<float>& kernel) {
        kernel = memory::Buffer<float>(9);
        float h_kernel[] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
        kernel.copy_from(h_kernel, 9);
    }

    void createSobelKernelY(memory::Buffer<float>& kernel) {
        kernel = memory::Buffer<float>(9);
        float h_kernel[] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};
        kernel.copy_from(h_kernel, 9);
    }

}  // namespace cuda::algo
