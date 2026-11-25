#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "convolution/conv2d.h"
#include "cuda/algo/kernel_launcher.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace {

    template <typename T>
    __global__ __launch_bounds__(256, 2) void convolve2DKernel(const T* input, T* output, const T* kernel, size_t width, size_t height, int kernel_size, int half) {
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

    namespace {

        /// Rejects kernel sizes the two loops below cannot honour.
        ///
        /// Both walk the kernel from -half to +half, which covers (2*half+1)^2 entries while the buffer
        /// holds size*size. For an odd size those agree. For an even one half = size/2 makes the walk
        /// one row and one column longer than the buffer: size 4 allocates 16 floats and indexes up to
        /// 20, reading past the end into whatever the allocator put there. Nothing checked the
        /// argument, and every existing case passes 3 or 5, so nothing noticed.
        void require_odd_kernel_size(int size, const char* what) {
            if (size <= 0 || size % 2 == 0) {
                throw std::invalid_argument(std::string(what) + ": kernel size must be odd and positive, got " + std::to_string(size));
            }
        }

    }  // namespace

    void convolve2D(const memory::Buffer<float>& input, memory::Buffer<float>& output, const memory::Buffer<float>& kernel, size_t width, size_t height, int kernel_size) {
        require_odd_kernel_size(kernel_size, "convolve2D");
        const int half = kernel_size / 2;

        detail::KernelLauncher launcher;
        launcher.block({16, 16, 1});
        launcher.grid(detail::calc_grid_2d(width, height, {16, 16, 1}));

        launcher.launch(convolve2DKernel<float>, input.data(), output.data(), kernel.data(), width, height, kernel_size, half);

        launcher.synchronize();
    }

    void createGaussianKernel(memory::Buffer<float>& kernel, int size, float sigma) {
        require_odd_kernel_size(size, "createGaussianKernel");
        // A zero or negative sigma divides by zero in the exponent below, which yields inf/NaN weights
        // and a kernel whose sum is NaN, so every convolution with it produces NaN.
        if (!(sigma > 0.0f)) {
            throw std::invalid_argument("createGaussianKernel: sigma must be positive, got " + std::to_string(sigma));
        }
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
        require_odd_kernel_size(size, "createBoxKernel");
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
