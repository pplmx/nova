#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/algo/kernel_launcher.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "image/gaussian_blur.h"

namespace {

    __global__ __launch_bounds__(256,
                                 2) void gaussianBlurHorizontal(const uint8_t* input, float* temp, size_t width, size_t height, const float* kernel, int kernel_size, int half) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -half; k <= half; ++k) {
                int sx = min(max(static_cast<int>(x) + k, 0), static_cast<int>(width) - 1);
                float weight = kernel[k + half];
                sum += static_cast<float>(input[(y * width + sx) * 3 + c]) * weight;
                weight_sum += weight;
            }

            temp[(y * width + x) * 3 + c] = sum / weight_sum;
        }
    }

    __global__ __launch_bounds__(256,
                                 2) void gaussianBlurVertical(const float* temp, uint8_t* output, size_t width, size_t height, const float* kernel, int kernel_size, int half) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -half; k <= half; ++k) {
                int sy = min(max(static_cast<int>(y) + k, 0), static_cast<int>(height) - 1);
                float weight = kernel[k + half];
                sum += temp[(sy * width + x) * 3 + c] * weight;
                weight_sum += weight;
            }

            output[(y * width + x) * 3 + c] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, sum / weight_sum)));
        }
    }

}  // namespace

namespace cuda::algo {

    void gaussianBlur(const memory::Buffer<uint8_t>& input, memory::Buffer<uint8_t>& output, size_t width, size_t height, float sigma, int kernel_size) {
        // Neither argument was checked, and both have a way to produce a kernel of NaNs that silently
        // turns the image into garbage:
        //
        //   sigma 0      -> the exponent is -(x*x)/(2*sigma*sigma), which divides by zero. Every
        //                   weight becomes exp(-inf) = 0, sum stays 0, and the normalisation below
        //                   computes 0/0 = NaN.
        //   kernel_size 0 -> the fill loop never runs, sum stays 0, d_kernel is empty, and the two
        //                   passes read and write nothing.
        //
        // A negative size makes std::vector<float> throw a length_error instead of a message about
        // the argument, and an even size -- while the separable passes do run -- gives an asymmetric
        // kernel because the centre index half = size/2 sits one past the true middle.
        if (width == 0 || height == 0) {
            return;
        }
        if (!(sigma > 0.0f)) {
            throw std::invalid_argument("gaussianBlur: sigma must be positive, got " + std::to_string(sigma));
        }
        if (kernel_size <= 0 || kernel_size % 2 == 0) {
            throw std::invalid_argument("gaussianBlur: kernel size must be odd and positive, got " + std::to_string(kernel_size));
        }
        const int half = kernel_size / 2;

        std::vector<float> h_kernel(kernel_size);
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            const int x = i - half;
            h_kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
            sum += h_kernel[i];
        }
        for (int i = 0; i < kernel_size; ++i) {
            h_kernel[i] /= sum;
        }

        memory::Buffer<float> d_kernel(kernel_size);
        d_kernel.copy_from(h_kernel.data(), kernel_size);

        memory::Buffer<float> d_temp(width * height * 3);

        detail::KernelLauncher launcher;
        launcher.block({16, 16, 1});
        launcher.grid(detail::calc_grid_2d(width, height, {16, 16, 1}));

        launcher.launch(gaussianBlurHorizontal, input.data(), d_temp.data(), width, height, d_kernel.data(), kernel_size, static_cast<int>(half));

        launcher.launch(gaussianBlurVertical, d_temp.data(), output.data(), width, height, d_kernel.data(), kernel_size, static_cast<int>(half));

        launcher.synchronize();
    }

}  // namespace cuda::algo
