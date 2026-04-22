#include "image/gaussian_blur.h"
#include "cuda/device/device_utils.h"
#include "cuda/memory/buffer.h"
#include <vector>

constexpr int MAX_KERNEL_SIZE = 31;

__constant__ float d_kernel_horizontal[MAX_KERNEL_SIZE];
__constant__ float d_kernel_vertical[MAX_KERNEL_SIZE];

__global__ void gaussianBlurHorizontal(const uint8_t* input, float* temp,
                                       size_t width, size_t height,
                                       int kernel_size, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int k = -half; k <= half; ++k) {
            int sx = min(max(static_cast<int>(x) + k, 0), static_cast<int>(width) - 1);
            float weight = d_kernel_horizontal[k + half];
            sum += static_cast<float>(input[(y * width + sx) * 3 + c]) * weight;
            weight_sum += weight;
        }

        temp[(y * width + x) * 3 + c] = sum / weight_sum;
    }
}

__global__ void gaussianBlurVertical(const float* temp, uint8_t* output,
                                     size_t width, size_t height,
                                     int kernel_size, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int k = -half; k <= half; ++k) {
            int sy = min(max(static_cast<int>(y) + k, 0), static_cast<int>(height) - 1);
            float weight = d_kernel_vertical[k + half];
            sum += temp[(sy * width + x) * 3 + c] * weight;
            weight_sum += weight;
        }

        output[(y * width + x) * 3 + c] = static_cast<uint8_t>(
            fminf(255.0f, fmaxf(0.0f, sum / weight_sum)));
    }
}

void gaussianBlur(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  float sigma, int kernel_size) {
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

    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_horizontal, h_kernel.data(), kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_vertical, h_kernel.data(), kernel_size * sizeof(float)));

    cuda::memory::Buffer<float> d_temp(width * height * 3);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    gaussianBlurHorizontal<<<grid, block>>>(d_input, d_temp.data(), width, height, kernel_size, half);
    CUDA_CHECK(cudaGetLastError());

    gaussianBlurVertical<<<grid, block>>>(d_temp.data(), d_output, width, height, kernel_size, half);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}
