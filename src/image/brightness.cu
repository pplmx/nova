#include "image/brightness.h"
#include "image/types.h"

__global__ void brightnessContrastKernel(const uint8_t* input, uint8_t* output,
                                         size_t width, size_t height,
                                         float alpha, float beta) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    size_t idx = (y * width + x) * 3;

    for (int c = 0; c < 3; ++c) {
        float value = static_cast<float>(input[idx + c]);
        value = alpha * value + beta;
        value = fminf(255.0f, fmaxf(0.0f, value));
        output[idx + c] = static_cast<uint8_t>(value);
    }
}

void adjustBrightnessContrast(const uint8_t* d_input, uint8_t* d_output,
                              size_t width, size_t height,
                              float alpha, float beta) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    brightnessContrastKernel<<<grid, block>>>(d_input, d_output, width, height, alpha, beta);
    CUDA_CHECK_IMAGE(cudaGetLastError());
    CUDA_CHECK_IMAGE(cudaDeviceSynchronize());
}
