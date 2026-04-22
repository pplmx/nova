#include "image/morphology.h"
#include "cuda/device/device_utils.h"
#include <cuda_runtime.h>

__global__ void sharpenKernel(const uint8_t* input, uint8_t* output,
                               size_t width, size_t height, float strength) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height || x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
        return;
    }

    float laplacian = 0.0f;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            size_t idx = ((y + dy) * width + (x + dx)) * 3;
            float center = static_cast<float>(input[idx]);

            if (dx == 0 && dy == 0) {
                laplacian += 5.0f * center;
            } else {
                laplacian -= center;
            }
        }
    }

        for (int c = 0; c < 3; ++c) {
            size_t idx = (y * width + x) * 3 + c;
            float val = static_cast<float>(input[idx]) - strength * laplacian / 3.0f;
            val = max(0.0f, min(255.0f, val));
            output[idx] = static_cast<uint8_t>(val);
        }
}

void sharpenImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height, float strength) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sharpenKernel<<<grid, block>>>(d_input, d_output, width, height, strength);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void thresholdKernel(const uint8_t* input, uint8_t* output,
                                size_t width, size_t height,
                                uint8_t threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = width * height * 3;

    if (idx < total) {
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

void applyThreshold(const uint8_t* d_input, uint8_t* d_output,
                    size_t width, size_t height, uint8_t threshold) {
    size_t total = width * height * 3;
    int block = 256;
    int grid = (total + block - 1) / block;

    thresholdKernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void erodeKernel(const uint8_t* input, uint8_t* output,
                            size_t width, size_t height, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t minVal = 255;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            size_t nx = min(max((int)x + dx, 0), (int)width - 1);
            size_t ny = min(max((int)y + dy, 0), (int)height - 1);
            size_t idx = (ny * width + nx) * 3;

            for (int c = 0; c < 3; ++c) {
                minVal = min(minVal, input[idx + c]);
            }
        }
    }

    size_t outIdx = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        output[outIdx + c] = minVal;
    }
}

void erodeImage(const uint8_t* d_input, uint8_t* d_output,
                size_t width, size_t height, int kernel_size) {
    int half = kernel_size / 2;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    erodeKernel<<<grid, block>>>(d_input, d_output, width, height, half);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void dilateKernel(const uint8_t* input, uint8_t* output,
                             size_t width, size_t height, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t maxVal = 0;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            size_t nx = min(max((int)x + dx, 0), (int)width - 1);
            size_t ny = min(max((int)y + dy, 0), (int)height - 1);
            size_t idx = (ny * width + nx) * 3;

            for (int c = 0; c < 3; ++c) {
                maxVal = max(maxVal, input[idx + c]);
            }
        }
    }

    size_t outIdx = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        output[outIdx + c] = maxVal;
    }
}

void dilateImage(const uint8_t* d_input, uint8_t* d_output,
                 size_t width, size_t height, int kernel_size) {
    int half = kernel_size / 2;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    dilateKernel<<<grid, block>>>(d_input, d_output, width, height, half);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void openingImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height, int kernel_size) {
    uint8_t* temp;
    CUDA_CHECK(cudaMalloc(&temp, width * height * 3 * sizeof(uint8_t)));

    erodeImage(d_input, temp, width, height, kernel_size);
    dilateImage(temp, d_output, width, height, kernel_size);

    CUDA_CHECK(cudaFree(temp));
}

void closingImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height, int kernel_size) {
    uint8_t* temp;
    CUDA_CHECK(cudaMalloc(&temp, width * height * 3 * sizeof(uint8_t)));

    dilateImage(d_input, temp, width, height, kernel_size);
    erodeImage(temp, d_output, width, height, kernel_size);

    CUDA_CHECK(cudaFree(temp));
}
