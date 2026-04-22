#include "convolution/conv2d.h"
#include "cuda/device/device_utils.h"
#include <cuda_runtime.h>
#include <cmath>

template<typename T>
__global__ void convolve2DKernel(const T* input, T* output,
                                  const T* kernel,
                                  size_t width, size_t height,
                                  int kernel_size, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    T sum = 0;

    for (int ky = -half; ky <= half; ++ky) {
        for (int kx = -half; kx <= half; ++kx) {
            size_t nx = min(max((int)x + kx, 0), (int)width - 1);
            size_t ny = min(max((int)y + ky, 0), (int)height - 1);

            int kidx = (ky + half) * kernel_size + (kx + half);
            sum += input[ny * width + nx] * kernel[kidx];
        }
    }

    output[y * width + x] = sum;
}

template<typename T>
void convolve2D(const T* d_input, T* d_output,
                const T* d_kernel,
                size_t width, size_t height,
                int kernel_size) {
    int half = kernel_size / 2;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    convolve2DKernel<T><<<grid, block>>>(d_input, d_output, d_kernel,
                                          width, height, kernel_size, half);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void createGaussianKernel(float* d_kernel, int size, float sigma) {
    float* h_kernel = new float[size * size];
    int half = size / 2;
    float sum = 0.0f;

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

    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, size * size * sizeof(float),
                          cudaMemcpyHostToDevice));
    delete[] h_kernel;
}

void createBoxKernel(float* d_kernel, int size) {
    float* h_kernel = new float[size * size];
    float value = 1.0f / (size * size);

    for (int i = 0; i < size * size; ++i) {
        h_kernel[i] = value;
    }

    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, size * size * sizeof(float),
                          cudaMemcpyHostToDevice));
    delete[] h_kernel;
}

void createSobelKernelX(float* d_kernel) {
    float h_kernel[] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };

    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void createSobelKernelY(float* d_kernel) {
    float h_kernel[] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };

    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float),
                          cudaMemcpyHostToDevice));
}

template void convolve2D<float>(const float*, float*, const float*,
                                size_t, size_t, int);
template void convolve2D<double>(const double*, double*, const double*,
                                 size_t, size_t, int);
