#include <cuda_runtime.h>

#include "cuda/device/device_utils.h"
#include "parallel/histogram.h"

constexpr int HIST_BLOCK_SIZE = 256;
constexpr int HIST_BINS = 256;

__global__ __launch_bounds__(HIST_BLOCK_SIZE, 2) void histogramKernel(const uint8_t* input, uint32_t* histogram, size_t width, size_t height) {
    __shared__ uint32_t temp[HIST_BINS];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t total = width * height * 3;

    if (tid < HIST_BINS) {
        temp[tid] = 0;
    }
    __syncthreads();

    for (size_t i = idx; i < total; i += stride) {
        atomicAdd(&temp[input[i]], 1);
    }
    __syncthreads();

    if (tid < HIST_BINS) {
        atomicAdd(&histogram[tid], temp[tid]);
    }
}

__global__ __launch_bounds__(HIST_BLOCK_SIZE, 2) void histogramPerChannelKernel(const uint8_t* input, uint32_t* histogram_r, uint32_t* histogram_g, uint32_t* histogram_b, size_t width, size_t height) {
    __shared__ uint32_t temp_r[HIST_BINS];
    __shared__ uint32_t temp_g[HIST_BINS];
    __shared__ uint32_t temp_b[HIST_BINS];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t total = width * height;

    if (tid < HIST_BINS) {
        temp_r[tid] = 0;
        temp_g[tid] = 0;
        temp_b[tid] = 0;
    }
    __syncthreads();

    for (size_t i = idx; i < total; i += stride) {
        size_t base = i * 3;
        atomicAdd(&temp_r[input[base]], 1);
        atomicAdd(&temp_g[input[base + 1]], 1);
        atomicAdd(&temp_b[input[base + 2]], 1);
    }
    __syncthreads();

    if (tid < HIST_BINS) {
        atomicAdd(&histogram_r[tid], temp_r[tid]);
        atomicAdd(&histogram_g[tid], temp_g[tid]);
        atomicAdd(&histogram_b[tid], temp_b[tid]);
    }
}

void computeHistogram(const uint8_t* d_input, uint32_t* d_histogram, size_t width, size_t height, int bins) {
    CUDA_CHECK(cudaMemset(d_histogram, 0, bins * sizeof(uint32_t)));

    int block = HIST_BLOCK_SIZE;
    int grid = 256;

    histogramKernel<<<grid, block>>>(d_input, d_histogram, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeHistogramPerChannel(const uint8_t* d_input, uint32_t* d_histogram_r, uint32_t* d_histogram_g, uint32_t* d_histogram_b, size_t width, size_t height) {
    CUDA_CHECK(cudaMemset(d_histogram_r, 0, HIST_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_histogram_g, 0, HIST_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_histogram_b, 0, HIST_BINS * sizeof(uint32_t)));

    int block = HIST_BLOCK_SIZE;
    int grid = 256;

    histogramPerChannelKernel<<<grid, block>>>(d_input, d_histogram_r, d_histogram_g, d_histogram_b, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ __launch_bounds__(256, 2) void histogramEqualizeKernel(const uint8_t* input, uint8_t* output, const uint32_t* histogram, size_t width, size_t height, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = width * height * 3;

    if (idx >= total) {
        return;
    }

    uint32_t cdf = 0;
    uint8_t val = input[idx];

    for (int i = 0; i <= val; ++i) {
        cdf += histogram[i];
    }

    output[idx] = static_cast<uint8_t>(min(255.0f, cdf * scale));
}

void equalizeHistogram(const uint8_t* d_input, uint8_t* d_output, const uint32_t* d_histogram, size_t width, size_t height) {
    size_t total = width * height * 3;
    int block = 256;
    int grid = (total + block - 1) / block;

    uint32_t totalPixels = width * height;
    float scale = 255.0f / (totalPixels * 3);

    histogramEqualizeKernel<<<grid, block>>>(d_input, d_output, d_histogram, width, height, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
