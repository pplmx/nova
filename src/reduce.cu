#include "reduce.h"
#include "cuda_utils.h"
#include <vector>
#include <algorithm>

template<typename T>
__global__ void reduceBasicKernel(const T* input, T* output, size_t size) {
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T sum = 0;
    if (i < size) sum = input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

template<typename T>
T reduceSum(const T* d_input, size_t size) {
    if (size == 0) return T{};

    const int blockSize = 256;
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(T)));

    reduceBasicKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, gridSize * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    T result = 0;
    for (T val : h_output) result += val;
    return result;
}

template<typename T>
__device__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__global__ void reduceOptimizedKernel(const T* input, T* output, size_t size) {
    __shared__ T sdata[32];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) val += input[i + blockDim.x];

    val = warpReduceSum(val);

    if (tid % warpSize == 0) sdata[tid / warpSize] = val;
    __syncthreads();

    if (tid < warpSize) val = (tid < blockDim.x / warpSize) ? sdata[tid] : 0;
    val = warpReduceSum(val);

    if (tid == 0) output[blockIdx.x] = val;
}

template<typename T>
T reduceSumOptimized(const T* d_input, size_t size) {
    if (size == 0) return T{};

    const int blockSize = 256;
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(T)));

    reduceOptimizedKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, gridSize * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    T result = 0;
    for (T val : h_output) result += val;
    return result;
}

template<typename T>
__global__ void reduceMaxKernel(const T* input, T* output, size_t size) {
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) val = max(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

template<typename T>
T reduceMax(const T* d_input, size_t size) {
    if (size == 0) return T{};

    const int blockSize = 256;
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(T)));

    reduceMaxKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, gridSize * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    T result = h_output[0];
    for (size_t i = 1; i < h_output.size(); ++i) {
        result = max(result, h_output[i]);
    }
    return result;
}

template<typename T>
__global__ void reduceMinKernel(const T* input, T* output, size_t size) {
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = 0;
    if (i < size) val = input[i];
    if (i + blockDim.x < size) val = min(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

template<typename T>
T reduceMin(const T* d_input, size_t size) {
    if (size == 0) return T{};

    const int blockSize = 256;
    int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(T)));

    reduceMinKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_output(gridSize);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, gridSize * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    T result = h_output[0];
    for (size_t i = 1; i < h_output.size(); ++i) {
        result = min(result, h_output[i]);
    }
    return result;
}

template int reduceSum(const int* d_input, size_t size);
template float reduceSum(const float* d_input, size_t size);
template double reduceSum(const double* d_input, size_t size);
template unsigned int reduceSum(const unsigned int* d_input, size_t size);

template int reduceSumOptimized(const int* d_input, size_t size);
template float reduceSumOptimized(const float* d_input, size_t size);
template double reduceSumOptimized(const double* d_input, size_t size);
template unsigned int reduceSumOptimized(const unsigned int* d_input, size_t size);

template int reduceMax(const int* d_input, size_t size);
template float reduceMax(const float* d_input, size_t size);
template double reduceMax(const double* d_input, size_t size);
template unsigned int reduceMax(const unsigned int* d_input, size_t size);

template int reduceMin(const int* d_input, size_t size);
template float reduceMin(const float* d_input, size_t size);
template double reduceMin(const double* d_input, size_t size);
template unsigned int reduceMin(const unsigned int* d_input, size_t size);
