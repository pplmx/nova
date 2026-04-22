#include "parallel/sort.h"
#include "cuda/kernel/cuda_utils.h"
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

template<typename T>
__global__ void oddEvenPhaseKernel(const T* input, T* output, size_t size, bool evenPhase) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = evenPhase ? (tid * 2) : (tid * 2 + 1);

    if (i + 1 < size) {
        T a = input[i];
        T b = input[i + 1];
        if (a > b) {
            output[i] = b;
            output[i + 1] = a;
        } else {
            output[i] = a;
            output[i + 1] = b;
        }
    }
}

template<typename T>
__global__ void copyKernel(const T* input, T* output, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input[i];
    }
}

template<typename T>
void oddEvenSort(const T* d_input, T* d_output, size_t size) {
    if (size == 0) return;

    CUDA_CHECK(cudaMemcpy(d_output, d_input, size * sizeof(T), cudaMemcpyHostToDevice));

    T* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(T)));

    const int blockSize = 256;
    dim3 block(blockSize);

    for (size_t phase = 0; phase < size; ++phase) {
        int gridSize = (size + blockSize - 1) / blockSize;
        dim3 grid(gridSize);
        copyKernel<<<grid, block>>>(d_output, d_temp, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        bool evenPhase = (phase % 2 == 0);
        int elementsPerBlock = blockSize * 2;
        int compGridSize = (size + elementsPerBlock - 1) / elementsPerBlock;
        dim3 compGrid(compGridSize);
        oddEvenPhaseKernel<<<compGrid, block>>>(d_temp, d_output, size, evenPhase);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_temp));
}

template<typename T>
__device__ void bitonicCompare(T* data, size_t i, size_t j, bool ascending) {
    size_t ixj = i ^ j;
    if (ixj > i && ixj < blockDim.x * gridDim.x) {
        if ((data[i] > data[ixj]) == ascending) {
            T temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

template<typename T>
__global__ void bitonicSortKernel(T* data, size_t size, int k, int j) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ixj = i ^ j;

    if (ixj > i && ixj < size) {
        bool ascending = ((i & k) == 0);
        if ((data[i] > data[ixj]) == ascending) {
            T temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

static size_t nextPowerOf2(size_t n) {
    size_t p = 1;
    while (p < n) p *= 2;
    return p;
}

template<typename T>
__global__ void padWithMaxKernel(T* data, size_t size, size_t paddedSize, T maxVal) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size && i < paddedSize) {
        data[i] = maxVal;
    }
}

template<typename T>
void bitonicSort(const T* d_input, T* d_output, size_t size) {
    if (size == 0) return;

    size_t paddedSize = nextPowerOf2(size);

    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, paddedSize * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, d_input, size * sizeof(T), cudaMemcpyHostToDevice));

    T maxVal = 0;
    if (std::is_integral<T>::value) {
        maxVal = std::numeric_limits<T>::max();
    }

    if (paddedSize > size) {
        dim3 block(256);
        dim3 grid((paddedSize - size + block.x - 1) / block.x);
        padWithMaxKernel<<<grid, block>>>(d_data, size, paddedSize, maxVal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int k = 2; k <= static_cast<int>(paddedSize); k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<<<1, paddedSize>>>(d_data, paddedSize, k, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(d_output, d_data, size * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

template void bitonicSort<int>(const int*, int*, size_t);
template void bitonicSort<float>(const float*, float*, size_t);

template void oddEvenSort<int>(const int*, int*, size_t);
template void oddEvenSort<float>(const float*, float*, size_t);
