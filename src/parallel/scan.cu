#include "parallel/scan.h"
#include "cuda/kernel/cuda_utils.h"

template<typename T>
__global__ void exclusiveScanKernel(const T* input, T* output, size_t size) {
    extern __shared__ T temp[];
    size_t tid = threadIdx.x;

    if (tid < size) {
        temp[tid] = input[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < size; offset *= 2) {
        T val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();

        if (tid >= offset) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    if (tid < size) {
        output[tid] = (tid > 0) ? temp[tid - 1] : 0;
    }
}

template<typename T>
void exclusiveScan(const T* d_input, T* d_output, size_t size) {
    if (size == 0) return;

    if (size > 1024) {
        fprintf(stderr, "Error: exclusiveScan only supports size <= 1024. Got %zu\n", size);
        exit(EXIT_FAILURE);
    }

    exclusiveScanKernel<<<1, 1024, 1024 * sizeof(T)>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
__global__ void inclusiveScanKernel(const T* input, T* output, size_t size) {
    extern __shared__ T temp[];
    size_t tid = threadIdx.x;

    if (tid < size) {
        temp[tid] = input[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < size; offset *= 2) {
        T val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();

        if (tid >= offset) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    if (tid < size) {
        output[tid] = temp[tid];
    }
}

template<typename T>
void inclusiveScan(const T* d_input, T* d_output, size_t size) {
    if (size == 0) return;

    if (size > 1024) {
        fprintf(stderr, "Error: inclusiveScan only supports size <= 1024. Got %zu\n", size);
        exit(EXIT_FAILURE);
    }

    inclusiveScanKernel<<<1, 1024, 1024 * sizeof(T)>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
__global__ void exclusiveScanOptimizedKernel(const T* input, T* output, size_t size) {
    extern __shared__ T temp[];
    size_t tid = threadIdx.x;

    if (tid < size) {
        temp[tid] = input[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < size; offset *= 2) {
        if (tid >= offset) {
            temp[tid] += temp[tid - offset];
        }
        __syncthreads();
    }

    if (tid < size) {
        output[tid] = (tid > 0) ? temp[tid - 1] : 0;
    }
}

template<typename T>
void exclusiveScanOptimized(const T* d_input, T* d_output, size_t size) {
    if (size == 0) return;

    if (size > 1024) {
        fprintf(stderr, "Error: exclusiveScanOptimized only supports size <= 1024. Got %zu\n", size);
        exit(EXIT_FAILURE);
    }

    exclusiveScanOptimizedKernel<<<1, 1024, 1024 * sizeof(T)>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void exclusiveScan<int>(const int*, int*, size_t);
template void inclusiveScan<int>(const int*, int*, size_t);
template void exclusiveScanOptimized<int>(const int*, int*, size_t);
