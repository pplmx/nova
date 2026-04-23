#include <cuda_runtime.h>

#include "cuda/device/device_utils.h"
#include "matrix/ops.h"

constexpr int TILE_SIZE = 16;

template <typename T>
__global__ __launch_bounds__(256, 2) void transposeKernel(const T* input, T* output, int rows, int cols) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    output[x * rows + y] = input[y * cols + x];
}

template <typename T>
void transposeMatrix(const T* d_input, T* d_output, int rows, int cols) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    transposeKernel<T><<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
__global__ __launch_bounds__(256, 2) void transposeTiledKernel(const T* input, T* output, int rows, int cols) {
    __shared__ T tile[TILE_SIZE][TILE_SIZE];

    size_t x = blockIdx.x * TILE_SIZE + threadIdx.x;
    size_t y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

template <typename T>
void transposeMatrixTiled(const T* d_input, T* d_output, int rows, int cols) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    transposeTiledKernel<T><<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
__global__ __launch_bounds__(256, 2) void elementwiseAddKernel(const T* a, const T* b, T* c, int rows, int cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx < total) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
void matrixElementwiseAdd(const T* d_a, const T* d_b, T* d_c, int rows, int cols) {
    size_t total = rows * cols;
    int block = 256;
    int grid = (total + block - 1) / block;

    elementwiseAddKernel<T><<<grid, block>>>(d_a, d_b, d_c, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
__global__ __launch_bounds__(256, 2) void elementwiseMultiplyKernel(const T* a, const T* b, T* c, int rows, int cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx < total) {
        c[idx] = a[idx] * b[idx];
    }
}

template <typename T>
void matrixElementwiseMultiply(const T* d_a, const T* d_b, T* d_c, int rows, int cols) {
    size_t total = rows * cols;
    int block = 256;
    int grid = (total + block - 1) / block;

    elementwiseMultiplyKernel<T><<<grid, block>>>(d_a, d_b, d_c, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
__global__ __launch_bounds__(256, 2) void scaleKernel(T* matrix, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        matrix[idx] *= scalar;
    }
}

template <typename T>
void matrixScale(T* d_matrix, T scalar, int size) {
    int block = 256;
    int grid = (size + block - 1) / block;

    scaleKernel<T><<<grid, block>>>(d_matrix, scalar, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define MATRIX_OPS_INSTANTIATE(T)                                                 \
    template void transposeMatrix<T>(const T*, T*, int, int);                     \
    template void transposeMatrixTiled<T>(const T*, T*, int, int);                \
    template void matrixElementwiseAdd<T>(const T*, const T*, T*, int, int);      \
    template void matrixElementwiseMultiply<T>(const T*, const T*, T*, int, int); \
    template void matrixScale<T>(T*, T, int);

MATRIX_OPS_INSTANTIATE(float)
MATRIX_OPS_INSTANTIATE(double)
