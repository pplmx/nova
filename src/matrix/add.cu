#include "cuda/device/device_utils.h"
#include "cuda/memory/buffer.h"
#include "matrix/add.h"

namespace cuda_kernel {

template <typename T>
__global__ void addMatricesKernel(const T* matrixA, const T* matrixB, T* resultMatrix, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        resultMatrix[index] = matrixA[index] + matrixB[index];
    }
}

}  // namespace cuda_kernel

template <typename T>
void addMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix, int numRows, int numCols) {
    const size_t numElements = static_cast<size_t>(numRows) * static_cast<size_t>(numCols);

    cuda::memory::Buffer<T> deviceMatrixA(numElements);
    cuda::memory::Buffer<T> deviceMatrixB(numElements);
    cuda::memory::Buffer<T> deviceResultMatrix(numElements);

    deviceMatrixA.copy_from(hostMatrixA, numElements);
    deviceMatrixB.copy_from(hostMatrixB, numElements);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuda_kernel::addMatricesKernel<<<numBlocks, threadsPerBlock>>>(
        deviceMatrixA.data(), deviceMatrixB.data(), deviceResultMatrix.data(), numRows, numCols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    deviceResultMatrix.copy_to(hostResultMatrix, numElements);
}

template void addMatricesOnGPU<float>(const float*, const float*, float*, int, int);
template void addMatricesOnGPU<double>(const double*, const double*, double*, int, int);
