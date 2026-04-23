#include "cuda/algo/kernel_launcher.h"
#include "cuda/memory/buffer.h"
#include "matrix/add.h"

namespace cuda::algo {

    namespace {

        template <typename T>
        __global__ __launch_bounds__(256, 2) void addMatricesKernel(const T* matrixA, const T* matrixB, T* resultMatrix, int numRows, int numCols) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < numRows && col < numCols) {
                int index = row * numCols + col;
                resultMatrix[index] = matrixA[index] + matrixB[index];
            }
        }

    }  // namespace

    void matrixAdd(memory::Buffer<float> a, memory::Buffer<float> b, memory::Buffer<float> c, int numRows, int numCols) {
        detail::KernelLauncher launcher;
        launcher.block({16, 16, 1});
        launcher.grid(detail::calc_grid_2d(numCols, numRows, {16, 16, 1}));

        launcher.launch(addMatricesKernel<float>, a.data(), b.data(), c.data(), numRows, numCols);

        launcher.synchronize();
    }

}  // namespace cuda::algo
