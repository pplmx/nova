#include "cuda/algo/kernel_launcher.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "matrix/add.h"

#include <stdexcept>
#include <string>

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

    void matrixAdd(const memory::Buffer<float>& a, const memory::Buffer<float>& b, memory::Buffer<float>& c, int numRows, int numCols) {
        // Reject a shape the kernel cannot cover instead of launching a grid of
        // zero blocks (which CUDA rejects) or reading past the buffers.
        if (numRows <= 0 || numCols <= 0) {
            return;
        }

        const size_t required = static_cast<size_t>(numRows) * static_cast<size_t>(numCols);
        if (a.size() < required || b.size() < required || c.size() < required) {
            throw std::invalid_argument("matrixAdd: buffers must hold at least numRows*numCols = " + std::to_string(required) + " elements");
        }

        detail::KernelLauncher launcher;
        launcher.block({16, 16, 1});
        launcher.grid(detail::calc_grid_2d(numCols, numRows, {16, 16, 1}));

        launcher.launch(addMatricesKernel<float>, a.data(), b.data(), c.data(), numRows, numCols);

        launcher.synchronize();
    }

}  // namespace cuda::algo
