#include <limits>
#include <type_traits>

#include "cuda/algo/kernel_launcher.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "parallel/sort.h"

namespace cuda::parallel {

    namespace {

        template <typename T>
        __global__ __launch_bounds__(256, 2) void oddEvenPhaseKernel(const T* input, T* output, size_t size, bool evenPhase) {
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

        template <typename T>
        __global__ __launch_bounds__(256, 2) void copyKernel(const T* input, T* output, size_t size) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                output[i] = input[i];
            }
        }

        template <typename T>
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

        template <typename T>
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

        constexpr size_t nextPowerOf2(size_t n) {
            size_t p = 1;
            while (p < n) {
                p *= 2;
            }
            return p;
        }

        template <typename T>
        __global__ __launch_bounds__(256, 2) void padWithMaxKernel(T* data, size_t size, size_t paddedSize, T maxVal) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= size && i < paddedSize) {
                data[i] = maxVal;
            }
        }

    }  // namespace

    template <typename T>
    void oddEvenSort(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size) {
        if (size == 0) {
            return;
        }

        output.copy_from(input.data(), size);

        constexpr size_t blockSize = 256;
        detail::KernelLauncher launcher;
        launcher.block({blockSize, 1, 1});

        for (size_t phase = 0; phase < size; ++phase) {
            launcher.grid(detail::calc_grid_1d(size, {blockSize, 1, 1}));
            launcher.launch(copyKernel<T>, output.data(), input.data(), size);
            launcher.synchronize();

            bool evenPhase = (phase % 2 == 0);
            // calc_grid_1d divides its argument by block.x, so it takes an ELEMENT count. The
            // argument used to be (size + elementsPerBlock - 1) / elementsPerBlock -- already a
            // block count -- and was therefore divided by the block size a second time. For
            // size = 1024 that gave ceil(2 / 256) = 1 block: 256 threads compared the first 512
            // elements, the rest were never touched, and the function returned normally. Passing
            // the element count directly lets the helper do its own rounding.
            launcher.grid(detail::calc_grid_1d(size, {blockSize, 1, 1}));
            launcher.launch(oddEvenPhaseKernel<T>, input.data(), output.data(), size, evenPhase);
            launcher.synchronize();
        }
    }

    template <typename T>
    void bitonicSort(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size) {
        if (size == 0) {
            return;
        }

        const size_t paddedSize = nextPowerOf2(size);
        memory::Buffer<T> d_data(paddedSize);
        d_data.copy_from(input.data(), size);

        T maxVal = T {};
        if constexpr (std::is_integral_v<T>) {
            maxVal = std::numeric_limits<T>::max();
        }

        if (paddedSize > size) {
            detail::KernelLauncher launcher;
            launcher.block({256, 1, 1});
            launcher.grid(detail::calc_grid_1d(paddedSize - size, {256, 1, 1}));
            launcher.launch(padWithMaxKernel<T>, d_data.data(), size, paddedSize, maxVal);
            launcher.synchronize();
        }

        // One thread per element, but in blocks of at most kMaxThreadsPerBlock rather
        // than a single block of paddedSize. A block cannot exceed 1024 threads, so
        // the old launch ({paddedSize, 1, 1}) was rejected for any input above 1024:
        //
        //   SortTest.SortsMoreElementsThanABlockCanHold (4096 elements)
        //   kernel_launcher.h:100 - CUDA error: invalid configuration argument
        //
        // The kernel already indexes with blockIdx.x * blockDim.x + threadIdx.x, so
        // only the launch shape had to change. Every synchronize() between phases is
        // still required: a pass compares against elements another block may have
        // swapped, so removing the barrier would race.
        constexpr size_t kMaxThreadsPerBlock = 256;
        for (int k = 2; k <= static_cast<int>(paddedSize); k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                detail::KernelLauncher launcher;
                launcher.block({kMaxThreadsPerBlock, 1, 1});
                launcher.grid(detail::calc_grid_1d(paddedSize, {kMaxThreadsPerBlock, 1, 1}));
                launcher.launch(bitonicSortKernel<T>, d_data.data(), paddedSize, k, j);
                launcher.synchronize();
            }
        }

        output.copy_from(d_data.data(), size);
    }

    template void bitonicSort<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);
    template void bitonicSort<float>(const memory::Buffer<float>&, memory::Buffer<float>&, size_t);

    template void oddEvenSort<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);
    template void oddEvenSort<float>(const memory::Buffer<float>&, memory::Buffer<float>&, size_t);

}  // namespace cuda::parallel
