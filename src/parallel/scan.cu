#include "cuda/algo/kernel_launcher.h"
#include "parallel/scan.h"

namespace {

    template <typename T>
    __global__ __launch_bounds__(MAX_SCAN_SIZE, 1) void exclusiveScanKernel(const T* input, T* output, size_t size) {
        extern __shared__ T temp[];
        const size_t tid = threadIdx.x;

        if (tid < size) {
            temp[tid] = input[tid];
        } else {
            temp[tid] = T{};
        }
        __syncthreads();

        for (int offset = 1; offset < size; offset *= 2) {
            T val = T{};
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
            output[tid] = (tid > 0) ? temp[tid - 1] : T{};
        }
    }

    template <typename T>
    __global__ __launch_bounds__(MAX_SCAN_SIZE, 1) void inclusiveScanKernel(const T* input, T* output, size_t size) {
        extern __shared__ T temp[];
        const size_t tid = threadIdx.x;

        if (tid < size) {
            temp[tid] = input[tid];
        } else {
            temp[tid] = T{};
        }
        __syncthreads();

        for (int offset = 1; offset < size; offset *= 2) {
            T val = T{};
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

    template <typename T>
    __global__ __launch_bounds__(MAX_SCAN_SIZE, 1) void exclusiveScanOptimizedKernel(const T* input, T* output, size_t size) {
        extern __shared__ T temp[];
        const size_t tid = threadIdx.x;

        if (tid < size) {
            temp[tid] = input[tid];
        } else {
            temp[tid] = T{};
        }
        __syncthreads();

        for (int offset = 1; offset < size; offset *= 2) {
            if (tid >= offset) {
                temp[tid] += temp[tid - offset];
            }
            __syncthreads();
        }

        if (tid < size) {
            output[tid] = (tid > 0) ? temp[tid - 1] : T{};
        }
    }

}  // namespace

namespace cuda::algo {

    template <typename T>
    void exclusiveScan(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size) {
        if (size == 0) {
            return;
        }
        if (size > MAX_SCAN_SIZE) {
            throw ScanSizeException(size, MAX_SCAN_SIZE);
        }

        detail::KernelLauncher launcher;
        launcher.block({MAX_SCAN_SIZE, 1, 1});
        launcher.shared(MAX_SCAN_SIZE * sizeof(T));

        launcher.launch(exclusiveScanKernel<T>, input.data(), output.data(), size);
        launcher.synchronize();
    }

    template <typename T>
    void inclusiveScan(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size) {
        if (size == 0) {
            return;
        }
        if (size > MAX_SCAN_SIZE) {
            throw ScanSizeException(size, MAX_SCAN_SIZE);
        }

        detail::KernelLauncher launcher;
        launcher.block({MAX_SCAN_SIZE, 1, 1});
        launcher.shared(MAX_SCAN_SIZE * sizeof(T));

        launcher.launch(inclusiveScanKernel<T>, input.data(), output.data(), size);
        launcher.synchronize();
    }

    template <typename T>
    void exclusiveScanOptimized(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size) {
        if (size == 0) {
            return;
        }
        if (size > MAX_SCAN_SIZE) {
            throw ScanSizeException(size, MAX_SCAN_SIZE);
        }

        detail::KernelLauncher launcher;
        launcher.block({MAX_SCAN_SIZE, 1, 1});
        launcher.shared(MAX_SCAN_SIZE * sizeof(T));

        launcher.launch(exclusiveScanOptimizedKernel<T>, input.data(), output.data(), size);
        launcher.synchronize();
    }

    template void exclusiveScan<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);
    template void inclusiveScan<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);
    template void exclusiveScanOptimized<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);

}  // namespace cuda::algo
