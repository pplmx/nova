#include "cuda/algo/kernel_launcher.h"
#include "parallel/scan.h"

namespace cuda::algo {

    namespace detail {

        template <typename T, int KernelId>
        __global__ __launch_bounds__(MAX_SCAN_SIZE, 1) void exclusiveScanKernelImpl(const T* input, T* output, size_t size) {
            extern __shared__ char shared_mem[];
            T* temp = reinterpret_cast<T*>(shared_mem);
            const size_t tid = threadIdx.x;

            if (tid < size) {
                temp[tid] = input[tid];
            } else {
                temp[tid] = T {};
            }
            __syncthreads();

            for (int offset = 1; offset < size; offset *= 2) {
                T val = T {};
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
                output[tid] = (tid > 0) ? temp[tid - 1] : T {};
            }
        }

        template <typename T, int KernelId>
        __global__ __launch_bounds__(MAX_SCAN_SIZE, 1) void inclusiveScanKernelImpl(const T* input, T* output, size_t size) {
            extern __shared__ char shared_mem[];
            T* temp = reinterpret_cast<T*>(shared_mem);
            const size_t tid = threadIdx.x;

            if (tid < size) {
                temp[tid] = input[tid];
            } else {
                temp[tid] = T {};
            }
            __syncthreads();

            for (int offset = 1; offset < size; offset *= 2) {
                T val = T {};
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

        template <typename T, int KernelId>
        __global__ __launch_bounds__(MAX_SCAN_SIZE, 1) void exclusiveScanOptimizedKernelImpl(const T* input, T* output, size_t size) {
            extern __shared__ char shared_mem[];
            T* temp = reinterpret_cast<T*>(shared_mem);
            const size_t tid = threadIdx.x;

            if (tid < size) {
                temp[tid] = input[tid];
            } else {
                temp[tid] = T {};
            }
            __syncthreads();

            // Hillis-Steele inclusive scan. Each step must be split into read, sync,
            // write: without the barrier between the read and the write, thread `tid`
            // reads temp[tid - offset] while thread `tid - offset` may be overwriting it
            // in the same step. compute-sanitizer --tool racecheck reports exactly that
            // on this kernel:
            //     Race reported between Read access at
            //     exclusiveScanOptimizedKernelImpl<int, 2>(...)+0x170
            // The results happened to be right because the value a thread reads is from
            // the previous step whenever the hardware schedules the read first, but that
            // is a scheduling assumption, not a guarantee.
            for (int offset = 1; offset < size; offset *= 2) {
                const T value = (tid >= static_cast<size_t>(offset)) ? temp[tid - offset] : T {};
                __syncthreads();
                if (tid >= static_cast<size_t>(offset)) {
                    temp[tid] += value;
                }
                __syncthreads();
            }

            if (tid < size) {
                output[tid] = (tid > 0) ? temp[tid - 1] : T {};
            }
        }

    }  // namespace detail

    template <typename T>
    void exclusiveScan(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size) {
        if (size == 0) {
            return;
        }
        if (size > MAX_SCAN_SIZE) {
            throw ScanSizeException(size, MAX_SCAN_SIZE);
        }

        cuda::detail::KernelLauncher launcher;
        launcher.block({MAX_SCAN_SIZE, 1, 1});
        launcher.shared(MAX_SCAN_SIZE * sizeof(T));

        launcher.launch(detail::exclusiveScanKernelImpl<T, 0>, input.data(), output.data(), size);
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

        cuda::detail::KernelLauncher launcher;
        launcher.block({MAX_SCAN_SIZE, 1, 1});
        launcher.shared(MAX_SCAN_SIZE * sizeof(T));

        launcher.launch(detail::inclusiveScanKernelImpl<T, 1>, input.data(), output.data(), size);
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

        cuda::detail::KernelLauncher launcher;
        launcher.block({MAX_SCAN_SIZE, 1, 1});
        launcher.shared(MAX_SCAN_SIZE * sizeof(T));

        launcher.launch(detail::exclusiveScanOptimizedKernelImpl<T, 2>, input.data(), output.data(), size);
        launcher.synchronize();
    }

    template void exclusiveScan<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);
    template void inclusiveScan<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);
    template void exclusiveScanOptimized<int>(const memory::Buffer<int>&, memory::Buffer<int>&, size_t);

    template void exclusiveScan<float>(const memory::Buffer<float>&, memory::Buffer<float>&, size_t);
    template void inclusiveScan<float>(const memory::Buffer<float>&, memory::Buffer<float>&, size_t);

}  // namespace cuda::algo
