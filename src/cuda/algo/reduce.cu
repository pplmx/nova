#include <algorithm>
#include <vector>

#include "cuda/algo/reduce.h"
#include "cuda/device/reduce_kernels.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/device/error.h"

namespace cuda::algo {

    namespace {
        // Internal reduce implementation - handles kernel launch and result aggregation
        // Two-phase reduction: GPU reduces to block-level results, CPU aggregates final result
        template <typename T>
        T execute_reduce(const T* input, size_t size, bool optimized, cuda::device::ReduceOp op) {
            // Handle empty input - return identity value for the operation
            if (size == 0) {
                return T{};
            }

            // Each block processes 2*blockSize elements (two-pass loading)
            constexpr size_t blockSize = cuda::device::REDUCE_BLOCK_SIZE;

            // Grid size ensures coverage of all elements
            // Rounds up to handle non-multiple-of-blockSize*2 cases
            const size_t gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

            // Allocate output buffer for block-level results
            // Each block produces one value
            cuda::memory::Buffer<T> output(gridSize);

            // Launch chosen kernel variant
            // Basic: simpler, more predictable latency
            // Optimized: warp shuffle reduction, ~30% faster
            if (optimized) {
                cuda::device::reduce_optimized_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
            } else {
                cuda::device::reduce_basic_kernel<T><<<gridSize, blockSize>>>(input, output.data(), size, op);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy block results to host for final aggregation
            // Grid size is typically small (<1000), so host-side reduction is fast
            std::vector<T> h_output(gridSize);
            output.copy_to(h_output.data(), gridSize);

            // Final CPU-side aggregation
            // For large arrays, this is negligible compared to GPU time
            if (op == cuda::device::ReduceOp::SUM) {
                T result = 0;
                for (T val : h_output) {
                    result += val;
                }
                return result;
            } else if (op == cuda::device::ReduceOp::MAX) {
                // Initialize with first element to avoid identity value issues
                T result = h_output[0];
                for (size_t i = 1; i < h_output.size(); ++i) {
                    result = std::max(result, h_output[i]);
                }
                return result;
            } else {
                T result = h_output[0];
                for (size_t i = 1; i < h_output.size(); ++i) {
                    result = std::min(result, h_output[i]);
                }
                return result;
            }
        }
    }  // namespace

    // Public API: basic sum reduction
    // Suitable for most use cases; optimized variant available for hot paths
    template <typename T>
    T reduce_sum(const T* input, size_t size) {
        return execute_reduce(input, size, false, cuda::device::ReduceOp::SUM);
    }

    // Optimized sum reduction using warp-level primitives
    // Recommended for large reductions in performance-critical code
    template <typename T>
    T reduce_sum_optimized(const T* input, size_t size) {
        return execute_reduce(input, size, true, cuda::device::ReduceOp::SUM);
    }

    // Maximum value reduction
    template <typename T>
    T reduce_max(const T* input, size_t size) {
        return execute_reduce(input, size, false, cuda::device::ReduceOp::MAX);
    }

    // Minimum value reduction
    template <typename T>
    T reduce_min(const T* input, size_t size) {
        return execute_reduce(input, size, false, cuda::device::ReduceOp::MIN);
    }

// Explicit template instantiation for supported types
#define REDUCE_ALGO_INSTANTIATE(T)                        \
    template T reduce_sum<T>(const T*, size_t);           \
    template T reduce_sum_optimized<T>(const T*, size_t); \
    template T reduce_max<T>(const T*, size_t);           \
    template T reduce_min<T>(const T*, size_t);

    REDUCE_ALGO_INSTANTIATE(int)
    REDUCE_ALGO_INSTANTIATE(float)
    REDUCE_ALGO_INSTANTIATE(double)
    REDUCE_ALGO_INSTANTIATE(unsigned int)

}  // namespace cuda::algo
