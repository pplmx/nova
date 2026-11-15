#include "cuda/device/device_utils.h"
#include "cuda/device/reduce_kernels.h"

#include <limits>

namespace cuda::device {

namespace {

/**
 * @brief Identity element for a reduction operation
 *
 * The out-of-range slots of the last block contribute this value, so it must be
 * neutral for the operation: 0 for SUM, the lowest representable value for MAX,
 * and the highest for MIN. Initialising every slot to T{} (zero) was only
 * correct for SUM — for MAX over an all-negative array the padding zero won and
 * the kernel returned 0, and for MIN over an all-positive array it returned 0
 * the same way.
 *
 * The bounds are built with bit operations because std::numeric_limits members
 * are __host__ functions, which a __device__ function cannot call without
 * --expt-relaxed-constexpr.
 */
template <typename T>
__device__ constexpr T reduce_identity(ReduceOp op) {
    if constexpr (std::numeric_limits<T>::is_integer) {
        constexpr T kAllBits = static_cast<T>(~static_cast<T>(0));
        constexpr int kWidth = static_cast<int>(sizeof(T) * 8);
        constexpr T kSignBit = static_cast<T>(static_cast<T>(1) << (kWidth - 1));
        switch (op) {
            case ReduceOp::SUM: return T{};
            case ReduceOp::MAX:
                return std::numeric_limits<T>::is_signed ? kSignBit : T{};
            case ReduceOp::MIN:
                return std::numeric_limits<T>::is_signed ? static_cast<T>(kAllBits ^ kSignBit)
                                                         : kAllBits;
        }
        return T{};
    } else {
        // for float/double, build the largest finite magnitude without calling
        // any __host__ constexpr helper.
        const T huge = static_cast<T>(1e38) * static_cast<T>(1e38);
        switch (op) {
            case ReduceOp::SUM: return T{};
            case ReduceOp::MAX: return -huge;
            case ReduceOp::MIN: return huge;
        }
        return T{};
    }
}

}  // namespace

// Basic parallel reduction kernel
// Each block reduces a segment of size 2*blockDim.x using tree reduction
// Grid size should be ceil(size / (2*blockDim.x)) for full coverage
// __launch_bounds__(REDUCE_BLOCK_SIZE, 2) hints register usage for better occupancy
template <typename T>
__global__ __launch_bounds__(REDUCE_BLOCK_SIZE, 2) void reduce_basic_kernel(const T* input, T* output, size_t size, ReduceOp op) {
    // Shared memory for block-level reduction
    // Size matches blockDim.x for straightforward indexing
    __shared__ T sdata[REDUCE_BLOCK_SIZE];

    // Thread index within block
    const size_t tid = threadIdx.x;

    // Global index - each thread handles 2 elements (i and i+blockDim.x)
    // This 2-element-per-thread pattern hides memory latency
    const size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Initialize to the operation's identity, not zero: zero is only neutral
    // for SUM.
    T val = reduce_identity<T>(op);

    // Load first element (bounds check required for non-multiple-of-2 sizes)
    if (i < size) {
        val = input[i];
    }

    // First level of reduction: combine element i with element i+blockDim.x
    // This happens before shared memory reduction, hiding more memory latency
    if (i + blockDim.x < size) {
        if (op == ReduceOp::SUM) {
            val += input[i + blockDim.x];
        } else if (op == ReduceOp::MAX) {
            val = max(val, input[i + blockDim.x]);
        } else {
            val = min(val, input[i + blockDim.x]);
        }
    }

    // Store to shared memory for block-level reduction
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory
    // Each step halves the number of active threads
    // Loop unrolled by compiler with proper __syncthreads()
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (op == ReduceOp::SUM) {
                sdata[tid] += sdata[tid + s];
            } else if (op == ReduceOp::MAX) {
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            } else {
                sdata[tid] = min(sdata[tid], sdata[tid + s]);
            }
        }
        __syncthreads();
    }

    // Write block result to output array
    // Each block produces one output value
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized reduction using warp-level primitives
// Warp shuffle reduces shared memory traffic and enables faster reductions
// ~30% faster than basic kernel for typical input sizes
template <typename T>
__global__ __launch_bounds__(REDUCE_BLOCK_SIZE, 2) void reduce_optimized_kernel(const T* input, T* output, size_t size, ReduceOp op) {
    // Larger shared memory to hold one value per warp (not per thread)
    __shared__ T sdata[REDUCE_OPTIMIZED_SHMEM_SIZE];

    const size_t tid = threadIdx.x;

    // Same 2-element-per-thread initial load pattern
    const size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T val = reduce_identity<T>(op);
    if (i < size) {
        val = input[i];
    }
    if (i + blockDim.x < size) {
        if (op == ReduceOp::SUM) {
            val += input[i + blockDim.x];
        } else if (op == ReduceOp::MAX) {
            val = max(val, input[i + blockDim.x]);
        } else {
            val = min(val, input[i + blockDim.x]);
        }
    }

    // Warp-level reduction using shuffle instructions
    // This is faster than shared memory reduction within a warp
    val = warp_reduce(val, op);

    // First thread of each warp writes its result to shared memory
    // Threads 0, 32, 64, ... write their warp results
    if (tid % WARP_SIZE == 0) {
        sdata[tid / WARP_SIZE] = val;
    }
    __syncthreads();

    // Final warp reduction across warp results in shared memory
    // Only first WARP_SIZE threads participate
    if (tid < WARP_SIZE) {
        val = (tid < blockDim.x / WARP_SIZE) ? sdata[tid]
                                             : reduce_identity<T>(op);
    }
    val = warp_reduce(val, op);

    // Write final block result
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// Explicit template instantiation for supported types
// Required because kernel is defined in .cu file (NVCC compiles separately)
#define REDUCE_KERNEL_INSTANTIATE(T)                                                 \
    template __global__ void reduce_basic_kernel<T>(const T*, T*, size_t, ReduceOp); \
    template __global__ void reduce_optimized_kernel<T>(const T*, T*, size_t, ReduceOp);

    REDUCE_KERNEL_INSTANTIATE(int)
    REDUCE_KERNEL_INSTANTIATE(float)
    REDUCE_KERNEL_INSTANTIATE(double)
    REDUCE_KERNEL_INSTANTIATE(unsigned int)

}  // namespace cuda::device
