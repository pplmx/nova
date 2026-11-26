#include "cuda/neural/softmax.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>

namespace cuda::neural {

    SoftmaxResult::SoftmaxResult(int outer_dim, int inner_dim)
        : output(nullptr),
          d_output(nullptr),
          outer_dim(outer_dim),
          inner_dim(inner_dim) {
        size = outer_dim * inner_dim;
        // Two allocations, and the second can fail: a throwing constructor does not run its
        // destructor, so output was left with no owner when cudaMalloc failed. The pointers are also
        // set here rather than only in the default constructor -- the guard calls clear(), which
        // delete[]s output, and running that on an indeterminate pointer is how LayerNormResult
        // crashed the test binary last round.
        try {
            output = new float[size];
            CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        } catch (...) {
            clear();
            throw;
        }
    }

    SoftmaxResult::~SoftmaxResult() {
        clear();
    }

    SoftmaxResult::SoftmaxResult(SoftmaxResult&& other) noexcept
        : output(other.output),
          d_output(other.d_output),
          size(other.size),
          outer_dim(other.outer_dim),
          inner_dim(other.inner_dim) {
        other.output = nullptr;
        other.d_output = nullptr;
        other.size = 0;
        other.outer_dim = 0;
        other.inner_dim = 0;
    }

    SoftmaxResult& SoftmaxResult::operator=(SoftmaxResult&& other) noexcept {
        if (this != &other) {
            clear();

            output = other.output;
            d_output = other.d_output;
            size = other.size;
            outer_dim = other.outer_dim;
            inner_dim = other.inner_dim;

            other.output = nullptr;
            other.d_output = nullptr;
            other.size = 0;
            other.outer_dim = 0;
            other.inner_dim = 0;
        }
        return *this;
    }

    void SoftmaxResult::upload() {
        CUDA_CHECK(cudaMemcpy(d_output, output, size * sizeof(float), cudaMemcpyHostToDevice));
    }

    void SoftmaxResult::download() {
        CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void SoftmaxResult::clear() {
        delete[] output;
        output = nullptr;
        if (d_output) {
            cudaFree(d_output);
            d_output = nullptr;
        }
        output = nullptr;
    }

    namespace {

        __global__ void softmax_kernel(const float* input, float* output, int outer_dim, int inner_dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= outer_dim * inner_dim) {
                return;
            }

            int outer_idx = idx / inner_dim;
            int inner_idx = idx % inner_dim;

            float max_val = -INFINITY;
            for (int j = 0; j < inner_dim; ++j) {
                max_val = fmaxf(max_val, input[outer_idx * inner_dim + j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < inner_dim; ++j) {
                sum += expf(input[outer_idx * inner_dim + j] - max_val);
            }

            output[idx] = expf(input[idx] - max_val) / sum;
        }

        __global__ void softmax_stable_kernel(const float* input, float* output, int outer_dim, int inner_dim) {
            extern __shared__ float shared_max[];
            int tid = threadIdx.x;
            int block_offset = blockIdx.x * inner_dim;

            float thread_max = -INFINITY;
            for (int i = tid; i < inner_dim; i += blockDim.x) {
                thread_max = fmaxf(thread_max, input[block_offset + i]);
            }
            shared_max[tid] = thread_max;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
                }
                __syncthreads();
            }

            float max_val = shared_max[0];

            __shared__ float shared_sum[256];
            float thread_sum = 0.0f;
            for (int i = tid; i < inner_dim; i += blockDim.x) {
                thread_sum += expf(input[block_offset + i] - max_val);
            }
            shared_sum[tid] = thread_sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_sum[tid] += shared_sum[tid + s];
                }
                __syncthreads();
            }

            float sum = shared_sum[0];

            for (int i = tid; i < inner_dim; i += blockDim.x) {
                output[block_offset + i] = expf(input[block_offset + i] - max_val) / sum;
            }
        }

        /// log(softmax(x)) computed as (x - max) - log(sum), which keeps the values finite where the
        /// softmax itself would underflow. Shares the reduction structure with softmax_stable_kernel;
        /// log_softmax() used to call that kernel instead, so it returned probabilities.
        __global__ void log_softmax_kernel(const float* input, float* output, int outer_dim, int inner_dim) {
            extern __shared__ float shared_max[];
            int tid = threadIdx.x;
            int block_offset = blockIdx.x * inner_dim;

            float thread_max = -INFINITY;
            for (int i = tid; i < inner_dim; i += blockDim.x) {
                thread_max = fmaxf(thread_max, input[block_offset + i]);
            }
            shared_max[tid] = thread_max;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
                }
                __syncthreads();
            }

            float max_val = shared_max[0];

            __shared__ float shared_sum[256];
            float thread_sum = 0.0f;
            for (int i = tid; i < inner_dim; i += blockDim.x) {
                thread_sum += expf(input[block_offset + i] - max_val);
            }
            shared_sum[tid] = thread_sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_sum[tid] += shared_sum[tid + s];
                }
                __syncthreads();
            }

            const float log_sum = logf(shared_sum[0]);

            for (int i = tid; i < inner_dim; i += blockDim.x) {
                output[block_offset + i] = (input[block_offset + i] - max_val) - log_sum;
            }
        }

    }  // anonymous namespace

    void softmax(const float* input, float* output, int outer_dim, int inner_dim, cudaStream_t stream) {
        // The kernel writes one OUTPUT ELEMENT per thread (idx runs over
        // outer_dim * inner_dim), so the grid must cover that many. It used to launch
        // outer_dim blocks, which is 256x too few: measured on a 64x1024 input, only the
        // first 16384 of 65536 elements were written and the rest kept whatever the output
        // buffer held. The row sums of the result were off by up to 1.0.
        //
        // Each thread still walks the whole row to compute its own element's denominator,
        // so this is O(inner) work per element. That is why it is 195x slower than
        // softmax_stable (6.62 ms vs 0.034 ms on the same input) even when correct;
        // softmax_stable is the entry point to use for anything larger than a toy.
        if (outer_dim <= 0 || inner_dim <= 0) {
            return;
        }

        const long long element_count = static_cast<long long>(outer_dim) * static_cast<long long>(inner_dim);
        const int block_size = 256;
        const int grid_size = static_cast<int>((element_count + block_size - 1) / block_size);

        softmax_kernel<<<grid_size, block_size, 0, stream>>>(input, output, outer_dim, inner_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_stable(const float* input, float* output, int outer_dim, int inner_dim, cudaStream_t stream) {
        // softmax() next door already returned early for this; this entry point launched
        // <<<outer_dim, ...>>> regardless, and a grid dimension of zero or less is an
        // invalid configuration -- reported by CUDA_CHECK as cudaErrorInvalidConfiguration
        // from a line that does not say which argument was wrong. Measured.
        if (outer_dim <= 0 || inner_dim <= 0) {
            return;
        }

        int block_size = 256;
        int shared_mem = block_size * sizeof(float) * 2;

        softmax_stable_kernel<<<outer_dim, block_size, shared_mem, stream>>>(input, output, outer_dim, inner_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax(const float* input, float* output, int outer_dim, int inner_dim, cudaStream_t stream) {
        // Same missing guard as softmax_stable above: grid_size comes straight from
        // outer_dim, so a non-positive value is an invalid launch.
        if (outer_dim <= 0 || inner_dim <= 0) {
            return;
        }

        const int grid_size = outer_dim;
        const int block_size = 256;
        // The dynamic shared-memory size was passed as 0 while the kernel writes shared_max[tid]
        // for every thread, and softmax_stable -- the other caller of the same kernel -- passes
        // block_size * sizeof(float) * 2. A zero-sized dynamic allocation with those writes is an
        // out-of-bounds access.
        const int shared_mem = block_size * sizeof(float) * 2;

        // This launched softmax_stable_kernel, so it produced exp(x - max) / sum rather than the
        // logarithm of it: log_softmax of two equal classes came back as {0.5, 0.5} where
        // {-0.693, -0.693} is the answer. log_softmax_kernel is the one that computes x - max - log(sum).
        log_softmax_kernel<<<grid_size, block_size, shared_mem, stream>>>(input, output, outer_dim, inner_dim);
        CUDA_CHECK(cudaGetLastError());
    }

}  // namespace cuda::neural
