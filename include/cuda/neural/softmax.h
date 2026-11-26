#pragma once

#include <cuda_runtime.h>

namespace cuda::neural {

    struct SoftmaxResult {
        float* output;
        float* d_output;
        int size;
        int outer_dim;
        int inner_dim;

        SoftmaxResult()
            : output(nullptr),
              d_output(nullptr),
              size(0),
              outer_dim(0),
              inner_dim(0) {}
        explicit SoftmaxResult(int outer_dim, int inner_dim);
        ~SoftmaxResult();

        /// Exclusive ownership, for the same reason as LayerNormResult: the destructor releases the
        /// host array with delete[] and the device array with cudaFree, and the compiler-generated
        /// copy would have made two objects own both -- a double free at scope exit, plus a leak on
        /// copy assignment. Moves transfer ownership and leave the source empty.
        SoftmaxResult(const SoftmaxResult&) = delete;
        SoftmaxResult& operator=(const SoftmaxResult&) = delete;
        SoftmaxResult(SoftmaxResult&& other) noexcept;
        SoftmaxResult& operator=(SoftmaxResult&& other) noexcept;

        void upload();
        void download();
        void clear();
    };

    void softmax(const float* input, float* output, int outer_dim, int inner_dim, cudaStream_t stream = nullptr);

    void softmax_stable(const float* input, float* output, int outer_dim, int inner_dim, cudaStream_t stream = nullptr);

    void log_softmax(const float* input, float* output, int outer_dim, int inner_dim, cudaStream_t stream = nullptr);

}  // namespace cuda::neural
