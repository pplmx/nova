#pragma once

#include <cuda_runtime.h>

namespace cuda::neural {

struct ActivationOptions {
    float alpha = 0.01f;
    float negative_slope = 0.01f;
};

void relu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

void relu_inplace(
    float* data,
    int size,
    cudaStream_t stream = nullptr
);

void leaky_relu(
    const float* input,
    float* output,
    int size,
    float alpha = 0.01f,
    cudaStream_t stream = nullptr
);

void leaky_relu_inplace(
    float* data,
    int size,
    float alpha = 0.01f,
    cudaStream_t stream = nullptr
);

void sigmoid(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

void tanh_activation(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

/**
 * @brief SiLU-gated linear unit: out[i] = silu(gate[i]) * up[i]
 *
 * The gated activation used by the tensor-parallel MLP (TASK-010 / v2.21):
 * gate passes through SiLU (x * sigmoid(x)), then is multiplied elementwise
 * by the up stream, matching the standard transformer FFN gate pattern.
 *
 * @param gate Gate stream [size]
 * @param up Up stream [size]
 * @param output Output stream [size]
 * @param size Element count
 * @param stream CUDA stream (default null = current stream)
 */
void silu_and_mul(
    const float* gate,
    const float* up,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

}  // namespace cuda::neural
