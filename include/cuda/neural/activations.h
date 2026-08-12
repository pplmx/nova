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

/**
 * @brief Backward of the SiLU-gated activation (v2.23)
 *
 * out = silu(gate) * up  ⇒
 *   grad_gate[i] = grad_output[i] * up[i] * silu'(gate[i])
 *   grad_up[i]   = grad_output[i] * silu(gate[i])
 * with silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
 */
void silu_and_mul_backward(
    const float* gate,
    const float* up,
    const float* grad_output,
    float* grad_gate,
    float* grad_up,
    int size,
    cudaStream_t stream = nullptr
);

/**
 * @brief Elementwise add: output[i] = a[i] + b[i].
 */
void elementwise_add(
    const float* a,
    const float* b,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

/**
 * @brief In-place transpose helper (row-major) used by the layer backward:
 * output [cols x rows] = input [rows x cols]^T.
 */
void transpose(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream = nullptr
);

}  // namespace cuda::neural
