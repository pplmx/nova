#include "cuda/neural/activations.h"

#include "cuda/device/error.h"

#include <cmath>

namespace cuda::neural {

namespace {

__global__ void relu_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_inplace_kernel(
    float* data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void leaky_relu_kernel(
    const float* input,
    float* output,
    int size,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void leaky_relu_inplace_kernel(
    float* data,
    int size,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void sigmoid_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void tanh_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void silu_and_mul_kernel(
    const float* gate,
    const float* up,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        // out = silu(gate) * up = (gate * sigmoid(gate)) * up
        output[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

__global__ void silu_and_mul_backward_kernel(
    const float* gate,
    const float* up,
    const float* grad_output,
    float* grad_gate,
    float* grad_up,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float g = gate[idx];
        const float u = up[idx];
        const float d = grad_output[idx];
        const float sig = 1.0f / (1.0f + expf(-g));
        // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        const float silu_prime = sig * (1.0f + g * (1.0f - sig));
        grad_gate[idx] = d * u * silu_prime;
        grad_up[idx] = d * (g * sig);
    }
}

__global__ void elementwise_add_kernel(
    const float* a,
    const float* b,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

__global__ void transpose_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    // input [rows x cols] row-major -> output [cols x rows] row-major.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int r = idx / cols;
        const int c = idx % cols;
        output[c * rows + r] = input[r * cols + c];
    }
}

}  // anonymous namespace

void relu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void relu_inplace(
    float* data,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_inplace_kernel<<<grid_size, block_size, 0, stream>>>(
        data, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu(
    const float* input,
    float* output,
    int size,
    float alpha,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size, alpha
    );
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu_inplace(
    float* data,
    int size,
    float alpha,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    leaky_relu_inplace_kernel<<<grid_size, block_size, 0, stream>>>(
        data, size, alpha
    );
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void tanh_activation(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    tanh_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void silu_and_mul(
    const float* gate,
    const float* up,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    silu_and_mul_kernel<<<grid_size, block_size, 0, stream>>>(
        gate, up, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void silu_and_mul_backward(
    const float* gate,
    const float* up,
    const float* grad_output,
    float* grad_gate,
    float* grad_up,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    silu_and_mul_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        gate, up, grad_output, grad_gate, grad_up, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void elementwise_add(
    const float* a,
    const float* b,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<grid_size, block_size, 0, stream>>>(
        a, b, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void transpose(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
) {
    const int total = rows * cols;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    transpose_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, rows, cols
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda::neural
