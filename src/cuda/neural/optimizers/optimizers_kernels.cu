/**
 * @file optimizers_kernels.cu
 * @brief Device kernels for the optimizer stack (milestone v2.27)
 *
 * The per-element AdamW update, the gradient L2/Inf norm reduction, and the
 * gradient clip scaling — the operations that used to round-trip the whole
 * gradient/params through the host every step. Compiled as CUDA; the host-side
 * API in optimizers.cpp calls into these via the detail:: functions.
 */

#include "cuda/neural/optimizers/optimizers.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace cuda::neural::optimizers {

namespace {

// Fused AdamW update: w' = w - lr_t*(m_hat/(sqrt(v_hat)+eps) + wd*w), with the
// bias-corrected m/v moments updated in place on device.
__global__ void adamw_step_kernel(
    float* params, const float* grads, float* m, float* v, size_t n,
    float lr_t, float beta1, float beta2, float eps, float wd,
    float beta1_pow, float beta2_pow) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float grad = grads[i];
    const float param = params[i];
    const float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
    const float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    const float m_hat = m_new / (1.0f - beta1_pow);
    const float v_hat = v_new / (1.0f - beta2_pow);
    const float update = m_hat / (sqrtf(v_hat) + eps) + wd * param;
    params[i] = param - lr_t * update;
    m[i] = m_new;
    v[i] = v_new;
}

__global__ void grad_scale_kernel(float* grads, size_t n, float scale) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    grads[i] *= scale;
}

// Tree-reduce sum of squares (for L2 norm) into partials[gridDim.x].
__global__ void l2_partial_kernel(const float* grads, size_t n, float* partials,
                                  size_t partials_count) {
    const size_t span =
        (n + partials_count - 1) / partials_count;  // per-block chunk
    const size_t begin = blockIdx.x * span;
    const size_t end = (begin + span < n) ? begin + span : n;
    float acc = 0.0f;
    for (size_t i = begin; i < end; ++i) {
        acc += grads[i] * grads[i];
    }
    partials[blockIdx.x] = acc;
}

__global__ void inf_partial_kernel(const float* grads, size_t n, float* partials,
                                   size_t partials_count) {
    const size_t span =
        (n + partials_count - 1) / partials_count;  // per-block chunk
    const size_t begin = blockIdx.x * span;
    const size_t end = (begin + span < n) ? begin + span : n;
    float mx = 0.0f;
    for (size_t i = begin; i < end; ++i) {
        mx = fmaxf(mx, fabsf(grads[i]));
    }
    partials[blockIdx.x] = mx;
}

}  // namespace

namespace detail {

void adamw_step_device(float* params, const float* grads, float* m, float* v,
                       size_t n, float lr_t, float beta1, float beta2, float eps,
                       float wd, float beta1_pow, float beta2_pow,
                       cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    adamw_step_kernel<<<grid, block, 0, stream>>>(params, grads, m, v, n, lr_t,
                                                  beta1, beta2, eps, wd,
                                                  beta1_pow, beta2_pow);
    CUDA_CHECK(cudaGetLastError());
}

float gradient_norm_device(const float* grads, size_t n,
                           GradientClipConfig::NormType norm_type,
                           cudaStream_t stream) {
    if (n == 0) return 0.0f;
    const size_t partials_count = 512;
    auto partials = std::make_unique<float[]>(partials_count);
    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, partials_count * sizeof(float)));
    const int grid = static_cast<int>(partials_count);
    if (norm_type == GradientClipConfig::NormType::Inf) {
        inf_partial_kernel<<<grid, 1, 0, stream>>>(grads, n, d_partials,
                                                   partials_count);
    } else {
        l2_partial_kernel<<<grid, 1, 0, stream>>>(grads, n, d_partials,
                                                  partials_count);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(partials.get(), d_partials,
                          partials_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partials));
    float result = 0.0f;
    for (size_t i = 0; i < partials_count; ++i) {
        result = (norm_type == GradientClipConfig::NormType::Inf)
                     ? std::max(result, partials[i])
                     : result + partials[i];
    }
    return (norm_type == GradientClipConfig::NormType::Inf)
               ? result
               : std::sqrt(result);
}

void clip_device(float* grads, size_t n, float scale, cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    grad_scale_kernel<<<grid, block, 0, stream>>>(grads, n, scale);
    CUDA_CHECK(cudaGetLastError());
}

// Fused LAMB update (milestone v2.31 / DEC-017): the exact per-element math of
// the host LAMBOptimizer::step loop, run on device — bias-corrected m/v, the
// AdamW-style update, then the per-element trust ratio
// r = clamp(|w|/|update|, 1/clamp_val, clamp_val) when layer adaptation is
// enabled (rtw = the host-computed phi_1/phi_2 layer ratio, a scalar arg).
__global__ void lamb_step_kernel(
    float* params, const float* grads, float* m, float* v, size_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float beta1_pow, float beta2_pow, float rtw, float clamp_val,
    bool use_layer_adaptation) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float grad = grads[i];
    const float param = params[i];
    const float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
    const float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    const float m_hat = m_new / (1.0f - beta1_pow);
    const float v_hat = v_new / (1.0f - beta2_pow);
    float update = m_hat / (sqrtf(v_hat) + eps) + wd * param;
    float r = 1.0f;
    if (use_layer_adaptation) {
        const float nu = fabsf(update);
        const float np = fabsf(param);
        if (np > 0.0f) {
            r = np / nu;
            r = fminf(fmaxf(r, 1.0f / clamp_val), clamp_val);
        }
    }
    params[i] = param - lr * r * rtw * update;
    m[i] = m_new;
    v[i] = v_new;
}

void lamb_step_device(
    float* params, const float* grads, float* m, float* v, size_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float beta1_pow, float beta2_pow, float rtw, float clamp_val,
    bool use_layer_adaptation, cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    lamb_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, m, v, n, lr, beta1, beta2, eps, wd, beta1_pow,
        beta2_pow, rtw, clamp_val, use_layer_adaptation);
    // async launch errors surface on the next sync; check immediately so a
    // bad launch is reported here, not in a caller's later cudaStreamSync.
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace detail

}  // namespace cuda::neural::optimizers
