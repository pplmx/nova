/**
 * @file optimizers.cpp
 * @brief Optimizer implementations (milestone v2.27)
 *
 * The per-element compute now runs on device (see optimizers_kernels.cu): AdamW
 * m/v are device buffers updated by one fused kernel, the gradient norm uses a
 * device reduction, and clipping scales on device — removing the D2H/H2D full
 * round-trips the host loop previously did. LAMB keeps its host-side
 * implementation (its layer-adaptation needs host-side layer-norm reads and
 * per-element clamping) and is unaffected.
 */

#include "cuda/neural/optimizers/optimizers.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuda::neural::optimizers {

AdamWOptimizer::AdamWOptimizer(const OptimizerConfig& config)
    : config_(config), m_data_(), v_data_(), initialized_(false) {}

AdamWOptimizer::~AdamWOptimizer() = default;

void AdamWOptimizer::step(
    float* params,
    const float* grads,
    size_t num_elements,
    int step,
    cudaStream_t stream
) {
    if (!initialized_) {
        m_data_ = std::make_unique<cuda::memory::Buffer<float>>(num_elements);
        v_data_ = std::make_unique<cuda::memory::Buffer<float>>(num_elements);
        m_data_->fill(0.0f);
        v_data_->fill(0.0f);
        initialized_ = true;
    }

    float lr = config_.learning_rate;
    float beta1 = config_.beta1;
    float beta2 = config_.beta2;
    float eps = config_.epsilon;
    float wd = config_.weight_decay;

    float beta1_pow = std::pow(beta1, step);
    float beta2_pow = std::pow(beta2, step);

    float lr_t = lr * std::sqrt(1.0f - beta2_pow) / (1.0f - beta1_pow);

    // The update runs entirely on device (v2.27): no D2H/H2D round-trip.
    detail::adamw_step_device(params, grads, m_data_->data(), v_data_->data(),
                              num_elements, lr_t, beta1, beta2, eps, wd,
                              beta1_pow, beta2_pow, stream);
}

void AdamWOptimizer::set_learning_rate(float lr) {
    config_.learning_rate = lr;
}

void AdamWOptimizer::set_weight_decay(float wd) {
    config_.weight_decay = wd;
}

void AdamWOptimizer::zero_momentum() {
    if (m_data_) {
        m_data_->fill(0.0f);
        v_data_->fill(0.0f);
    }
}

void AdamWOptimizer::zero_grad() {}

LAMBOptimizer::LAMBOptimizer(const LAMBConfig& config)
    : config_(config), m_data_(), v_data_(), initialized_(false) {}

LAMBOptimizer::~LAMBOptimizer() {}

void LAMBOptimizer::step(
    float* params,
    const float* grads,
    size_t num_elements,
    int step,
    float* layer_norm_1,
    float* layer_norm_2,
    cudaStream_t stream
) {
    if (!initialized_) {
        m_data_.resize(num_elements, 0.0f);
        v_data_.resize(num_elements, 0.0f);
        initialized_ = true;
    }

    float lr = config_.learning_rate;
    float beta1 = config_.beta1;
    float beta2 = config_.beta2;
    float eps = config_.epsilon;
    float wd = config_.weight_decay;
    float clamp_val = config_.clamp_value;

    float beta1_pow = std::pow(beta1, step);
    float beta2_pow = std::pow(beta2, step);

    std::vector<float> h_grads(num_elements);
    std::vector<float> h_params(num_elements);

    CUDA_CHECK(cudaMemcpy(h_grads.data(), grads, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_params.data(), params, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    float rtw = 0.0f;
    if (config_.use_layer_adaptation && layer_norm_1 && layer_norm_2) {
        float phi_1 = *layer_norm_1;
        float phi_2 = *layer_norm_2;
        if (phi_1 > 0.0f && phi_2 > 0.0f) {
            rtw = phi_1 / phi_2;
        }
    }
    if (rtw == 0.0f) rtw = 1.0f;

    for (size_t i = 0; i < num_elements; ++i) {
        float grad = h_grads[i];
        float param = h_params[i];

        m_data_[i] = beta1 * m_data_[i] + (1.0f - beta1) * grad;
        v_data_[i] = beta2 * v_data_[i] + (1.0f - beta2) * grad * grad;

        float m_hat = m_data_[i] / (1.0f - beta1_pow);
        float v_hat = v_data_[i] / (1.0f - beta2_pow);

        float update = m_hat / (std::sqrt(v_hat) + eps);
        update += wd * param;

        float r = 1.0f;
        if (config_.use_layer_adaptation) {
            float norm_update = std::abs(update);
            float norm_param = std::abs(param);
            if (norm_param > 0.0f) {
                r = norm_param / norm_update;
                r = std::min(std::max(r, 1.0f / clamp_val), clamp_val);
            }
        }

        h_params[i] = param - lr * r * rtw * update;
    }

    CUDA_CHECK(cudaMemcpy(params, h_params.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
}

void LAMBOptimizer::set_learning_rate(float lr) {
    config_.learning_rate = lr;
}

void LAMBOptimizer::zero_momentum() {
    std::fill(m_data_.begin(), m_data_.end(), 0.0f);
    std::fill(v_data_.begin(), v_data_.end(), 0.0f);
}

void LAMBOptimizer::zero_grad() {}

float clip_gradients(
    float* grads,
    size_t num_elements,
    const GradientClipConfig& config,
    cudaStream_t stream
) {
    float norm = compute_gradient_norm(grads, num_elements, config.norm_type, stream);

    if (norm > config.max_norm) {
        float scale = config.max_norm / norm;
        detail::clip_device(grads, num_elements, scale, stream);
    }

    return norm;
}

float compute_gradient_norm(
    const float* grads,
    size_t num_elements,
    GradientClipConfig::NormType norm_type,
    cudaStream_t stream
) {
    // Device reduction (v2.27): no D2H of the full gradient.
    return detail::gradient_norm_device(grads, num_elements, norm_type, stream);
}

GradientClipper::GradientClipper(const GradientClipConfig& config)
    : config_(config) {}

float GradientClipper::clip(float* grads, size_t num_elements, cudaStream_t stream) {
    return clip_gradients(grads, num_elements, config_, stream);
}

float GradientClipper::compute_norm(const float* grads, size_t num_elements, cudaStream_t stream) {
    return compute_gradient_norm(grads, num_elements, config_.norm_type, stream);
}

void GradientClipper::set_max_norm(float max_norm) {
    config_.max_norm = max_norm;
}

}  // namespace cuda::neural::optimizers
