/**
 * @file optimizers.cpp
 * @brief Optimizer implementations (milestone v2.27)
 *
 * The per-element compute now runs on device (see optimizers_kernels.cu): AdamW
 * and LAMB m/v are device buffers updated by fused kernels, the gradient norm
 * uses a device reduction, and clipping scales on device — removing the D2H/H2D
 * full round-trips the host loops previously did. LAMB's layer-adaptation ratio
 * (phi_1/phi_2) is a host-computed scalar passed to the fused kernel
 * (v2.31 / DEC-017), so the per-element update runs on device like AdamW's.
 */

#include "cuda/neural/optimizers/optimizers.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
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

size_t AdamWOptimizer::momentum_capacity() const {
    return m_data_ ? m_data_->size() : 0;
}

void AdamWOptimizer::copy_moments_to(float* m, float* v, size_t n,
                                     cudaStream_t stream) const {
    (void)stream;
    if (n > momentum_capacity()) {
        throw std::invalid_argument(
            "AdamWOptimizer::copy_moments_to: n exceeds the moment capacity");
    }
    if (n == 0) return;
    m_data_->copy_to(m, n);
    v_data_->copy_to(v, n);
}

void AdamWOptimizer::copy_moments_from(const float* m, const float* v,
                                       size_t n, cudaStream_t stream) {
    (void)stream;
    if (n == 0) {
        // A zero-size restore means "the saved optimizer was fresh": reset
        // whatever moments this optimizer already holds so a restored fresh
        // checkpoint leaves it cold, matching the original.
        zero_momentum();
        return;
    }
    if (!initialized_ || momentum_capacity() < n) {
        m_data_ = std::make_unique<cuda::memory::Buffer<float>>(n);
        v_data_ = std::make_unique<cuda::memory::Buffer<float>>(n);
        m_data_->fill(0.0f);
        v_data_->fill(0.0f);
        initialized_ = true;
    }
    m_data_->copy_from(m, n);
    v_data_->copy_from(v, n);
}

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
        m_data_ = std::make_unique<cuda::memory::Buffer<float>>(num_elements);
        v_data_ = std::make_unique<cuda::memory::Buffer<float>>(num_elements);
        m_data_->fill(0.0f);
        v_data_->fill(0.0f);
        initialized_ = true;
    }
    if (m_data_->size() < num_elements) {
        m_data_ = std::make_unique<cuda::memory::Buffer<float>>(num_elements);
        v_data_ = std::make_unique<cuda::memory::Buffer<float>>(num_elements);
        m_data_->fill(0.0f);
        v_data_->fill(0.0f);
    }

    // Host-computed layer-adaptation ratio (phi_1/phi_2): a scalar, passed to
    // the device kernel as an argument (v2.31 — the math itself runs on device).
    float rtw = 0.0f;
    if (config_.use_layer_adaptation && layer_norm_1 && layer_norm_2) {
        const float phi_1 = *layer_norm_1;
        const float phi_2 = *layer_norm_2;
        if (phi_1 > 0.0f && phi_2 > 0.0f) {
            rtw = phi_1 / phi_2;
        }
    }
    if (rtw == 0.0f) rtw = 1.0f;

    const float beta1_pow = std::pow(config_.beta1, step);
    const float beta2_pow = std::pow(config_.beta2, step);

    detail::lamb_step_device(
        params, grads, m_data_->data(), v_data_->data(), num_elements,
        config_.learning_rate, config_.beta1, config_.beta2, config_.epsilon,
        config_.weight_decay, beta1_pow, beta2_pow, rtw, config_.clamp_value,
        config_.use_layer_adaptation, stream);
}

void LAMBOptimizer::set_learning_rate(float lr) {
    config_.learning_rate = lr;
}

void LAMBOptimizer::zero_momentum() {
    if (m_data_) {
        m_data_->fill(0.0f);
        v_data_->fill(0.0f);
    }
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
