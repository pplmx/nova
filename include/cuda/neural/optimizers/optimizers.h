#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace cuda::neural::optimizers {

struct OptimizerConfig {
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.01f;
    bool fused = true;
};

class AdamWOptimizer {
public:
    explicit AdamWOptimizer(const OptimizerConfig& config);
    ~AdamWOptimizer();

    void step(
        float* params,
        const float* grads,
        size_t num_elements,
        int step,
        cudaStream_t stream = nullptr
    );

    void set_learning_rate(float lr);
    void set_weight_decay(float wd);
    float get_learning_rate() const { return config_.learning_rate; }
    float get_weight_decay() const { return config_.weight_decay; }

    void zero_momentum();
    void zero_grad();

    /**
     * @brief Moment-state export/import for checkpointing (v2.32, TASK-056)
     *
     * momentum_capacity() is the m/v buffer size in elements — 0 for a fresh
     * optimizer that has never stepped. copy_moments_to D2Hs the buffers into
     * host m/v with n == momentum_capacity() (anything else throws);
     * copy_moments_from H2Ds n elements in, growing (allocating) when n
     * exceeds the capacity but rejecting n < capacity — a partial overwrite
     * would leave a stale tail that the next step() reads over — and
     * resetting the moments to zero for n == 0. A restored optimizer steps
     * exactly like the one it came from (the AdamW update reads m/v, so
     * resume needs them byte-identical).
     */
    [[nodiscard]] size_t momentum_capacity() const;
    void copy_moments_to(float* m, float* v, size_t n,
                         cudaStream_t stream = nullptr) const;
    void copy_moments_from(const float* m, const float* v, size_t n,
                           cudaStream_t stream = nullptr);

private:
    OptimizerConfig config_;
    // Device moment buffers (v2.27): the update runs in a fused kernel instead
    // of the former D2H -> host vector loop -> H2D.
    std::unique_ptr<cuda::memory::Buffer<float>> m_data_;
    std::unique_ptr<cuda::memory::Buffer<float>> v_data_;
    bool initialized_ = false;
};

struct LAMBConfig {
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-6f;
    float weight_decay = 0.01f;
    float clamp_value = 10.0f;
    bool use_layer_adaptation = true;
};

class LAMBOptimizer {
public:
    explicit LAMBOptimizer(const LAMBConfig& config);
    ~LAMBOptimizer();

    void step(
        float* params,
        const float* grads,
        size_t num_elements,
        int step,
        float* layer_norm_1 = nullptr,
        float* layer_norm_2 = nullptr,
        cudaStream_t stream = nullptr
    );

    void set_learning_rate(float lr);
    float get_learning_rate() const { return config_.learning_rate; }

    void zero_momentum();
    void zero_grad();

private:
    LAMBConfig config_;
    // Device moment buffers (v2.31 / DEC-017): the update runs in a fused
    // kernel instead of the former D2H -> host vector loop -> H2D (the same
    // change v2.27 made to AdamW).
    std::unique_ptr<cuda::memory::Buffer<float>> m_data_;
    std::unique_ptr<cuda::memory::Buffer<float>> v_data_;
    bool initialized_ = false;
};

struct GradientClipConfig {
    float max_norm = 1.0f;
    enum class NormType { L2, Inf };
    NormType norm_type = NormType::L2;
};

float clip_gradients(
    float* grads,
    size_t num_elements,
    const GradientClipConfig& config,
    cudaStream_t stream = nullptr
);

float compute_gradient_norm(
    const float* grads,
    size_t num_elements,
    GradientClipConfig::NormType norm_type = GradientClipConfig::NormType::L2,
    cudaStream_t stream = nullptr
);

class GradientClipper {
public:
    explicit GradientClipper(const GradientClipConfig& config);

    float clip(float* grads, size_t num_elements, cudaStream_t stream = nullptr);
    float compute_norm(const float* grads, size_t num_elements, cudaStream_t stream = nullptr);

    void set_max_norm(float max_norm);
    float get_max_norm() const { return config_.max_norm; }

private:
    GradientClipConfig config_;
};

namespace detail {

// Fused AdamW update kernel (v2.27): the per-element bias-corrected update the
// host loop previously did, now run entirely on device.
void adamw_step_device(
    float* params, const float* grads, float* m, float* v, size_t n,
    float lr_t, float beta1, float beta2, float eps, float wd,
    float beta1_pow, float beta2_pow, cudaStream_t stream);

// Device gradient norm (L2 or Inf) and clipping (v2.27).
float gradient_norm_device(const float* grads, size_t n,
                           GradientClipConfig::NormType norm_type,
                           cudaStream_t stream);
void clip_device(float* grads, size_t n, float scale, cudaStream_t stream);

// Fused LAMB update kernel (milestone v2.31 / DEC-017): per-element
// bias-corrected moments + trust-ratio update. `rtw` is the host-computed
// layer-adaptation ratio (phi_1/phi_2, 1.0 when disabled) — a scalar kernel
// arg, per-element `r = clamp(param/update, 1/clamp_val, clamp_val)`.
void lamb_step_device(
    float* params, const float* grads, float* m, float* v, size_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float beta1_pow, float beta2_pow, float rtw, float clamp_val,
    bool use_layer_adaptation, cudaStream_t stream);

}  // namespace detail

}  // namespace cuda::neural::optimizers
