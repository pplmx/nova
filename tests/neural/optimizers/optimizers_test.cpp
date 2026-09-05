#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/optimizers/optimizers.h>
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace cuda::neural::optimizers::test {

class OptimizersTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
    }

    int device_ = 0;
};

TEST_F(OptimizersTest, AdamWOptimizerConstruction) {
    OptimizerConfig config;
    config.learning_rate = 0.001f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.weight_decay = 0.01f;

    AdamWOptimizer optimizer(config);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.001f);
    EXPECT_EQ(optimizer.get_weight_decay(), 0.01f);
}

TEST_F(OptimizersTest, AdamWOptimizerSetters) {
    OptimizerConfig config;
    AdamWOptimizer optimizer(config);

    optimizer.set_learning_rate(0.01f);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.01f);

    optimizer.set_weight_decay(0.001f);
    EXPECT_EQ(optimizer.get_weight_decay(), 0.001f);
}

TEST_F(OptimizersTest, AdamWZeroMomentumZerosMoments) {
    // The old test was construct-only (called zero_momentum on a fresh, never-
    // stepped optimizer and asserted nothing). Drive one real step first so the
    // moments are non-zero, then zero_momentum must zero both m and v.
    const size_t n = 64;
    std::vector<float> w(n, 1.0f), g(n, 0.01f);
    OptimizerConfig config;
    AdamWOptimizer optimizer(config);
    cuda::memory::Buffer<float> d_w(n), d_g(n);
    d_w.copy_from(w.data(), n);
    d_g.copy_from(g.data(), n);
    optimizer.step(d_w.data(), d_g.data(), n, 1);
    ASSERT_GT(optimizer.momentum_capacity(), 0u);

    optimizer.zero_momentum();

    std::vector<float> m(n, -1.0f), v(n, -1.0f);
    optimizer.copy_moments_to(m.data(), v.data(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(m[i], 0.0f) << "m must be zeroed at " << i;
        EXPECT_EQ(v[i], 0.0f) << "v must be zeroed at " << i;
    }
}

TEST_F(OptimizersTest, AdamWStepGrowsMomentCapacityZeroed) {
    // Regression (RIL TASK-079, ISS-017): AdamW only allocated m/v on the FIRST
    // step, so a later LARGER num_elements ran the fused kernel over the stale
    // smaller buffers — an out-of-bounds device read/write — and the grown
    // region resumed from uninitialized memory. The moment pair must now grow
    // with num_elements and the grown region must start cold (zero), so the
    // post-grow update equals the host single-step formula from zero moments.
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    AdamWOptimizer opt(cfg);

    const size_t n1 = 32, n2 = 256;
    std::vector<float> w2(n2, 1.0f), g2(n2, 0.01f);
    cuda::memory::Buffer<float> d_w2(n2), d_g2(n2);
    d_w2.copy_from(w2.data(), n2);
    d_g2.copy_from(g2.data(), n2);

    // Prime at the small size first.
    cuda::memory::Buffer<float> d_w1(n1), d_g1(n1);
    std::vector<float> w1(n1, 1.0f), g1(n1, 0.01f);
    d_w1.copy_from(w1.data(), n1);
    d_g1.copy_from(g1.data(), n1);
    opt.step(d_w1.data(), d_g1.data(), n1, 1);
    EXPECT_EQ(opt.momentum_capacity(), n1);

    // Grow: must not OOB, capacity must track the new size, and the moments
    // across the WHOLE grown range must equal the zero-cold-start formula
    // (grow semantics match LAMB — resize discards history, resumes cold):
    // m = (1-b1)*g = 1e-3, v = (1-b2)*g^2 = 1e-7 (g == 0.01).
    EXPECT_NO_THROW(opt.step(d_w2.data(), d_g2.data(), n2, 2));
    EXPECT_EQ(opt.momentum_capacity(), n2);

    std::vector<float> m(n2, -1.0f), v(n2, -1.0f);
    opt.copy_moments_to(m.data(), v.data(), n2);
    for (size_t i = 0; i < n2; ++i) {
        EXPECT_NEAR(m[i], 1e-3f, 1e-5f) << "m cold start (whole range) at " << i;
        EXPECT_NEAR(v[i], 1e-7f, 1e-9f) << "v cold start (whole range) at " << i;
    }
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(OptimizersTest, LAMBOptimizerConstruction) {
    LAMBConfig config;
    config.learning_rate = 0.001f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.use_layer_adaptation = true;

    LAMBOptimizer optimizer(config);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.001f);
}

TEST_F(OptimizersTest, LAMBOptimizerSetters) {
    LAMBConfig config;
    LAMBOptimizer optimizer(config);

    optimizer.set_learning_rate(0.005f);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.005f);
}

TEST_F(OptimizersTest, LAMBConfigDefaults) {
    LAMBConfig config;
    EXPECT_EQ(config.beta1, 0.9f);
    EXPECT_EQ(config.beta2, 0.999f);
    EXPECT_EQ(config.epsilon, 1e-6f);
    EXPECT_TRUE(config.use_layer_adaptation);
}

TEST_F(OptimizersTest, GradientClipperConstruction) {
    GradientClipConfig config;
    config.max_norm = 1.0f;
    config.norm_type = GradientClipConfig::NormType::L2;

    GradientClipper clipper(config);
    EXPECT_EQ(clipper.get_max_norm(), 1.0f);
}

TEST_F(OptimizersTest, GradientClipperSetters) {
    GradientClipConfig config;
    GradientClipper clipper(config);

    clipper.set_max_norm(5.0f);
    EXPECT_EQ(clipper.get_max_norm(), 5.0f);
}

TEST_F(OptimizersTest, GradientClipConfigEnums) {
    GradientClipConfig config;
    config.norm_type = GradientClipConfig::NormType::L2;
    EXPECT_EQ(config.norm_type, GradientClipConfig::NormType::L2);

    config.norm_type = GradientClipConfig::NormType::Inf;
    EXPECT_EQ(config.norm_type, GradientClipConfig::NormType::Inf);
}

TEST_F(OptimizersTest, OptimizerConfigDefaults) {
    OptimizerConfig config;
    EXPECT_EQ(config.learning_rate, 0.001f);
    EXPECT_EQ(config.beta1, 0.9f);
    EXPECT_EQ(config.beta2, 0.999f);
    EXPECT_EQ(config.epsilon, 1e-8f);
    EXPECT_EQ(config.weight_decay, 0.01f);
    EXPECT_TRUE(config.fused);
}

TEST_F(OptimizersTest, LAMBOptimizerLayerAdaptation) {
    LAMBConfig config;
    config.use_layer_adaptation = true;
    EXPECT_TRUE(config.use_layer_adaptation);

    LAMBOptimizer optimizer(config);
}

TEST_F(OptimizersTest, GradientClipperL2Config) {
    GradientClipConfig config;
    config.max_norm = 2.0f;
    config.norm_type = GradientClipConfig::NormType::L2;

    GradientClipper clipper(config);
    EXPECT_EQ(clipper.get_max_norm(), 2.0f);
}

}  // namespace cuda::neural::optimizers::test

namespace cuda::neural::optimizers::test {

// ============================================================================
// v2.27 device-kernel parity tests (TASK-036 / DEC-013): the fused AdamW,
// gradient-norm, and clip kernels must produce the exact same results as the
// host formulas they replaced (the 9-blocking-D2H/H2D per step that the v2.25
// cpp-review flagged).
// ============================================================================

namespace {

// Host AdamW formula (identical to the former optimizers.cpp host loop).
void host_adamw(const float* w, const float* g, float* w_out,
                std::vector<float>& m, std::vector<float>& v, size_t n,
                int step, float lr, float b1, float b2, float eps, float wd) {
    const float b1p = std::pow(b1, step);
    const float b2p = std::pow(b2, step);
    const float lr_t = lr * std::sqrt(1.0f - b2p) / (1.0f - b1p);
    for (size_t i = 0; i < n; ++i) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        const float mh = m[i] / (1.0f - b1p);
        const float vh = v[i] / (1.0f - b2p);
        const float up = mh / (std::sqrt(vh) + eps) + wd * w[i];
        w_out[i] = w[i] - lr_t * up;
    }
}

// Reference for the Milestone v2.31 device LAMB kernel: the exact
// LAMBOptimizer::step host math (the oracle the device must reproduce
// element-for-element). rtw is passed in (phi_1/phi_2 when layer adaptation
// is on, 1.0 otherwise); per-element trust ratio r = |w|/|update| clamped to
// [1/clamp_val, clamp_val] only when use_layer_adaptation.
void host_lamb(const float* w, const float* g, float* w_out,
               std::vector<float>& m, std::vector<float>& v, size_t n,
               int step, float lr, float b1, float b2, float eps, float wd,
               float rtw, float clamp_val, bool use_layer_adaptation) {
    const float b1p = std::pow(b1, step);
    const float b2p = std::pow(b2, step);
    for (size_t i = 0; i < n; ++i) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        const float mh = m[i] / (1.0f - b1p);
        const float vh = v[i] / (1.0f - b2p);
        float up = mh / (std::sqrt(vh) + eps) + wd * w[i];
        float r = 1.0f;
        if (use_layer_adaptation) {
            const float nu = std::abs(up);
            const float np = std::abs(w[i]);
            if (np > 0.0f) {
                r = np / nu;
                r = std::min(std::max(r, 1.0f / clamp_val), clamp_val);
            }
        }
        w_out[i] = w[i] - lr * r * rtw * up;
    }
}

}  // namespace

TEST_F(OptimizersTest, LAMBZeroMomentumResetsMoments) {
    // LAMB has no moment read-back (AdamW does), so prove zero_momentum works
    // by observable behavior: prime moments with step 1, zero them, then step 2
    // must evolve exactly like a re-start from zero moments at step index 2
    // (the host_lamb oracle). If zero_momentum were a no-op, the step-1 history
    // would leak into the update and the params would drift.
    const size_t n = 64;
    std::mt19937 rng(20260825);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w(n), g(n);
    for (size_t i = 0; i < n; ++i) { w[i] = dist(rng); g[i] = dist(rng); }

    LAMBConfig cfg;
    LAMBOptimizer opt(cfg);
    cuda::memory::Buffer<float> d_w(n), d_g(n);
    d_w.copy_from(w.data(), n);
    d_g.copy_from(g.data(), n);

    opt.step(d_w.data(), d_g.data(), n, 1);  // prime non-zero moments

    std::vector<float> w1(n);
    d_w.copy_to(w1.data(), n);
    std::vector<float> mh(n, 0.0f), vh(n, 0.0f), w_ref(n);
    host_lamb(w1.data(), g.data(), w_ref.data(), mh, vh, n, 2,
              cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon,
              cfg.weight_decay, /*rtw=*/1.0f, cfg.clamp_value,
              cfg.use_layer_adaptation);

    opt.zero_momentum();
    opt.step(d_w.data(), d_g.data(), n, 2);

    std::vector<float> w_after(n);
    d_w.copy_to(w_after.data(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(w_ref[i], w_after[i], 1e-4f)
            << "zero_momentum restart drift at " << i;
    }
}

// The device AdamW kernel must leave params equal to the host formula over K
// steps (same config, same moments, monotonic same sequence).
TEST_F(OptimizersTest, AdamWDeviceMatchesHostFormula) {
    const size_t n = 256;
    std::vector<float> w0(n), g(n);
    std::mt19937 rng(20260830);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) { w0[i] = dist(rng); g[i] = dist(rng); }

    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;

    // Reference: host AdamW moments.
    std::vector<float> hw = w0;
    std::vector<float> hm(n, 0.0f), hv(n, 0.0f);
    // Device: AdamWOptimizer with its internal device moments.
    AdamWOptimizer opt(cfg);
    cuda::memory::Buffer<float> d_w(n), d_g(n);
    d_w.copy_from(w0.data(), n);
    d_g.copy_from(g.data(), n);

    const int K = 5;
    for (int s = 1; s <= K; ++s) {
        host_adamw(hw.data(), g.data(), hw.data(), hm, hv, n, s,
                   cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon,
                   cfg.weight_decay);
        opt.step(d_w.data(), d_g.data(), n, s);
    }
    std::vector<float> dw(n);
    d_w.copy_to(dw.data(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(hw[i], dw[i], 1e-4f) << "AdamW device/host drift at " << i;
    }
}

// ============================================================================
// Milestone v2.31 (DEC-017 / TASK-052) — device LAMB optimizer kernel.
// detail::lamb_step_device must leave params element-for-element equal to the
// host LAMBOptimizer::step math (the oracle in host_lamb above) over K steps,
// for both layer-adaptation on/off and clamping on/off. Pinned against the
// throw-stub (RED); GREEN with the P2 fused kernel.
// ============================================================================

// Device LAMB over K steps with layer adaptation ON must match host_lamb
// element-for-element. Runs detail::lamb_step_device with persistent m/v
// device buffers (the P2 shape: LAMBOptimizer's m/v become device Buffers).
TEST_F(OptimizersTest, LAMBDeviceMatchesHostFormulaWithLayerAdaptation) {
    const size_t n = 256;
    std::vector<float> w0(n), g(n);
    std::mt19937 rng(20260824);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < n; ++i) { w0[i] = dist(rng); g[i] = dist(rng); }

    LAMBConfig cfg;
    cfg.learning_rate = 0.005f;
    cfg.use_layer_adaptation = true;
    cfg.clamp_value = 10.0f;
    const float rtw = 1.2f;  // phi_1/phi_2 layer ratio (host scalars)

    std::vector<float> hw = w0;
    std::vector<float> hm(n, 0.0f), hv(n, 0.0f);

    cuda::memory::Buffer<float> d_w(n), d_g(n), d_m(n), d_v(n);
    d_w.copy_from(w0.data(), n);
    d_g.copy_from(g.data(), n);
    d_m.fill(0.0f);
    d_v.fill(0.0f);

    const int K = 5;
    for (int s = 1; s <= K; ++s) {
        host_lamb(hw.data(), g.data(), hw.data(), hm, hv, n, s,
                  cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon,
                  cfg.weight_decay, rtw, cfg.clamp_value,
                  cfg.use_layer_adaptation);
        EXPECT_NO_THROW(
            detail::lamb_step_device(
                d_w.data(), d_g.data(), d_m.data(), d_v.data(), n,
                cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon,
                cfg.weight_decay, std::pow(cfg.beta1, s),
                std::pow(cfg.beta2, s), rtw, cfg.clamp_value,
                cfg.use_layer_adaptation, nullptr));
    }
    std::vector<float> dw(n);
    d_w.copy_to(dw.data(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(hw[i], dw[i], 1e-4f) << "LAMB device/host drift at " << i;
    }
}

// Layer adaptation OFF: r stays 1.0, so the update is lr*rtw times the
// AdamW-style update. Still must equal host_lamb element-for-element.
TEST_F(OptimizersTest, LAMBDeviceMatchesHostFormulaNoLayerAdaptation) {
    const size_t n = 128;
    std::vector<float> w0(n), g(n);
    std::mt19937 rng(20260825);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) { w0[i] = dist(rng); g[i] = dist(rng); }

    LAMBConfig cfg;
    cfg.learning_rate = 0.01f;
    cfg.use_layer_adaptation = false;
    const float rtw = 1.0f;

    std::vector<float> hw = w0;
    std::vector<float> hm(n, 0.0f), hv(n, 0.0f);

    cuda::memory::Buffer<float> d_w(n), d_g(n), d_m(n), d_v(n);
    d_w.copy_from(w0.data(), n);
    d_g.copy_from(g.data(), n);
    d_m.fill(0.0f);
    d_v.fill(0.0f);

    const int K = 3;
    for (int s = 1; s <= K; ++s) {
        host_lamb(hw.data(), g.data(), hw.data(), hm, hv, n, s,
                  cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon,
                  cfg.weight_decay, rtw, cfg.clamp_value,
                  cfg.use_layer_adaptation);
        EXPECT_NO_THROW(
            detail::lamb_step_device(
                d_w.data(), d_g.data(), d_m.data(), d_v.data(), n,
                cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon,
                cfg.weight_decay, std::pow(cfg.beta1, s),
                std::pow(cfg.beta2, s), rtw, cfg.clamp_value,
                cfg.use_layer_adaptation, nullptr));
    }
    std::vector<float> dw(n);
    d_w.copy_to(dw.data(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(hw[i], dw[i], 1e-4f) << "LAMB device/host drift at " << i;
    }
}

// Device gradient L2 norm must equal the host sum-of-squares.
TEST_F(OptimizersTest, GradientNormDeviceMatchesHost) {
    const size_t n = 513;
    std::vector<float> g(n);
    std::mt19937 rng(20260831);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < n; ++i) g[i] = dist(rng);
    cuda::memory::Buffer<float> d_g(n);
    d_g.copy_from(g.data(), n);

    double sum_sq = 0.0;
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += (double)g[i] * g[i];
        mx = std::max(mx, std::abs(g[i]));
    }
    const float ref_l2 = std::sqrt((float)sum_sq);

    EXPECT_NEAR(ref_l2,
                compute_gradient_norm(d_g.data(), n,
                                      GradientClipConfig::NormType::L2),
                1e-3f);
    EXPECT_NEAR(mx,
                compute_gradient_norm(d_g.data(), n,
                                      GradientClipConfig::NormType::Inf),
                1e-4f);
}

// Device clipping must scale by max_norm/norm exactly as the host did.
TEST_F(OptimizersTest, GradientClipDeviceMatchesHost) {
    const size_t n = 128;
    std::vector<float> g(n);
    std::mt19937 rng(20260901);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < n; ++i) g[i] = dist(rng);

    // Trigger clipping: sum of squares is large.
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) sum_sq += (double)g[i] * g[i];
    const float norm = std::sqrt((float)sum_sq);
    GradientClipConfig ccfg;
    ccfg.max_norm = norm * 0.5f;  // force scale < 1

    // Device clip.
    cuda::memory::Buffer<float> d_g(n);
    d_g.copy_from(g.data(), n);
    float dev_norm = clip_gradients(d_g.data(), n, ccfg);
    std::vector<float> dev_out(n);
    d_g.copy_to(dev_out.data(), n);

    // Host reference.
    const float scale = ccfg.max_norm / dev_norm;
    std::vector<float> ref_out(n);
    for (size_t i = 0; i < n; ++i) ref_out[i] = g[i] * scale;

    EXPECT_NEAR(norm, dev_norm, 1e-3f);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(ref_out[i], dev_out[i], 1e-4f)
            << "clip mismatch at " << i;
    }
}

}  // namespace cuda::neural::optimizers::test
