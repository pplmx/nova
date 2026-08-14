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

TEST_F(OptimizersTest, AdamWOptimizerZeroMomentum) {
    OptimizerConfig config;
    AdamWOptimizer optimizer(config);
    optimizer.zero_momentum();
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

TEST_F(OptimizersTest, LAMBOptimizerZeroMomentum) {
    LAMBConfig config;
    LAMBOptimizer optimizer(config);
    optimizer.zero_momentum();
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

}  // namespace

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
