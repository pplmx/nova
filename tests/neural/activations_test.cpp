#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/activations.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::neural;

class ActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(ActivationTest, ReLUPositiveValues) {
    EXPECT_EQ(std::fmaxf(0.0f, 5.0f), 5.0f);
    EXPECT_EQ(std::fmaxf(0.0f, 100.0f), 100.0f);
}

TEST_F(ActivationTest, ReLUNegativeValues) {
    EXPECT_EQ(std::fmaxf(0.0f, -5.0f), 0.0f);
    EXPECT_EQ(std::fmaxf(0.0f, -100.0f), 0.0f);
}

TEST_F(ActivationTest, LeakyReLUPositiveValues) {
    float alpha = 0.01f;
    float x = 5.0f;
    float expected = x > 0.0f ? x : alpha * x;
    EXPECT_EQ(expected, 5.0f);
}

TEST_F(ActivationTest, LeakyReLUNegativeValues) {
    float alpha = 0.01f;
    float x = -5.0f;
    float expected = x > 0.0f ? x : alpha * x;
    EXPECT_NEAR(expected, -0.05f, 0.001f);
}

TEST_F(ActivationTest, LeakyReLUCustomAlpha) {
    float alpha = 0.1f;
    float x = -5.0f;
    float expected = x > 0.0f ? x : alpha * x;
    EXPECT_NEAR(expected, -0.5f, 0.001f);
}

TEST_F(ActivationTest, SigmoidPositiveValues) {
    float x = 0.0f;
    float expected = 1.0f / (1.0f + std::exp(-x));
    EXPECT_NEAR(expected, 0.5f, 0.001f);
}

TEST_F(ActivationTest, SigmoidBounds) {
    float x_large_pos = 10.0f;
    float x_large_neg = -10.0f;

    float sigmoid_pos = 1.0f / (1.0f + std::exp(-x_large_pos));
    float sigmoid_neg = 1.0f / (1.0f + std::exp(-x_large_neg));

    EXPECT_GT(sigmoid_pos, 0.99f);
    EXPECT_LT(sigmoid_neg, 0.01f);
}

TEST_F(ActivationTest, TanhBounds) {
    float x = 0.0f;
    float expected = std::tanh(x);
    EXPECT_NEAR(expected, 0.0f, 0.001f);
}

TEST_F(ActivationTest, TanhAsymptotes) {
    float x_large_pos = 10.0f;
    float x_large_neg = -10.0f;

    EXPECT_GT(std::tanh(x_large_pos), 0.99f);
    EXPECT_LT(std::tanh(x_large_neg), -0.99f);
}

TEST_F(ActivationTest, ActivationOptionsDefaults) {
    ActivationOptions options;
    EXPECT_EQ(options.alpha, 0.01f);
    EXPECT_EQ(options.negative_slope, 0.01f);
}

// Milestone v2.21 (TASK-010): silu_and_mul is the gated-activation used by the
// weight-managed TensorParallelMLP (gate projection through SiLU, elementwise
// with up). Pin it against the host formula out = silu(gate) * up.
TEST_F(ActivationTest, SiluAndMulMatchesHostFormula) {
    const int size = 2048;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> gate(size), up(size);
    for (int i = 0; i < size; ++i) {
        gate[i] = dist(rng);
        up[i] = dist(rng);
    }

    std::vector<float> expected(size);
    for (int i = 0; i < size; ++i) {
        const float g = gate[i];
        expected[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }

    cuda::memory::Buffer<float> d_gate(size), d_up(size), d_out(size);
    d_gate.copy_from(gate.data(), size);
    d_up.copy_from(up.data(), size);
    cuda::neural::silu_and_mul(d_gate.data(), d_up.data(), d_out.data(), size);
    d_out.copy_to(gate.data(), size);  // reuse gate as output scratch

    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(gate[i], expected[i], 1e-4f) << "element " << i;
    }
}

// Negative gate inputs must flip sign like x*sigmoid(x), not clamp.
TEST_F(ActivationTest, SiluAndMulNegativeGate) {
    const float gate[] = {-2.0f, 0.0f, 2.0f};
    const float up[] = {1.0f, 1.0f, 3.0f};
    cuda::memory::Buffer<float> d_gate(3), d_up(3), d_out(3);
    d_gate.copy_from(gate, 3);
    d_up.copy_from(up, 3);
    cuda::neural::silu_and_mul(d_gate.data(), d_up.data(), d_out.data(), 3);

    float out[3];
    d_out.copy_to(out, 3);
    const float g0 = -2.0f;
    EXPECT_NEAR(out[0], (g0 / (1.0f + std::exp(-g0))), 1e-5f);
    EXPECT_NEAR(out[1], 0.0f, 1e-6f);
    EXPECT_NEAR(out[2], 3.0f * (2.0f / (1.0f + std::exp(-2.0f))), 1e-5f);
}
