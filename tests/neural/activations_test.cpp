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
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

namespace {

    /// Runs a device activation over `input` and returns the host copy of the result.
    template <typename Launch>
    std::vector<float> run_activation(const std::vector<float>& input, Launch&& launch) {
        const int size = static_cast<int>(input.size());
        cuda::memory::Buffer<float> d_in(size), d_out(size);
        d_in.copy_from(input.data(), input.size());
        launch(d_in.data(), d_out.data(), size);
        std::vector<float> out(static_cast<size_t>(size));
        d_out.copy_to(out.data(), out.size());
        return out;
    }

}  // namespace

// These seven cases used to evaluate std::fmaxf, std::tanh, std::exp and a ternary against
// literals -- for example `EXPECT_EQ(std::fmaxf(0.0f, 5.0f), 5.0f)`. Not one of them called
// relu(), leaky_relu(), sigmoid() or tanh() from this library, so they tested the C++ standard
// library and could never fail because of a nova bug. Rewritten to drive the device
// implementations and compare against a defensible reference.

TEST_F(ActivationTest, ReLUZeroesNegativesAndPassesPositives) {
    const std::vector<float> input {-5.0f, -0.5f, -0.0f, 0.0f, 0.5f, 5.0f, 100.0f};
    const auto out = run_activation(input, [](const float* in, float* dst, int n) { relu(in, dst, n); });
    ASSERT_EQ(out.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_EQ(out[i], std::max(0.0f, input[i])) << "relu at index " << i << " (input " << input[i] << ")";
    }
}

TEST_F(ActivationTest, LeakyReLUScalesNegativesByAlpha) {
    const std::vector<float> input {-100.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 100.0f};
    for (const float alpha : {0.01f, 0.1f}) {
        const auto out = run_activation(input, [alpha](const float* in, float* dst, int n) { leaky_relu(in, dst, n, alpha); });
        ASSERT_EQ(out.size(), input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            const float expected = input[i] > 0.0f ? input[i] : alpha * input[i];
            EXPECT_NEAR(out[i], expected, 1e-5f) << "alpha=" << alpha << " at index " << i;
        }
    }
}

// A wrong alpha would still produce the right sign, so pin the magnitude explicitly.
TEST_F(ActivationTest, LeakyReLUHonoursACustomAlpha) {
    const std::vector<float> input {-5.0f};
    const auto out = run_activation(input, [](const float* in, float* dst, int n) { leaky_relu(in, dst, n, 0.1f); });
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0], -0.5f, 1e-6f) << "alpha=0.1 applied to -5 must give -0.5, not -0.05";
}

TEST_F(ActivationTest, SigmoidIsHalfAtZeroAndSaturatesAtTheBounds) {
    const std::vector<float> input {-10.0f, 0.0f, 10.0f};
    const auto out = run_activation(input, [](const float* in, float* dst, int n) { sigmoid(in, dst, n); });
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], 0.0f, 1e-4f) << "sigmoid(-10) saturates low";
    EXPECT_NEAR(out[1], 0.5f, 1e-6f) << "sigmoid(0) is exactly one half";
    EXPECT_NEAR(out[2], 1.0f, 1e-4f) << "sigmoid(10) saturates high";
    for (float v : out) {
        EXPECT_GT(v, 0.0f) << "sigmoid never reaches 0";
        EXPECT_LT(v, 1.0f) << "sigmoid never reaches 1";
    }
}

TEST_F(ActivationTest, TanhMatchesTheAnalyticValue) {
    const std::vector<float> input {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    const auto out = run_activation(input, [](const float* in, float* dst, int n) { tanh_activation(in, dst, n); });
    ASSERT_EQ(out.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(out[i], std::tanh(static_cast<double>(input[i])), 1e-5f) << "tanh at index " << i;
    }
}

TEST_F(ActivationTest, ActivationOptionsDefaults) {
    ActivationOptions options;
    EXPECT_EQ(options.alpha, 0.01f);
    EXPECT_EQ(options.negative_slope, 0.01f);
}

// Milestone v2.21: silu_and_mul is the gated-activation used by the
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

// v2.23: silu_and_mul_backward must match the analytic derivatives
// (dout = du .* up .* silu'(g), dup = dout .* silu(g)).
TEST_F(ActivationTest, SiluAndMulBackwardMatchesFormula) {
    const int size = 1024;
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> gate(size), up(size), dout(size);
    for (int i = 0; i < size; ++i) {
        gate[i] = dist(rng);
        up[i] = dist(rng);
        dout[i] = dist(rng);
    }
    auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    auto silu_prime = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 + x * (1.0 - s));
    };
    std::vector<float> exp_dg(size), exp_du(size);
    for (int i = 0; i < size; ++i) {
        exp_dg[i] = static_cast<float>(dout[i] * up[i] * silu_prime(gate[i]));
        exp_du[i] = static_cast<float>(dout[i] * silu(gate[i]));
    }

    cuda::memory::Buffer<float> d_gate(size), d_up(size), d_dout(size), d_dg(size), d_du(size);
    d_gate.copy_from(gate.data(), size);
    d_up.copy_from(up.data(), size);
    d_dout.copy_from(dout.data(), size);
    cuda::neural::silu_and_mul_backward(d_gate.data(), d_up.data(), d_dout.data(), d_dg.data(), d_du.data(), size);
    d_dg.copy_to(gate.data(), size);
    d_du.copy_to(up.data(), size);

    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(gate[i], exp_dg[i], 1e-4f) << "grad_gate element " << i;
        EXPECT_NEAR(up[i], exp_du[i], 1e-4f) << "grad_up element " << i;
    }
}

// v2.23: elementwise_add and transpose must match their host references.
TEST_F(ActivationTest, ElementwiseAddMatchesReference) {
    const int size = 512;
    std::vector<float> a(size), b(size), ref(size);
    for (int i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 1.5f * static_cast<float>(i);
        ref[i] = a[i] + b[i];
    }
    cuda::memory::Buffer<float> d_a(size), d_b(size), d_out(size);
    d_a.copy_from(a.data(), size);
    d_b.copy_from(b.data(), size);
    cuda::neural::elementwise_add(d_a.data(), d_b.data(), d_out.data(), size);
    d_out.copy_to(a.data(), size);
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(a[i], ref[i]);
    }
}

TEST_F(ActivationTest, TransposeMatchesReference) {
    const int rows = 6, cols = 9;
    std::vector<float> in(static_cast<size_t>(rows) * cols), ref(static_cast<size_t>(cols) * rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in[r * cols + c] = static_cast<float>(r * 100 + c);
            ref[c * rows + r] = in[r * cols + c];
        }
    }
    cuda::memory::Buffer<float> d_in(rows * cols), d_out(cols * rows);
    d_in.copy_from(in.data(), in.size());
    cuda::neural::transpose(d_in.data(), d_out.data(), rows, cols);
    d_out.copy_to(in.data(), in.size());
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            EXPECT_FLOAT_EQ(in[c * rows + r], ref[c * rows + r]);
        }
    }
}
