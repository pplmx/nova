/**
 * @file activations_edge_test.cpp
 * @brief Contracts for activation edge cases and their maths
 *
 * src/cuda/neural/activations.cu has nineteen functions, of which six were
 * referenced from any test. Two problems were found.
 *
 * - All ten host entry points computed a grid from the element count and launched
 *   unconditionally. A zero-element tensor therefore produced a zero-sized grid,
 *   which CUDA rejects:
 *
 *       relu(nullptr, nullptr, 0) -> "invalid configuration argument"
 *
 *   An empty tensor is a legal input in every tensor library, so all ten now
 *   return without launching.
 *
 * - transpose computed `const int total = rows * cols`, which overflows int for a
 *   large matrix, wrapping the element count negative and producing a nonsense
 *   grid. The product is now computed in a wider type and rejected when it cannot
 *   be expressed.
 *
 * The maths themselves were checked and are correct: silu(x)*up with
 * silu'(x) = sigmoid(x)*(1 + x*(1 - sigmoid(x))), leaky relu's slope, the
 * transpose index arithmetic out[c*rows + r] = in[r*cols + c], and elementwise
 * add.
 */

#include <cuda/neural/activations.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Owns a device float array.
class DeviceArray {
public:
    explicit DeviceArray(const std::vector<float>& host) : count_(host.size()) {
        cudaMalloc(&ptr_, count_ * sizeof(float));
        if (count_ > 0) {
            cudaMemcpy(ptr_, host.data(), count_ * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
    }

    explicit DeviceArray(size_t count) : count_(count) {
        cudaMalloc(&ptr_, count_ * sizeof(float));
        if (count_ > 0) {
            cudaMemset(ptr_, 0, count_ * sizeof(float));
        }
    }

    ~DeviceArray() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    float* data() { return ptr_; }
    const float* data() const { return ptr_; }

    std::vector<float> read() const {
        std::vector<float> host(count_, 0.0f);
        if (count_ > 0) {
            cudaMemcpy(host.data(), ptr_, count_ * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        return host;
    }

private:
    float* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace

// Regression: every one of these threw "invalid configuration argument".
TEST(ActivationEdgeTest, EmptyTensorsDoNotLaunchAGrid) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A zero-element activation must be a no-op, not an error.
    EXPECT_NO_THROW(relu(nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(relu_inplace(nullptr, 0, nullptr));
    EXPECT_NO_THROW(leaky_relu(nullptr, nullptr, 0, 0.01f, nullptr));
    EXPECT_NO_THROW(leaky_relu_inplace(nullptr, 0, 0.01f, nullptr));
    EXPECT_NO_THROW(sigmoid(nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(tanh_activation(nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(silu_and_mul(nullptr, nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(
        silu_and_mul_backward(nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(elementwise_add(nullptr, nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(transpose(nullptr, nullptr, 0, 4, nullptr));
    EXPECT_NO_THROW(transpose(nullptr, nullptr, 4, 0, nullptr));
    (void)cudaGetLastError();
}

TEST(ActivationEdgeTest, ReluClampsNegativesToZero) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> input = {-2.0f, -0.5f, 0.0f, 0.5f, 3.0f};
    DeviceArray d_in(input);
    DeviceArray d_out(input.size());

    relu(d_in.data(), d_out.data(), static_cast<int>(input.size()), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_out.read();
    ASSERT_EQ(output.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], std::max(0.0f, input[i])) << "element " << i;
    }
}

TEST(ActivationEdgeTest, LeakyReluUsesAlphaForNegatives) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const float alpha = 0.1f;
    const std::vector<float> input = {-10.0f, -1.0f, 2.0f};
    DeviceArray d_in(input);
    DeviceArray d_out(input.size());

    leaky_relu(d_in.data(), d_out.data(), static_cast<int>(input.size()), alpha,
               nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_out.read();
    ASSERT_EQ(output.size(), 3u);
    EXPECT_NEAR(output[0], -1.0f, 1e-5f) << "-10 * 0.1";
    EXPECT_NEAR(output[1], -0.1f, 1e-5f) << "-1 * 0.1";
    EXPECT_FLOAT_EQ(output[2], 2.0f) << "positives pass through unchanged";
}

TEST(ActivationEdgeTest, SigmoidIsBoundedAndCentred) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> input = {-20.0f, 0.0f, 20.0f};
    DeviceArray d_in(input);
    DeviceArray d_out(input.size());

    sigmoid(d_in.data(), d_out.data(), static_cast<int>(input.size()), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_out.read();
    ASSERT_EQ(output.size(), 3u);
    EXPECT_NEAR(output[1], 0.5f, 1e-6f) << "sigmoid(0) is one half";
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-6f) << "a very negative input approaches zero";
    // sigmoid(20) is 1 - 2e-9, which is exactly 1.0 in float — 1 + 2e-9 rounds
    // away. Demanding strict inequality here would be asserting a precision the
    // type does not have, so the bound is stated as float accuracy.
    EXPECT_LE(output[2], 1.0f) << "sigmoid never exceeds one";
    EXPECT_NEAR(output[2], 1.0f, 1e-6f) << "a very positive input saturates at one";
}

TEST(ActivationEdgeTest, SiluAndMulMatchesItsDefinition) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> gate = {-1.0f, 0.0f, 2.0f};
    const std::vector<float> up = {3.0f, 5.0f, -2.0f};
    DeviceArray d_gate(gate);
    DeviceArray d_up(up);
    DeviceArray d_out(gate.size());

    silu_and_mul(d_gate.data(), d_up.data(), d_out.data(),
                 static_cast<int>(gate.size()), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_out.read();
    ASSERT_EQ(output.size(), 3u);
    for (size_t i = 0; i < gate.size(); ++i) {
        const float expected =
            (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
        EXPECT_NEAR(output[i], expected, 1e-5f) << "element " << i;
    }
}

TEST(ActivationEdgeTest, SiluBackwardMatchesItsDefinition) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> gate = {-1.0f, 0.5f, 2.0f};
    const std::vector<float> up = {3.0f, 5.0f, -2.0f};
    const std::vector<float> grad = {1.0f, -1.0f, 0.5f};

    DeviceArray d_gate(gate);
    DeviceArray d_up(up);
    DeviceArray d_grad(grad);
    DeviceArray d_gate_grad(gate.size());
    DeviceArray d_up_grad(gate.size());

    silu_and_mul_backward(d_gate.data(), d_up.data(), d_grad.data(),
                          d_gate_grad.data(), d_up_grad.data(),
                          static_cast<int>(gate.size()), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto gate_grad = d_gate_grad.read();
    const auto up_grad = d_up_grad.read();
    for (size_t i = 0; i < gate.size(); ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-gate[i]));
        const float silu = gate[i] * sig;
        const float silu_prime = sig * (1.0f + gate[i] * (1.0f - sig));

        EXPECT_NEAR(gate_grad[i], grad[i] * up[i] * silu_prime, 1e-5f)
            << "grad_gate at " << i;
        EXPECT_NEAR(up_grad[i], grad[i] * silu, 1e-5f) << "grad_up at " << i;
    }
}

TEST(ActivationEdgeTest, TransposeMovesElementsToTheirMirroredIndex) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // 2x3 row-major input becomes 3x2.
    const std::vector<float> input = {1.0f, 2.0f, 3.0f,
                                      4.0f, 5.0f, 6.0f};
    DeviceArray d_in(input);
    DeviceArray d_out(input.size());

    transpose(d_in.data(), d_out.data(), 2, 3, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_out.read();
    ASSERT_EQ(output.size(), 6u);
    const std::vector<float> expected = {1.0f, 4.0f,
                                         2.0f, 5.0f,
                                         3.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], expected[i]) << "element " << i;
    }
}

// Regression: rows * cols was computed in int and could wrap negative.
TEST(ActivationEdgeTest, TransposeRejectsAnUnlaunchableElementCount) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // 50000^2 = 2.5e9, past INT_MAX; computing it in int wraps negative.
    const std::string message = [&] {
        try {
            transpose(nullptr, nullptr, 50000, 50000, nullptr);
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
        return std::string();
    }();

    EXPECT_FALSE(message.empty())
        << "an element count past INT_MAX cannot be launched and should be reported";
    EXPECT_NE(message.find("exceeds"), std::string::npos) << message;
    (void)cudaGetLastError();
}

TEST(ActivationEdgeTest, ElementwiseAddSumsTwoBuffers) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> a = {1.0f, -2.0f, 3.5f};
    const std::vector<float> b = {0.5f, 2.0f, -3.5f};
    DeviceArray d_a(a);
    DeviceArray d_b(b);
    DeviceArray d_out(a.size());

    elementwise_add(d_a.data(), d_b.data(), d_out.data(),
                    static_cast<int>(a.size()), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_out.read();
    ASSERT_EQ(output.size(), 3u);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], a[i] + b[i]) << "element " << i;
    }
}

TEST(ActivationEdgeTest, SingleElementAndPartialBlockSizesWork) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // One element exercises the smallest non-empty grid; 300 spans more than one
    // 256-thread block with a partial tail.
    DeviceArray single_in(std::vector<float>{-1.0f});
    DeviceArray single_out(1);
    relu(single_in.data(), single_out.data(), 1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_FLOAT_EQ(single_out.read()[0], 0.0f);

    std::vector<float> many(300, 2.0f);
    DeviceArray d_many(many);
    DeviceArray d_many_out(many.size());
    sigmoid(d_many.data(), d_many_out.data(), static_cast<int>(many.size()),
            nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto output = d_many_out.read();
    ASSERT_EQ(output.size(), 300u);
    const float expected = 1.0f / (1.0f + std::exp(-2.0f));
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected, 1e-6f) << "element " << i
                                                << " (tail of the grid)";
    }
}

}  // namespace cuda::neural::test
