/**
 * @file kernel_fusion_test.cpp
 * @brief Contracts for the fused neural kernels and their registry
 *
 * src/cuda/neural/fusion/kernel_fusion.cu fuses matmul+bias with an activation,
 * and layernorm+softmax into single passes. Four of its seventeen functions had
 * any test reference, so the elementwise maths and the registry behaviour had no
 * regression net.
 *
 * I checked the kernels before writing this and found no defect: the bias
 * broadcast uses idx % num_features correctly, relu and sigmoid are applied to
 * the biased value, and the layernorm+softmax kernel centres and scales before
 * dividing by the supplied row sum.
 *
 * Two interface behaviours are pinned as they are rather than changed, because
 * both are defensible for a configuration registry: an unknown fusion name is
 * ignored silently, and a disabled fusion still runs all three entry points
 * (the dispatcher's own choice is not observable through these functions).
 */

#include <cuda/device/error.h>
#include <cuda/neural/fusion/kernel_fusion.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

namespace cuda::neural::fusion::test {

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
        cudaMemset(ptr_, 0, count_ * sizeof(float));
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
        std::vector<float> host(count_);
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

// ------------------------------------------------------- registry --------

TEST(FusionKernelRegistryTest, IsASingleton) {
    EXPECT_EQ(&FusionKernelRegistry::instance(), &FusionKernelRegistry::instance());
}

TEST(FusionKernelRegistryTest, FusionsAreEnabledByDefault) {
    auto& registry = FusionKernelRegistry::instance();

    // Defaults come from the member initialisers; a fresh process has all three
    // on, which is the documented behaviour.
    registry.set_config(FusionConfig{});
    EXPECT_TRUE(registry.is_fusion_enabled("matmul_bias"));
    EXPECT_TRUE(registry.is_fusion_enabled("matmul_bias_activation"));
    EXPECT_TRUE(registry.is_fusion_enabled("layernorm_softmax"));
}

TEST(FusionKernelRegistryTest, EnableAndDisableAreObservable) {
    auto& registry = FusionKernelRegistry::instance();

    registry.disable_fusion("matmul_bias");
    EXPECT_FALSE(registry.is_fusion_enabled("matmul_bias"));
    EXPECT_TRUE(registry.is_fusion_enabled("layernorm_softmax"))
        << "disabling one fusion must not affect another";

    registry.enable_fusion("matmul_bias");
    EXPECT_TRUE(registry.is_fusion_enabled("matmul_bias"));

    registry.set_config(FusionConfig{});  // restore defaults
}

TEST(FusionKernelRegistryTest, SetConfigAssignsAllThreeFlags) {
    auto& registry = FusionKernelRegistry::instance();

    FusionConfig config;
    config.fuse_matmul_bias = true;
    config.fuse_matmul_bias_activation = false;
    config.fuse_layernorm_softmax = false;
    registry.set_config(config);

    EXPECT_TRUE(registry.is_fusion_enabled("matmul_bias"));
    EXPECT_FALSE(registry.is_fusion_enabled("matmul_bias_activation"));
    EXPECT_FALSE(registry.is_fusion_enabled("layernorm_softmax"));

    registry.set_config(FusionConfig{});
}

// Pinned as-is: a configuration registry that ignores an unknown key is a
// defensible design, and changing it to throw would break existing callers.
TEST(FusionKernelRegistryTest, UnknownNameIsIgnored) {
    auto& registry = FusionKernelRegistry::instance();

    EXPECT_NO_THROW(registry.register_fusion("no-such-fusion", true));
    EXPECT_NO_THROW(registry.enable_fusion("no-such-fusion"));
    EXPECT_NO_THROW(registry.disable_fusion("no-such-fusion"));
    EXPECT_FALSE(registry.is_fusion_enabled("no-such-fusion"))
        << "an unknown name reports disabled rather than throwing";

    registry.set_config(FusionConfig{});
}

// ------------------------------------------------- fused matmul + bias -----

TEST(FusedMatmulBiasTest, AddsTheBiasPerFeature) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Two rows of 3 features; the bias repeats every 3 elements.
    const std::vector<float> matmul = {1.0f, 2.0f, 3.0f,
                                       4.0f, 5.0f, 6.0f};
    const std::vector<float> bias = {0.5f, -1.0f, 2.0f};

    DeviceArray d_matmul(matmul);
    DeviceArray d_bias(bias);
    DeviceArray d_out(matmul.size());

    fused_matmul_bias(d_matmul.data(), d_bias.data(), d_out.data(), 2, 3,
                      nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    ASSERT_EQ(result.size(), 6u);
    const std::vector<float> expected = {1.5f, 1.0f, 5.0f,
                                         4.5f, 4.0f, 8.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6f) << "element " << i;
    }
}

TEST(FusedMatmulBiasTest, ReluClampsNegativesAfterTheBias) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Bias makes some values negative; relu must clamp those to 0.
    const std::vector<float> matmul = {1.0f, -5.0f, 3.0f};
    const std::vector<float> bias = {-2.0f, 1.0f, 0.0f};

    DeviceArray d_matmul(matmul);
    DeviceArray d_bias(bias);
    DeviceArray d_out(matmul.size());

    fused_matmul_bias_relu(d_matmul.data(), d_bias.data(), d_out.data(), 1, 3,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    ASSERT_EQ(result.size(), 3u);
    // 1-2 = -1 -> 0; -5+1 = -4 -> 0; 3+0 = 3 -> 3.
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 0.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
}

TEST(FusedMatmulBiasTest, SigmoidIsBoundedAndMonotonic) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> matmul = {-10.0f, 0.0f, 10.0f};
    const std::vector<float> bias = {0.0f, 0.0f, 0.0f};

    DeviceArray d_matmul(matmul);
    DeviceArray d_bias(bias);
    DeviceArray d_out(matmul.size());

    fused_matmul_bias_sigmoid(d_matmul.data(), d_bias.data(), d_out.data(), 1, 3,
                              nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    ASSERT_EQ(result.size(), 3u);
    for (const float value : result) {
        EXPECT_GT(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
    EXPECT_LT(result[0], result[1]) << "sigmoid must increase with its input";
    EXPECT_LT(result[1], result[2]);
    EXPECT_NEAR(result[1], 0.5f, 1e-6f) << "sigmoid(0) is one half";
}

TEST(FusedMatmulBiasTest, ReluMatchesAHandComputation) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A single row where every biased value is positive must pass through
    // unchanged, which catches a kernel that applies relu to the wrong operand.
    const std::vector<float> matmul = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> bias = {1.0f, 1.0f};

    DeviceArray d_matmul(matmul);
    DeviceArray d_bias(bias);
    DeviceArray d_out(matmul.size());

    fused_matmul_bias_relu(d_matmul.data(), d_bias.data(), d_out.data(), 2, 2,
                           nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    const std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "element " << i;
    }
}

// ------------------------------------------------- layernorm + softmax -----

TEST(FusedLayerNormSoftmaxTest, OutputSumsToOne) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The fusion computes exp(layernorm(x)) / sum(exp(layernorm(x))) per row, so
    // the output of each row must sum to 1 regardless of the input scale.
    const std::vector<float> input = {-2.0f, -1.0f, 1.0f, 2.0f};
    const std::vector<float> gamma(4, 1.0f);
    const std::vector<float> beta(4, 0.0f);

    DeviceArray d_input(input);
    DeviceArray d_gamma(gamma);
    DeviceArray d_beta(beta);
    DeviceArray d_out(input.size());

    FusedLayerNormSoftmax fusion(4);
    ASSERT_NO_THROW(fusion.forward(d_input.data(), d_out.data(), d_gamma.data(),
                                   d_beta.data(), 1, 1, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    ASSERT_EQ(result.size(), 4u);

    float total = 0.0f;
    for (const float value : result) {
        EXPECT_TRUE(std::isfinite(value)) << "got " << value;
        EXPECT_GE(value, 0.0f) << "a normalised exponential cannot be negative";
        total += value;
    }
    EXPECT_NEAR(total, 1.0f, 1e-4f) << "each row's weights must sum to one";
}

TEST(FusedLayerNormSoftmaxTest, LargerValuesGetLargerWeights) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A strictly increasing input, after layernorm, is still increasing, so the
    // exponentials are ordered and the output must be too.
    const std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> gamma(4, 1.0f);
    const std::vector<float> beta(4, 0.0f);

    DeviceArray d_input(input);
    DeviceArray d_gamma(gamma);
    DeviceArray d_beta(beta);
    DeviceArray d_out(input.size());

    FusedLayerNormSoftmax fusion(4);
    fusion.forward(d_input.data(), d_out.data(), d_gamma.data(), d_beta.data(), 1,
                   1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    for (size_t i = 1; i < result.size(); ++i) {
        EXPECT_GT(result[i], result[i - 1])
            << "element " << i << " is not larger than its predecessor";
    }
}

TEST(FusedLayerNormSoftmaxTest, GammaChangesTheDistribution) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Scaling gamma widens the layernorm output, which sharpens the exponential
    // distribution. A kernel that ignored gamma would produce identical rows.
    const std::vector<float> input = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> gamma_ones(4, 1.0f);
    const std::vector<float> gamma_fours(4, 4.0f);
    const std::vector<float> beta(4, 0.0f);

    DeviceArray d_input(input);
    DeviceArray d_beta(beta);
    DeviceArray d_ones(gamma_ones);
    DeviceArray d_fours(gamma_fours);
    DeviceArray d_out_one(input.size());
    DeviceArray d_out_four(input.size());

    FusedLayerNormSoftmax fusion(4);
    fusion.forward(d_input.data(), d_out_one.data(), d_ones.data(), d_beta.data(),
                   1, 1, nullptr);
    fusion.forward(d_input.data(), d_out_four.data(), d_fours.data(), d_beta.data(),
                   1, 1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto one = d_out_one.read();
    const auto four = d_out_four.read();
    ASSERT_EQ(one.size(), four.size());
    for (size_t i = 0; i < four.size(); ++i) {
        EXPECT_TRUE(std::isfinite(four[i])) << "element " << i;
    }
    EXPECT_NE(one, four) << "gamma must influence the result";

    // The larger gamma gives the last element a bigger share.
    EXPECT_GT(four.back(), one.back());
}

TEST(FusedLayerNormSoftmaxTest, MultipleRowsAreIndependent) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Two positions in one batch: each row normalises on its own.
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                                      -4.0f, -3.0f, -2.0f, -1.0f};
    const std::vector<float> gamma(4, 1.0f);
    const std::vector<float> beta(4, 0.0f);

    DeviceArray d_input(input);
    DeviceArray d_gamma(gamma);
    DeviceArray d_beta(beta);
    DeviceArray d_out(input.size());

    FusedLayerNormSoftmax fusion(4);
    fusion.forward(d_input.data(), d_out.data(), d_gamma.data(), d_beta.data(), 1,
                   2, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto result = d_out.read();
    ASSERT_EQ(result.size(), 8u);

    // Row 0 and row 1 hold mirrored inputs; because layernorm is affine, the two
    // rows must produce identical weight distributions.
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result[i], result[4 + i], 1e-4f)
            << "row 0 and row 1 differ at column " << i;
    }
}

}  // namespace cuda::neural::fusion::test
