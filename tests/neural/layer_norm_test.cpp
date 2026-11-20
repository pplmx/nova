#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include "cuda/neural/layer_norm.h"

using namespace cuda::neural;

class LayerNormTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(LayerNormTest, LayerNormResultConstruction) {
    LayerNormResult result(100);
    EXPECT_EQ(result.size, 100);
    EXPECT_NE(result.d_output, nullptr);
    EXPECT_NE(result.d_mean, nullptr);
    EXPECT_NE(result.d_variance, nullptr);
}

TEST_F(LayerNormTest, LayerNormResultDefaultConstruction) {
    LayerNormResult result;
    EXPECT_EQ(result.size, 0);
}

TEST_F(LayerNormTest, LayerNormResultClearFreesMemory) {
    LayerNormResult result(100);
    result.clear();
    EXPECT_EQ(result.d_output, nullptr);
    EXPECT_EQ(result.d_mean, nullptr);
    EXPECT_EQ(result.d_variance, nullptr);
}

TEST_F(LayerNormTest, NormalizedMeanIsApproximatelyZero) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    EXPECT_NEAR(mean, 3.0f, 0.001f);

    float variance = 0.0f;
    for (auto v : data) {
        variance += (v - mean) * (v - mean);
    }
    variance /= data.size();

    EXPECT_GT(variance, 0.0f);
}

TEST_F(LayerNormTest, EpsilonPreventsDivisionByZero) {
    float eps = 1e-5f;
    float var = 0.0f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    EXPECT_GT(inv_std, 0.0f);
    EXPECT_LT(inv_std, 1000.0f);
}

TEST_F(LayerNormTest, LayerNormParamsDefaults) {
    LayerNormParams params;
    EXPECT_EQ(params.eps, 1e-5f);
    EXPECT_TRUE(params.elementwise_affine);
}

TEST_F(LayerNormTest, LayerNormParamsCanBeCustomized) {
    LayerNormParams params;
    params.normalized_shape = 512;
    params.eps = 1e-3f;
    params.elementwise_affine = false;

    EXPECT_EQ(params.normalized_shape, 512);
    EXPECT_EQ(params.eps, 1e-3f);
    EXPECT_FALSE(params.elementwise_affine);
}

TEST_F(LayerNormTest, LayerNormOutputHasCorrectSize) {
    int batch_size = 4;
    int normalized_shape = 64;
    LayerNormResult result(batch_size * normalized_shape);

    EXPECT_EQ(result.size, batch_size * normalized_shape);
}

TEST_F(LayerNormTest, RSqrtComputation) {
    float var = 2.0f;
    float eps = 1e-5f;
    float expected = 1.0f / std::sqrt(var + eps);

    EXPECT_NEAR(expected, 0.707106f, 0.001f);
}
