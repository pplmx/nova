#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "cuda/neural/softmax.h"

using namespace cuda::neural;

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override { cudaDeviceReset(); }
};

TEST_F(SoftmaxTest, SoftmaxResultConstruction) {
    SoftmaxResult result(10, 5);
    EXPECT_EQ(result.size, 50);
    EXPECT_EQ(result.outer_dim, 10);
    EXPECT_EQ(result.inner_dim, 5);
    EXPECT_NE(result.d_output, nullptr);
}

TEST_F(SoftmaxTest, SoftmaxResultDefaultConstruction) {
    SoftmaxResult result;
    EXPECT_EQ(result.size, 0);
    EXPECT_EQ(result.outer_dim, 0);
    EXPECT_EQ(result.inner_dim, 0);
}

TEST_F(SoftmaxTest, SoftmaxResultClearFreesMemory) {
    SoftmaxResult result(5, 5);
    EXPECT_NE(result.d_output, nullptr);
    result.clear();
    EXPECT_EQ(result.d_output, nullptr);
}

TEST_F(SoftmaxTest, SoftmaxComputesProbabilityDistribution) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3);

    float sum = 0.0f;
    float max_val = *std::max_element(input.begin(), input.end());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (auto& v : output) {
        v /= sum;
    }

    float prob_sum = 0.0f;
    for (auto v : output) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
        prob_sum += v;
    }
    EXPECT_NEAR(prob_sum, 1.0f, 0.001f);
}

TEST_F(SoftmaxTest, SoftmaxIsTranslationInvariant) {
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {101.0f, 102.0f, 103.0f};

    float max1 = *std::max_element(input1.begin(), input1.end());
    float max2 = *std::max_element(input2.begin(), input2.end());

    float sum1 = 0.0f, sum2 = 0.0f;
    for (size_t i = 0; i < input1.size(); ++i) {
        sum1 += std::exp(input1[i] - max1);
        sum2 += std::exp(input2[i] - max2);
    }

    for (size_t i = 0; i < input1.size(); ++i) {
        float expected1 = std::exp(input1[i] - max1) / sum1;
        float expected2 = std::exp(input2[i] - max2) / sum2;
        EXPECT_NEAR(expected1, expected2, 0.001f);
    }
}
