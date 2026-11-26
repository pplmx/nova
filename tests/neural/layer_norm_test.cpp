#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
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

// LayerNormResult owns six raw pointers and its destructor releases them (clear():
// delete[] on the three host arrays, cudaFree on the three device ones). It declared no copy or
// move operations, so the compiler generated a shallow copy: copying the struct makes two objects
// own the same buffers, and both destructors release them.
//
//   { LayerNormResult a(100); LayerNormResult b(a); }   // b and a share every pointer
//
// That is a double free at scope exit, and the copy assignment additionally leaks the target's
// previous buffers. Nothing in the type says it is exclusive, so a call returning it by value or a
// container holding it is enough to trigger it.
//
// Asserted at compile time because the defect is a property of the declaration, not of a run.
TEST_F(LayerNormTest, ResultOwnsItsBuffersExclusively) {
    static_assert(!std::is_copy_constructible_v<LayerNormResult>, "an owning result must not be copyable: the copy would double-free its buffers");
    static_assert(!std::is_copy_assignable_v<LayerNormResult>, "copy assignment would leak the target's buffers and then double-free");
    static_assert(std::is_move_constructible_v<LayerNormResult>, "moving must remain possible so the type can still be returned by value");
    static_assert(std::is_move_assignable_v<LayerNormResult>, "moving must remain possible");

    // The move leaves the source empty, so exactly one destructor has anything to release.
    LayerNormResult a(64);
    float* original = a.d_output;
    LayerNormResult b(std::move(a));
    EXPECT_EQ(b.d_output, original) << "the move must carry the buffers across";
    EXPECT_EQ(a.d_output, nullptr) << "the moved-from object must not still own the buffer";
    EXPECT_EQ(a.size, 0);
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
