/**
 * @file fp8_conversion_edge_test.cpp
 * @brief Contracts for the FP8 conversion entry points on an empty input
 *
 * src/cuda/quantize/fp8_kernels.cu computed a grid as (n + 255) / 256 and launched it
 * unconditionally. With n == 0 that is a zero-sized grid, which CUDA rejects with a
 * generic "invalid configuration argument". The sibling file fp8_activation.cu already
 * short-circuits through its empty_launch helper; this one did not.
 *
 * Converting nothing is an ordinary request — a batch can legitimately be empty — so all
 * four entry points now return without launching.
 *
 * The conversion values themselves are covered by the existing fp8 suites; what this file
 * adds is the empty-input contract.
 */

#include <cuda/quantize/fp8_kernels.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace nova::quantize::cuda::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

// Regression: all four threw "invalid configuration argument" for n == 0.
TEST(Fp8ConversionEdgeTest, EmptyInputIsANoOp) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    (void)cudaGetLastError();

    EXPECT_NO_THROW(quantize_f32_to_fp8e4m3(nullptr, nullptr, 0));
    EXPECT_NO_THROW(quantize_f32_to_fp8e5m2(nullptr, nullptr, 0));
    EXPECT_NO_THROW(dequantize_fp8e4m3_to_f32(nullptr, nullptr, 0, 1.0f));
    EXPECT_NO_THROW(dequantize_fp8e5m2_to_f32(nullptr, nullptr, 0, 1.0f));

    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "an empty conversion must not leave a launch error behind";
}

TEST(Fp8ConversionEdgeTest, ASingleElementRoundTripsThroughE4M3) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // 1.0 and 2.0 are exactly representable in E4M3, so the round trip is lossless and
    // the expectation is not a tolerance question.
    const std::vector<float> input = {1.0f};

    float* d_input = nullptr;
    FP8E4M3* d_fp8 = nullptr;
    float* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_fp8, sizeof(FP8E4M3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    quantize_f32_to_fp8e4m3(d_input, d_fp8, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dequantize_fp8e4m3_to_f32(d_fp8, d_output, 1, 1.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float result = 0.0f;
    ASSERT_EQ(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_FLOAT_EQ(result, 1.0f);

    cudaFree(d_input);
    cudaFree(d_fp8);
    cudaFree(d_output);
}

TEST(Fp8ConversionEdgeTest, DequantizeAppliesTheScale) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Quantise 2.0 (exact in E4M3), then dequantise with a scale of 3: the result must be
    // 6, which distinguishes "applies the scale" from "ignores it".
    const std::vector<float> input = {2.0f};

    float* d_input = nullptr;
    FP8E4M3* d_fp8 = nullptr;
    float* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_fp8, sizeof(FP8E4M3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    quantize_f32_to_fp8e4m3(d_input, d_fp8, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dequantize_fp8e4m3_to_f32(d_fp8, d_output, 1, 3.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float result = 0.0f;
    ASSERT_EQ(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_FLOAT_EQ(result, 6.0f);

    cudaFree(d_input);
    cudaFree(d_fp8);
    cudaFree(d_output);
}

TEST(Fp8ConversionEdgeTest, RoundTripPreservesAContiguousBlock) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Powers of two up to the E4M3 range are exact, so every element must come back
    // unchanged. A block also exercises more than one thread of the grid.
    const std::vector<float> input = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
    const size_t n = input.size();

    float* d_input = nullptr;
    FP8E4M3* d_fp8 = nullptr;
    float* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_fp8, n * sizeof(FP8E4M3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);

    quantize_f32_to_fp8e4m3(d_input, d_fp8, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dequantize_fp8e4m3_to_f32(d_fp8, d_output, n, 1.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> result(n, 0.0f);
    ASSERT_EQ(cudaMemcpy(result.data(), d_output, n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], input[i]) << "element " << i;
    }

    cudaFree(d_input);
    cudaFree(d_fp8);
    cudaFree(d_output);
}

}  // namespace nova::quantize::cuda::test
