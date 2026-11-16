/**
 * @file int8_calibration_edge_test.cpp
 * @brief Contracts for the INT8 quantisation and calibration entry points
 *
 * src/cuda/quantize/int8_kernels.cu has five entry points that launch kernels. Two of
 * them, compute_minmax and build_histogram, were already written to handle an empty
 * input; the other three derived a grid straight from n:
 *
 *     size_t grid_size = (n + block_size - 1) / block_size;
 *
 * With n == 0 that is a zero-sized grid. Measured before the fix:
 *
 *     quantize_f32_to_int8(n=0)    -> "CUDA error in quantize_f32_to_int8:
 *                                      invalid configuration argument"
 *     dequantize_int8_to_f32(n=0)  -> the same
 *
 * build_histogram failed differently: n == 0 reached cudaMalloc(0) and the copy that
 * followed, and num_bins == 0 reached cudaMemset with a zero byte count. Both reported a
 * bare "invalid argument" naming neither the parameter nor the cause.
 */

#include <cuda/quantize/int8_kernels.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
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

// Regression: these printed "invalid configuration argument" for an empty input.
TEST(Int8CalibrationEdgeTest, EmptyInputIsANoOp) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    QuantizationParams params{};
    params.scale = 1.0f;
    params.zero_point = 0;

    (void)cudaGetLastError();

    EXPECT_NO_THROW(quantize_f32_to_int8(nullptr, nullptr, 0, params, 0));
    EXPECT_NO_THROW(dequantize_int8_to_f32(nullptr, nullptr, 0, params, 0));

    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "an empty conversion must not leave a launch error behind";
}

// Regression: num_bins == 0 reached cudaMemset with a zero count.
TEST(Int8CalibrationEdgeTest, ZeroBinsIsRejectedWithAReason) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    uint32_t bins[16] = {};
    const std::string message = [&] {
        try {
            build_histogram(nullptr, bins, 8, 0.0f, 1.0f, 0, 0);
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
        return std::string();
    }();

    EXPECT_FALSE(message.empty()) << "a histogram with no bins cannot be built";
    EXPECT_NE(message.find("num_bins"), std::string::npos)
        << "the message should name the parameter: " << message;
    (void)cudaGetLastError();
}

TEST(Int8CalibrationEdgeTest, EmptySampleSetBuildsNothing) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // n == 0 with a valid num_bins is an ordinary empty calibration.
    uint32_t bins[16] = {};
    (void)cudaGetLastError();
    EXPECT_NO_THROW(build_histogram(nullptr, bins, 0, 0.0f, 1.0f, 16, 0));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(Int8CalibrationEdgeTest, MinMaxOfAnEmptySetIsZero) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // This sibling was already correct; the case records the behaviour so a future edit
    // to the file keeps it consistent with the other four.
    float min_val = 123.0f;
    float max_val = 456.0f;

    EXPECT_NO_THROW(compute_minmax(nullptr, 0, &min_val, &max_val, 0));
    EXPECT_FLOAT_EQ(min_val, 0.0f);
    EXPECT_FLOAT_EQ(max_val, 0.0f);
}

TEST(Int8CalibrationEdgeTest, RoundTripThroughInt8IsWithinTheQuantisationStep) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Symmetric quantisation with scale 1/16: the round trip must land within half a
    // step of the original, which is the definition of correct rounding.
    const std::vector<float> input = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 0.25f, -0.25f};
    const size_t n = input.size();
    constexpr float scale = 1.0f / 16.0f;

    QuantizationParams params{};
    params.scale = scale;
    params.zero_point = 0;

    float* d_input = nullptr;
    int8_t* d_int8 = nullptr;
    float* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_int8, n * sizeof(int8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, n * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);

    quantize_f32_to_int8(d_input, d_int8, n, params, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dequantize_int8_to_f32(d_int8, d_output, n, params, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> result(n, 0.0f);
    ASSERT_EQ(cudaMemcpy(result.data(), d_output, n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(result[i], input[i], scale) << "element " << i;
    }

    cudaFree(d_input);
    cudaFree(d_int8);
    cudaFree(d_output);
}

}  // namespace nova::quantize::cuda::test
