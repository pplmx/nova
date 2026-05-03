#include <cuda/quantize/fp8_activation.hpp>
#include <cuda/quantize/fp8_types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

namespace nova {
namespace quantize {
namespace test {

class FP8ActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA not available";
        }
    }

    static constexpr float kTol = 0.1f;

    float reference_relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    float reference_gelu(float x) {
        constexpr float sqrt_2_inv = 0.7071067811865476f;
        constexpr float half = 0.5f;
        float cdf = half * (1.0f + std::erff(x * sqrt_2_inv));
        return x * cdf;
    }

    float reference_sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float reference_leaky_relu(float x, float alpha = 0.01f) {
        return x >= 0.0f ? x : alpha * x;
    }

    template<typename FP8Type>
    float fp8_to_float(FP8Type val) {
        return static_cast<float>(val);
    }

    float to_float(FP8E4M3 val) { return static_cast<float>(val); }
    float to_float(FP8E5M2 val) { return static_cast<float>(val); }

    float relative_error(float original, float recovered) {
        if (std::abs(original) < 1e-6f) return 0.0f;
        return std::abs(original - recovered) / std::abs(original);
    }
};

TEST_F(FP8ActivationTest, ReluFP8PositiveValuesUnchanged) {
    size_t n = 64;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(i + 1) * 0.5f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::relu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        EXPECT_EQ(out_float, orig_float) << "Positive values should be unchanged at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, ReluFP8NegativeValuesClampedToZero) {
    size_t n = 64;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = -static_cast<float>(i + 1) * 0.5f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::relu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float out_float = static_cast<float>(output[i]);
        EXPECT_EQ(out_float, 0.0f) << "Negative values should be clamped to zero at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, ReluFP8MixedValues) {
    size_t n = 128;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = (i % 2 == 0) ? static_cast<float>(i) * 0.5f : -static_cast<float>(i) * 0.5f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::relu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        float expected = reference_relu(orig_float);
        float rel_err = relative_error(expected, out_float);
        EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, GeluFP8Accuracy) {
    size_t n = 64;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::gelu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        float expected = reference_gelu(orig_float);
        float rel_err = relative_error(expected, out_float);
        EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i << ": expected " << expected << ", got " << out_float;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, SigmoidFP8Bounds) {
    size_t n = 64;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::sigmoid(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float out_float = static_cast<float>(output[i]);
        EXPECT_GE(out_float, 0.0f) << "Sigmoid should be >= 0 at index " << i;
        EXPECT_LE(out_float, 1.0f) << "Sigmoid should be <= 1 at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, SigmoidFP8Accuracy) {
    size_t n = 32;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::sigmoid(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        float expected = reference_sigmoid(orig_float);
        float rel_err = relative_error(expected, out_float);
        EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, LeakyReluFP8NegativeSlope) {
    size_t n = 64;
    float alpha = 0.02f;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = -static_cast<float>(i + 1) * 0.5f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::leaky_relu(d_input, d_output, n, alpha, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        float expected = reference_leaky_relu(orig_float, alpha);
        float rel_err = relative_error(expected, out_float);
        EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, LeakyReluFP8PositiveValuesUnchanged) {
    size_t n = 64;
    float alpha = 0.01f;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(i + 1) * 0.5f;
        input[i] = FP8E4M3(val);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::leaky_relu(d_input, d_output, n, alpha, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        EXPECT_EQ(out_float, orig_float) << "Positive values should be unchanged at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, FP8E5M2Relu) {
    size_t n = 32;
    std::vector<FP8E5M2> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
        input[i] = FP8E5M2(val);
    }

    FP8E5M2 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E5M2));
    cudaMalloc(&d_output, n * sizeof(FP8E5M2));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E5M2), cudaMemcpyHostToDevice);

    cuda::relu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E5M2), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float out_float = static_cast<float>(output[i]);
        EXPECT_GE(out_float, 0.0f) << "All outputs should be >= 0 at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, FP8E5M2Gelu) {
    size_t n = 32;
    std::vector<FP8E5M2> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 8.0f;
        input[i] = FP8E5M2(val);
    }

    FP8E5M2 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E5M2));
    cudaMalloc(&d_output, n * sizeof(FP8E5M2));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E5M2), cudaMemcpyHostToDevice);

    cuda::gelu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E5M2), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        float orig_float = static_cast<float>(input[i]);
        float out_float = static_cast<float>(output[i]);
        float expected = reference_gelu(orig_float);
        float rel_err = relative_error(expected, out_float);
        EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST_F(FP8ActivationTest, ActivationZeroInput) {
    size_t n = 16;
    std::vector<FP8E4M3> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] = FP8E4M3(0.0f);
    }

    FP8E4M3 *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(FP8E4M3));
    cudaMalloc(&d_output, n * sizeof(FP8E4M3));

    cudaMemcpy(d_input, input.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    cuda::relu(d_input, d_output, n, 0);

    cudaMemcpy(output.data(), d_output, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(static_cast<float>(output[i]), 0.0f) << "ReLU(0) should be 0 at index " << i;
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

} // namespace test
} // namespace quantize
} // namespace nova
