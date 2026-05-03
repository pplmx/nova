#include <cuda/quantize/int8_kernels.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdint>

namespace nova {
namespace quantize {
namespace test {

class INT8KernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA not available";
        }
    }

    static constexpr float kTol = 0.01f;

    float relative_error(float original, float recovered) {
        if (std::abs(original) < 1e-6f) return 0.0f;
        return std::abs(original - recovered) / std::abs(original);
    }

    void reference_quantize(
        const float* src, int8_t* dst, size_t n, float scale, float zero_point = 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            float val = src[i];
            int32_t quantized = static_cast<int32_t>(std::round((val - zero_point) / scale));
            quantized = std::max(-127, std::min(127, quantized));
            dst[i] = static_cast<int8_t>(quantized);
        }
    }

    void reference_dequantize(
        const int8_t* src, float* dst, size_t n, float scale, float zero_point = 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = (static_cast<float>(src[i]) + zero_point) * scale;
        }
    }
};

TEST_F(INT8KernelsTest, SymmetricQuantization) {
    size_t n = 1024;
    std::vector<float> src(n);
    std::vector<int8_t> ref_quant(n), dst_quant(n);

    float scale = 0.1f;
    for (size_t i = 0; i < n; ++i) {
        src[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
    }

    reference_quantize(src.data(), ref_quant.data(), n, scale);

    float* d_src;
    int8_t* d_dst;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, n * sizeof(int8_t));

    cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda::QuantizationParams params(scale, 0.0f, true);
    cuda::quantize_f32_to_int8(d_src, d_dst, n, params);

    cudaMemcpy(dst_quant.data(), d_dst, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(dst_quant[i], ref_quant[i]) << "Mismatch at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_F(INT8KernelsTest, SymmetricDequantization) {
    size_t n = 1024;
    std::vector<int8_t> src(n);
    std::vector<float> ref_dst(n), dst(n);

    float scale = 0.1f;
    for (size_t i = 0; i < n; ++i) {
        src[i] = static_cast<int8_t>((rand() % 255) - 127);
    }

    reference_dequantize(src.data(), ref_dst.data(), n, scale);

    int8_t* d_src;
    float* d_dst;
    cudaMalloc(&d_src, n * sizeof(int8_t));
    cudaMalloc(&d_dst, n * sizeof(float));

    cudaMemcpy(d_src, src.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice);

    cuda::QuantizationParams params(scale, 0.0f, true);
    cuda::dequantize_int8_to_f32(d_src, d_dst, n, params);

    cudaMemcpy(dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(dst[i], ref_dst[i], kTol) << "Mismatch at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_F(INT8KernelsTest, RoundtripAccuracy) {
    size_t n = 1024;
    std::vector<float> src(n), recovered(n);
    std::vector<int8_t> quantized(n);

    float scale = 0.05f;
    for (size_t i = 0; i < n; ++i) {
        src[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
    }

    float* d_src;
    int8_t* d_quant;
    float* d_dst;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_quant, n * sizeof(int8_t));
    cudaMalloc(&d_dst, n * sizeof(float));

    cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda::QuantizationParams params(scale);
    cuda::quantize_f32_to_int8(d_src, d_quant, n, params);
    cuda::dequantize_int8_to_f32(d_quant, d_dst, n, params);

    cudaMemcpy(recovered.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_rel_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float rel_err = relative_error(src[i], recovered[i]);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    EXPECT_LE(max_rel_err, 0.1f) << "Max relative error: " << max_rel_err;

    cudaFree(d_src);
    cudaFree(d_quant);
    cudaFree(d_dst);
}

TEST_F(INT8KernelsTest, ZeroValues) {
    size_t n = 100;
    std::vector<float> src(n, 0.0f);
    std::vector<int8_t> dst(n);

    float* d_src;
    int8_t* d_dst;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, n * sizeof(int8_t));

    cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda::QuantizationParams params(0.1f);
    cuda::quantize_f32_to_int8(d_src, d_dst, n, params);

    cudaMemcpy(dst.data(), d_dst, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(dst[i], 0) << "Zero should map to zero at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_F(INT8KernelsTest, Clamping) {
    size_t n = 100;
    std::vector<float> src(n);
    std::vector<int8_t> dst(n);

    for (size_t i = 0; i < n; ++i) {
        src[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;
    }

    float* d_src;
    int8_t* d_dst;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, n * sizeof(int8_t));

    cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda::QuantizationParams params(0.1f);
    cuda::quantize_f32_to_int8(d_src, d_dst, n, params);

    cudaMemcpy(dst.data(), d_dst, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(dst[i], -127) << "Value should be clamped to -127 at index " << i;
        EXPECT_LE(dst[i], 127) << "Value should be clamped to 127 at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_F(INT8KernelsTest, ComputeMinMax) {
    size_t n = 1024;
    std::vector<float> src(n);
    float h_min, h_max;

    for (size_t i = 0; i < n; ++i) {
        src[i] = static_cast<float>(i) / n * 100.0f - 50.0f;
    }

    float* d_src;
    float* d_min;
    float* d_max;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));

    cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda::compute_minmax(d_src, n, d_min, d_max);

    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_min, -50.0f, 0.1f);
    EXPECT_NEAR(h_max, 49.0f, 0.1f);

    cudaFree(d_src);
    cudaFree(d_min);
    cudaFree(d_max);
}

TEST_F(INT8KernelsTest, HistogramComputation) {
    size_t n = 1024;
    size_t hist_size = 256;
    std::vector<float> src(n);
    std::vector<uint32_t> histogram(hist_size, 0);

    for (size_t i = 0; i < n; ++i) {
        src[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }

    float* d_src;
    uint32_t* d_hist;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_hist, hist_size * sizeof(uint32_t));

    cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cuda::build_histogram(d_src, d_hist, n, 0.0f, 100.0f, static_cast<int>(hist_size));

    cudaMemcpy(histogram.data(), d_hist, hist_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint64_t total_count = 0;
    for (size_t i = 0; i < hist_size; ++i) {
        total_count += histogram[i];
    }

    EXPECT_EQ(total_count, n) << "Histogram should contain all elements";

    cudaFree(d_src);
    cudaFree(d_hist);
}

} // namespace test
} // namespace quantize
} // namespace nova
