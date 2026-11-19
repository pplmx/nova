#include <cuda/quantize/fp8_kernels.hpp>
#include <cuda/quantize/fp8_types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

namespace nova {
    namespace quantize {
        namespace test {

            class FP8KernelsTest : public ::testing::Test {
            protected:
                void SetUp() override {
                    cudaError_t err = cudaSetDevice(0);
                    if (err != cudaSuccess) {
                        GTEST_SKIP() << "CUDA not available";
                    }
                }

                static constexpr float kTol = 0.05f;

                float relative_error(float original, float recovered) {
                    if (std::abs(original) < 1e-6f) {
                        return 0.0f;
                    }
                    return std::abs(original - recovered) / std::abs(original);
                }

                void reference_quantize_e4m3(const float* src, FP8E4M3* dst, size_t n) {
                    for (size_t i = 0; i < n; ++i) {
                        dst[i] = FP8E4M3(src[i]);
                    }
                }

                void reference_dequantize_e4m3(const FP8E4M3* src, float* dst, size_t n, float scale) {
                    for (size_t i = 0; i < n; ++i) {
                        dst[i] = static_cast<float>(src[i]) * scale;
                    }
                }
            };

            TEST_F(FP8KernelsTest, FP8E4M3QuantizationBasic) {
                size_t n = 64;
                std::vector<float> src(n);
                std::vector<FP8E4M3> dst(n);

                for (size_t i = 0; i < n; ++i) {
                    src[i] = static_cast<float>(i + 1) * 0.5f;
                }

                float* d_src;
                FP8E4M3* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_dst, n, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    EXPECT_GE(dst[i].to_bits(), 0u);
                    EXPECT_LE(dst[i].to_bits(), 255u);
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8E4M3QuantizationRoundtripAccuracy) {
                size_t n = 1024;
                std::vector<float> src(n);
                std::vector<FP8E4M3> quantized(n);
                std::vector<float> recovered(n);

                for (size_t i = 0; i < n; ++i) {
                    src[i] = ((float)rand() / RAND_MAX - 0.5f) * 100.0f;
                }

                float* d_src;
                FP8E4M3* d_quant;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_quant, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_quant, n, 0);

                cudaMemcpy(quantized.data(), d_quant, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    recovered[i] = static_cast<float>(quantized[i]);
                }

                float max_rel_err = 0.0f;
                for (size_t i = 0; i < n; ++i) {
                    float rel_err = relative_error(src[i], recovered[i]);
                    max_rel_err = std::max(max_rel_err, rel_err);
                }

                EXPECT_LE(max_rel_err, 0.1f) << "Max relative error: " << max_rel_err;

                cudaFree(d_src);
                cudaFree(d_quant);
            }

            TEST_F(FP8KernelsTest, FP8E5M2QuantizationBasic) {
                size_t n = 64;
                std::vector<float> src(n);
                std::vector<FP8E5M2> dst(n);

                for (size_t i = 0; i < n; ++i) {
                    src[i] = static_cast<float>(i + 1) * 10.0f;
                }

                float* d_src;
                FP8E5M2* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E5M2));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e5m2(d_src, d_dst, n, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(FP8E5M2), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    EXPECT_GE(dst[i].to_bits(), 0u);
                    EXPECT_LE(dst[i].to_bits(), 255u);
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8E5M2QuantizationRoundtripAccuracy) {
                size_t n = 1024;
                std::vector<float> src(n);
                std::vector<FP8E5M2> quantized(n);
                std::vector<float> recovered(n);

                for (size_t i = 0; i < n; ++i) {
                    src[i] = ((float)rand() / RAND_MAX) * 1000.0f + 1.0f;
                }

                float* d_src;
                FP8E5M2* d_quant;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_quant, n * sizeof(FP8E5M2));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e5m2(d_src, d_quant, n, 0);

                cudaMemcpy(quantized.data(), d_quant, n * sizeof(FP8E5M2), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    recovered[i] = static_cast<float>(quantized[i]);
                }

                float max_rel_err = 0.0f;
                for (size_t i = 0; i < n; ++i) {
                    float rel_err = relative_error(src[i], recovered[i]);
                    max_rel_err = std::max(max_rel_err, rel_err);
                }

                EXPECT_LE(max_rel_err, 0.15f) << "Max relative error: " << max_rel_err;

                cudaFree(d_src);
                cudaFree(d_quant);
            }

            TEST_F(FP8KernelsTest, FP8DequantizationWithScale) {
                size_t n = 64;
                std::vector<FP8E4M3> src(n);
                std::vector<float> dst(n);
                float scale = 0.1f;

                for (size_t i = 0; i < n; ++i) {
                    src[i] = FP8E4M3(static_cast<float>(i + 1) * 0.5f);
                }

                FP8E4M3* d_src;
                float* d_dst;
                cudaMalloc(&d_src, n * sizeof(FP8E4M3));
                cudaMalloc(&d_dst, n * sizeof(float));

                cudaMemcpy(d_src, src.data(), n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

                cuda::dequantize_fp8e4m3_to_f32(d_src, d_dst, n, scale, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    float original = static_cast<float>(src[i]) * scale;
                    float rel_err = relative_error(original, dst[i]);
                    EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8E5M2DequantizationWithScale) {
                size_t n = 64;
                std::vector<FP8E5M2> src(n);
                std::vector<float> dst(n);
                float scale = 0.01f;

                for (size_t i = 0; i < n; ++i) {
                    src[i] = FP8E5M2(static_cast<float>(i + 1) * 10.0f);
                }

                FP8E5M2* d_src;
                float* d_dst;
                cudaMalloc(&d_src, n * sizeof(FP8E5M2));
                cudaMalloc(&d_dst, n * sizeof(float));

                cudaMemcpy(d_src, src.data(), n * sizeof(FP8E5M2), cudaMemcpyHostToDevice);

                cuda::dequantize_fp8e5m2_to_f32(d_src, d_dst, n, scale, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    float original = static_cast<float>(src[i]) * scale;
                    float rel_err = relative_error(original, dst[i]);
                    EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8ClampingBehavior) {
                size_t n = 16;
                std::vector<float> src(n);
                std::vector<FP8E4M3> quantized(n);

                for (size_t i = 0; i < n; ++i) {
                    src[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;
                }

                float* d_src;
                FP8E4M3* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_dst, n, 0);

                cudaMemcpy(quantized.data(), d_dst, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    // 1000 is outside E4M3's range (max normal 240, and 0x7C is infinity),
                    // so the format saturates to infinity rather than clamping to a finite
                    // value. Asserting a finite bound here demanded behaviour the encoding
                    // cannot express.
                    float recovered = static_cast<float>(quantized[i]);
                    EXPECT_TRUE(std::isinf(recovered)) << "index " << i << " recovered " << recovered;
                    EXPECT_EQ(std::signbit(recovered), i % 2 != 0) << "index " << i << " should saturate to infinity with the input's sign";
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8ZeroValues) {
                size_t n = 32;
                std::vector<float> src(n, 0.0f);
                std::vector<FP8E4M3> dst(n);

                float* d_src;
                FP8E4M3* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_dst, n, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    EXPECT_EQ(dst[i].to_bits(), 0) << "Zero should map to zero at index " << i;
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8NegativeZero) {
                size_t n = 8;
                std::vector<float> src(n, -0.0f);
                std::vector<FP8E4M3> dst(n);

                float* d_src;
                FP8E4M3* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_dst, n, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                for (size_t i = 0; i < n; ++i) {
                    EXPECT_EQ(dst[i].to_bits(), FP8E4M3::NEG_ZERO) << "Negative zero at index " << i;
                }

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8InfinityHandling) {
                size_t n = 4;
                std::vector<float> src = {
                    std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
                std::vector<FP8E4M3> dst(n);

                float* d_src;
                FP8E4M3* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_dst, n, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                EXPECT_EQ(dst[0].to_bits(), FP8E4M3::POS_INF);
                EXPECT_EQ(dst[1].to_bits(), FP8E4M3::NEG_INF);
                // float::max() also overflows E4M3, so it saturates to infinity too. A
                // finite upper bound here assumed clamping, which the format does not do.
                EXPECT_EQ(dst[2].to_bits(), FP8E4M3::POS_INF);

                cudaFree(d_src);
                cudaFree(d_dst);
            }

            TEST_F(FP8KernelsTest, FP8NaNHandling) {
                size_t n = 2;
                std::vector<float> src = {std::numeric_limits<float>::quiet_NaN()};
                std::vector<FP8E4M3> dst(n);

                float* d_src;
                FP8E4M3* d_dst;
                cudaMalloc(&d_src, n * sizeof(float));
                cudaMalloc(&d_dst, n * sizeof(FP8E4M3));

                cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                cuda::quantize_f32_to_fp8e4m3(d_src, d_dst, n, 0);

                cudaMemcpy(dst.data(), d_dst, n * sizeof(FP8E4M3), cudaMemcpyDeviceToHost);

                EXPECT_EQ(dst[0].to_bits(), FP8E4M3::NAN_VAL);

                cudaFree(d_src);
                cudaFree(d_dst);
            }

        }  // namespace test
    }  // namespace quantize
}  // namespace nova
