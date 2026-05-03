#include <cuda/quantize/fp8_gemm.hpp>
#include <cuda/quantize/fp8_types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

namespace nova {
namespace quantize {
namespace test {

class FP8GEMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA not available";
        }
    }

    static constexpr float kTol = 0.05f;

    float relative_error(float original, float recovered) {
        if (std::abs(original) < 1e-6f) return 0.0f;
        return std::abs(original - recovered) / std::abs(original);
    }

    void fill_random_fp32(std::vector<float>& vec, float scale = 1.0f) {
        for (auto& v : vec) {
            v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }

    std::vector<FP8E4M3> convert_to_fp8e4m3(const std::vector<float>& vec) {
        std::vector<FP8E4M3> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = FP8E4M3(vec[i]);
        }
        return result;
    }

    void reference_gemm(
        const float* a, const float* b, float* c,
        int m, int k, int n) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < k; ++p) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
};

TEST_F(FP8GEMMTest, SmallMatrixMultiplication) {
    int m = 4, k = 4, n = 4;

    std::vector<float> a(m * k), b(k * n), c_ref(m * n);

    a = {1.0f, 2.0f, 3.0f, 4.0f,
         5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f};

    b = {1.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 1.0f};

    reference_gemm(a.data(), b.data(), c_ref.data(), m, k, n);

    auto a_fp8 = convert_to_fp8e4m3(a);
    auto b_fp8 = convert_to_fp8e4m3(b);

    std::vector<float> c_gpu(m * n, 0.0f);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
    cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    FP8GEMM::Config config;
    config.scale_a = 1.0f;
    config.scale_b = 1.0f;
    config.scale_out = 1.0f;

    FP8GEMM::forward(
        reinterpret_cast<const FP8E4M3*>(d_a),
        reinterpret_cast<const FP8E4M3*>(d_b),
        d_c, m, k, n, config, 0);

    cudaMemcpy(c_gpu.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m * n; ++i) {
        float rel_err = relative_error(c_ref[i], c_gpu[i]);
        EXPECT_LE(rel_err, kTol) << "Mismatch at index " << i;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(FP8GEMMTest, IdentityMatrix) {
    int m = 8, k = 8, n = 8;

    std::vector<float> a(m * k), b(k * n), c_ref(m * n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            a[i * k + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            b[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    reference_gemm(a.data(), b.data(), c_ref.data(), m, k, n);

    auto a_fp8 = convert_to_fp8e4m3(a);
    auto b_fp8 = convert_to_fp8e4m3(b);

    std::vector<float> c_gpu(m * n, 0.0f);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
    cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    FP8GEMM::Config config;

    FP8GEMM::forward(
        reinterpret_cast<const FP8E4M3*>(d_a),
        reinterpret_cast<const FP8E4M3*>(d_b),
        d_c, m, k, n, config, 0);

    cudaMemcpy(c_gpu.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float rel_err = relative_error(expected, c_gpu[i * n + j]);
            EXPECT_LE(rel_err, kTol) << "Mismatch at [" << i << "," << j << "]";
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(FP8GEMMTest, RandomMatrices) {
    int m = 16, k = 16, n = 16;

    std::vector<float> a(m * k), b(k * n), c_ref(m * n);

    fill_random_fp32(a, 1.0f);
    fill_random_fp32(b, 1.0f);

    reference_gemm(a.data(), b.data(), c_ref.data(), m, k, n);

    auto a_fp8 = convert_to_fp8e4m3(a);
    auto b_fp8 = convert_to_fp8e4m3(b);

    std::vector<float> c_gpu(m * n, 0.0f);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
    cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    FP8GEMM::Config config;

    FP8GEMM::forward(
        reinterpret_cast<const FP8E4M3*>(d_a),
        reinterpret_cast<const FP8E4M3*>(d_b),
        d_c, m, k, n, config, 0);

    cudaMemcpy(c_gpu.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_rel_err = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float rel_err = relative_error(c_ref[i], c_gpu[i]);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    EXPECT_LE(max_rel_err, 0.1f) << "Max relative error: " << max_rel_err;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(FP8GEMMTest, ConfigScaling) {
    int m = 4, k = 4, n = 4;

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> b = {1.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f};

    auto a_fp8 = convert_to_fp8e4m3(a);
    auto b_fp8 = convert_to_fp8e4m3(b);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(FP8E4M3));
    cudaMalloc(&d_b, k * n * sizeof(FP8E4M3));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a_fp8.data(), m * k * sizeof(FP8E4M3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_fp8.data(), k * n * sizeof(FP8E4M3), cudaMemcpyHostToDevice);

    {
        FP8GEMM::Config config;
        config.scale_a = 1.0f;
        config.scale_b = 1.0f;
        config.scale_out = 1.0f;

        FP8GEMM::forward(
            reinterpret_cast<const FP8E4M3*>(d_a),
            reinterpret_cast<const FP8E4M3*>(d_b),
            d_c, m, k, n, config, 0);

        std::vector<float> c1(m * n);
        cudaMemcpy(c1.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    EXPECT_NEAR(c1[i * n + j], static_cast<float>(i + 1), 1.0f);
                }
            }
        }
    }

    {
        FP8GEMM::Config config;
        config.scale_a = 2.0f;
        config.scale_b = 0.5f;
        config.scale_out = 1.0f;

        FP8GEMM::forward(
            reinterpret_cast<const FP8E4M3*>(d_a),
            reinterpret_cast<const FP8E4M3*>(d_b),
            d_c, m, k, n, config, 0);

        std::vector<float> c2(m * n);
        cudaMemcpy(c2.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(FP8GEMMTest, WorkspaceSizeQuery) {
    int m = 128, k = 256, n = 128;

    FP8GEMM::Config config;

    size_t workspace = FP8GEMM::get_workspace_size(m, k, n, config);

    EXPECT_GE(workspace, static_cast<size_t>(m * k * sizeof(FP8E4M3)));
    EXPECT_GE(workspace, static_cast<size_t>(k * n * sizeof(FP8E4M3)));
    EXPECT_GE(workspace, static_cast<size_t>(m * n * sizeof(float)));
}

} // namespace test
} // namespace quantize
} // namespace nova
