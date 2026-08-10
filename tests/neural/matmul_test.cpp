#include <gtest/gtest.h>
#include "cuda/neural/matmul.h"

#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <random>
#include <vector>

using namespace cuda::neural;

class MatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(MatmulTest, MatmulBatchMatchesPerSliceMatmul) {
    // matmul_batch must equal applying the single-GEMM matmul to each batch
    // slice independently. A/B/C are contiguous row-major stacked matrices
    // (stride = one full matrix), which is the layout the benchmark and
    // callers pass.
    const int batch = 3, m = 4, n = 5, k = 6;
    const size_t a_count = static_cast<size_t>(batch) * m * k;
    const size_t b_count = static_cast<size_t>(batch) * k * n;
    const size_t c_count = static_cast<size_t>(batch) * m * n;

    std::vector<float> h_a(a_count), h_b(b_count), h_c(c_count);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : h_a) v = dist(rng);
    for (float& v : h_b) v = dist(rng);
    for (float& v : h_c) v = dist(rng);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* d_ref = nullptr;
    ASSERT_EQ(cudaMalloc(&d_a, a_count * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, b_count * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, c_count * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ref, c_count * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), c_count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    // Use a private handle so the fixture's cudaDeviceReset() cannot leave a
    // stale global handle behind.
    cublasHandle_t h;
    ASSERT_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
    MatmulOptions opts;
    opts.handle = h;

    matmul_batch(d_a, d_b, d_c, batch, m, n, k, opts);

    for (int b = 0; b < batch; ++b) {
        matmul(d_a + b * m * k, d_b + b * k * n, d_ref + b * m * n, m, n, k, opts);
    }
    cublasDestroy(h);

    std::vector<float> got(c_count), expected(c_count);
    ASSERT_EQ(cudaMemcpy(got.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(expected.data(), d_ref, c_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (size_t i = 0; i < c_count; ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-4f) << "index " << i;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ref);
}

TEST_F(MatmulTest, GetCublasHandleReturnsValidHandle) {
    cublasHandle_t handle = get_cublas_handle();
    EXPECT_NE(handle, nullptr);
}

TEST_F(MatmulTest, MatmulOptionsHaveDefaults) {
    MatmulOptions options;
    EXPECT_EQ(options.alpha, 1.0f);
    EXPECT_EQ(options.beta, 0.0f);
    EXPECT_EQ(options.trans_a, CUBLAS_OP_N);
    EXPECT_EQ(options.trans_b, CUBLAS_OP_N);
}

TEST_F(MatmulTest, MatmulOptionsCanBeCustomized) {
    MatmulOptions options;
    options.alpha = 2.0f;
    options.beta = 1.0f;
    options.trans_a = CUBLAS_OP_T;

    EXPECT_EQ(options.alpha, 2.0f);
    EXPECT_EQ(options.beta, 1.0f);
    EXPECT_EQ(options.trans_a, CUBLAS_OP_T);
}

TEST_F(MatmulTest, MatmulOptionsWithCublasHandle) {
    cublasHandle_t handle = get_cublas_handle();
    MatmulOptions options;
    options.handle = handle;

    EXPECT_EQ(options.handle, handle);
}
