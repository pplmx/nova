#include <gtest/gtest.h>
#include "matrix/mult.h"
#include "cuda/device/device_utils.h"

#include <vector>
#include <cmath>

class MatrixMultTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(MatrixMultTest, NaiveSquareMatrices) {
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C(4, 0);

    multiplyMatricesNaive(A.data(), B.data(), C.data(), 2, 2, 2);

    EXPECT_NEAR(C[0], 19, 0.001);
    EXPECT_NEAR(C[1], 22, 0.001);
    EXPECT_NEAR(C[2], 43, 0.001);
    EXPECT_NEAR(C[3], 50, 0.001);
}

TEST_F(MatrixMultTest, TiledMatchesNaive) {
    const int N = 64;
    std::vector<float> A(N * N), B(N * N);
    std::vector<float> C_naive(N * N), C_tiled(N * N);

    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i % 100) / 10.0f;
        B[i] = static_cast<float>((i * 7) % 100) / 10.0f;
    }

    multiplyMatricesNaive(A.data(), B.data(), C_naive.data(), N, N, N);
    multiplyMatricesTiled(A.data(), B.data(), C_tiled.data(), N, N, N);

    for (int i = 0; i < N * N; ++i) {
        EXPECT_NEAR(C_naive[i], C_tiled[i], 0.001);
    }
}

TEST_F(MatrixMultTest, NonSquareMatrices) {
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {1, 0, 0, 1, 1, 1};
    std::vector<float> C(4, 0);

    multiplyMatricesNaive(A.data(), B.data(), C.data(), 2, 3, 2);

    EXPECT_NEAR(C[0], 4, 0.001);
    EXPECT_NEAR(C[1], 5, 0.001);
    EXPECT_NEAR(C[2], 10, 0.001);
    EXPECT_NEAR(C[3], 11, 0.001);
}

TEST_F(MatrixMultTest, IdentityMatrix) {
    const int N = 4;
    std::vector<float> A(N * N), B(N * N, 0), C(N * N);

    for (int i = 0; i < N; ++i) {
        A[i * N + i] = 1.0f;
        B[i * N + i] = 1.0f;
    }

    multiplyMatricesNaive(A.data(), B.data(), C.data(), N, N, N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                EXPECT_NEAR(C[i * N + j], 1.0f, 0.001);
            } else {
                EXPECT_NEAR(C[i * N + j], 0.0f, 0.001);
            }
        }
    }
}

TEST_F(MatrixMultTest, ZeroMatrix) {
    const int N = 3;
    std::vector<float> A(N * N, 1.0f), B(N * N, 2.0f), C(N * N);

    multiplyMatricesNaive(A.data(), B.data(), C.data(), N, N, N);

    float expected = N * 1.0f * 2.0f;
    for (float val : C) {
        EXPECT_NEAR(val, expected, 0.001);
    }
}

TEST_F(MatrixMultTest, DoublePrecision) {
    std::vector<double> A = {1.5, 2.5, 3.5, 4.5};
    std::vector<double> B = {5.5, 6.5, 7.5, 8.5};
    std::vector<double> C(4, 0);

    multiplyMatricesNaive(A.data(), B.data(), C.data(), 2, 2, 2);

    EXPECT_NEAR(C[0], 27.0, 0.001);
    EXPECT_NEAR(C[1], 31.0, 0.001);
    EXPECT_NEAR(C[2], 53.0, 0.001);
    EXPECT_NEAR(C[3], 61.0, 0.001);
}

TEST_F(MatrixMultTest, LargeMatrix) {
    const int N = 512;
    std::vector<float> A(N * N), B(N * N), C_naive(N * N), C_tiled(N * N);

    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i % 10);
        B[i] = static_cast<float>((i * 3) % 10);
    }

    multiplyMatricesNaive(A.data(), B.data(), C_naive.data(), N, N, N);
    multiplyMatricesTiled(A.data(), B.data(), C_tiled.data(), N, N, N);

    float maxDiff = 0;
    for (int i = 0; i < N * N; ++i) {
        float diff = std::abs(C_naive[i] - C_tiled[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    EXPECT_LT(maxDiff, 0.001);
}
