/**
 * @file matrix_multiply_edge_test.cpp
 * @brief Contracts for the matrix-multiply entry points on empty extents
 *
 * src/matrix/mult.cu derives its launch grid from the caller's dimensions without
 * checking them:
 *
 *     dim3 numBlocks((numColsB + 15) / 16, (numRowsA + 15) / 16);
 *
 * A zero extent makes one grid dimension zero, which CUDA rejects. Measured before the
 * fix:
 *
 *     multiplyMatricesNaive(rows=0)  -> CUDA error: invalid configuration argument
 *
 * src/matrix/ops.cu already returns early for the same reason in its transpose entry
 * points, so this was an inconsistency within one module rather than a deliberate choice.
 *
 * The arithmetic itself was checked and is correct: C[m x n] = A[m x k] @ B[k x n], with
 * the plain and tiled variants agreeing.
 */

#include <matrix/mult.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace matrix_multiply_edge_test {

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

// Regression: these threw "invalid configuration argument".
TEST(MatrixMultiplyEdgeTest, EmptyExtentsAreNoOps) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The buffers are never read when the guard fires, so a one-element scratch array
    // satisfies the signature.
    float a[1] = {1.0f};
    float b[1] = {1.0f};
    float c[1] = {0.0f};

    (void)cudaGetLastError();

    EXPECT_NO_THROW(multiplyMatricesNaive<float>(a, b, c, 0, 1, 1));
    EXPECT_NO_THROW(multiplyMatricesNaive<float>(a, b, c, 1, 0, 1));
    EXPECT_NO_THROW(multiplyMatricesNaive<float>(a, b, c, 1, 1, 0));
    EXPECT_NO_THROW(multiplyMatricesTiled<float>(a, b, c, 0, 1, 1));
    EXPECT_NO_THROW(multiplyMatricesTiled<float>(a, b, c, 1, 0, 1));

    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "an empty product must not leave a launch error behind";
}

TEST(MatrixMultiplyEdgeTest, NaiveMatchesTheDefinition) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A[2x3] @ B[3x2] = C[2x2].
    const std::vector<float> a = {1, 2, 3,
                                  4, 5, 6};
    const std::vector<float> b = {7, 8,
                                  9, 10,
                                  11, 12};
    std::vector<float> c(4, 0.0f);

    multiplyMatricesNaive<float>(a.data(), b.data(), c.data(), 2, 3, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 1*7+2*9+3*11 = 58;  1*8+2*10+3*12 = 64
    // 4*7+5*9+6*11 = 139; 4*8+5*10+6*12 = 154
    const std::vector<float> expected = {58, 64, 139, 154};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(c[i], expected[i], 1e-4f) << "element " << i;
    }
}

TEST(MatrixMultiplyEdgeTest, TiledAgreesWithNaive) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A size that is not a multiple of the 16x16 tile, so the partial tiles are
    // exercised and any disagreement between the two kernels shows up.
    constexpr int m = 5, k = 7, n = 6;
    std::vector<float> a(static_cast<size_t>(m) * k);
    std::vector<float> b(static_cast<size_t>(k) * n);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i % 11) * 0.5f - 2.0f;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(i % 13) * 0.3f - 1.0f;
    }

    std::vector<float> naive(static_cast<size_t>(m) * n, 0.0f);
    std::vector<float> tiled(static_cast<size_t>(m) * n, 0.0f);

    multiplyMatricesNaive<float>(a.data(), b.data(), naive.data(), m, k, n);
    multiplyMatricesTiled<float>(a.data(), b.data(), tiled.data(), m, k, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (size_t i = 0; i < naive.size(); ++i) {
        EXPECT_NEAR(tiled[i], naive[i], 1e-4f)
            << "the tiled kernel disagrees at element " << i;
    }
}

TEST(MatrixMultiplyEdgeTest, IdentityMatrixLeavesTheOperandUnchanged) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A @ I = A, which pins the operand order: swapping them would give I @ A only when
    // A is square.
    constexpr int m = 3, k = 3;
    const std::vector<float> a = {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9};
    const std::vector<float> identity = {1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1};
    std::vector<float> c(static_cast<size_t>(m) * k, 0.0f);

    multiplyMatricesNaive<float>(a.data(), identity.data(), c.data(), m, k, k);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(c[i], a[i], 1e-4f) << "element " << i;
    }
}

}  // namespace matrix_multiply_edge_test
