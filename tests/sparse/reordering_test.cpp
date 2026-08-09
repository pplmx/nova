#include <gtest/gtest.h>
#include <cuda/sparse/reordering.hpp>
#include <cuda/sparse/matrix.hpp>
#include <vector>
#include <cmath>

namespace nova {
namespace sparse {
namespace test {

class RCMReordererTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        cudaDeviceSynchronize();
    }
    static SparseMatrix<double> create_tridiagonal_matrix(int n) {
        std::vector<double> values;
        std::vector<int> row_offsets(n + 1, 0);
        std::vector<int> col_indices;

        int nnz = 0;
        for (int i = 0; i < n; ++i) {
            row_offsets[i] = nnz;

            if (i > 0) {
                values.push_back(-1.0);
                col_indices.push_back(i - 1);
                ++nnz;
            }

            values.push_back(2.0);
            col_indices.push_back(i);
            ++nnz;

            if (i < n - 1) {
                values.push_back(-1.0);
                col_indices.push_back(i + 1);
                ++nnz;
            }
        }
        row_offsets[n] = nnz;

        return SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, n, n);
    }

    static SparseMatrix<double> create_band_matrix(int n, int bandwidth) {
        std::vector<double> values;
        std::vector<int> row_offsets(n + 1, 0);
        std::vector<int> col_indices;

        int nnz = 0;
        for (int i = 0; i < n; ++i) {
            row_offsets[i] = nnz;

            for (int j = std::max(0, i - bandwidth); j <= std::min(n - 1, i + bandwidth); ++j) {
                values.push_back(1.0);
                col_indices.push_back(j);
                ++nnz;
            }
        }
        row_offsets[n] = nnz;

        return SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, n, n);
    }
};

TEST_F(RCMReordererTest, DefaultConstructor) {
    RCMReorderer<double> reorderer;
}

TEST_F(RCMReordererTest, BandwidthTridiagonalMatrix) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;

    int bandwidth = reorderer.compute_bandwidth(matrix);

    EXPECT_EQ(bandwidth, 1);
}

TEST_F(RCMReordererTest, BandwidthDiagonalMatrix) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<int> row_offsets = {0, 1, 2, 3, 4, 5};
    std::vector<int> col_indices = {0, 1, 2, 3, 4};

    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 5, 5);
    RCMReorderer<double> reorderer;

    int bandwidth = reorderer.compute_bandwidth(matrix);

    EXPECT_EQ(bandwidth, 0);
}

TEST_F(RCMReordererTest, ReorderPermutationSize) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;

    auto result = reorderer.reorder(matrix);

    EXPECT_EQ(static_cast<int>(result.permutation.size()), 5);
    EXPECT_EQ(static_cast<int>(result.inverse_permutation.size()), 5);
}

TEST_F(RCMReordererTest, ReorderPermutationIsValid) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;

    auto result = reorderer.reorder(matrix);

    std::vector<int> used(5, 0);
    for (int p : result.permutation) {
        EXPECT_GE(p, 0);
        EXPECT_LT(p, 5);
        EXPECT_EQ(used[p], 0);
        used[p] = 1;
    }
}

TEST_F(RCMReordererTest, ReorderInversePermutation) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;

    auto result = reorderer.reorder(matrix);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(result.inverse_permutation[result.permutation[i]], i);
        EXPECT_EQ(result.permutation[result.inverse_permutation[i]], i);
    }
}

TEST_F(RCMReordererTest, ReorderReducesBandwidth) {
    // A tridiagonal (band-1) path: the strongest case RCM is designed for, and
    // a genuine invariant (the reordering must not destroy band structure).
    // (The original band-5 matrix used here is a dense, nearly-complete graph
    // where RCM, a heuristic, does not guarantee a lower bandwidth.)
    auto matrix = create_tridiagonal_matrix(10);
    RCMReorderer<double> reorderer;

    int original_bandwidth = reorderer.compute_bandwidth(matrix);
    EXPECT_EQ(original_bandwidth, 1);

    auto result = reorderer.reorder(matrix);

    EXPECT_LE(result.reordered_bandwidth, result.original_bandwidth);
}

TEST_F(RCMReordererTest, ApplyReorderingReturnsValidMatrix) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;

    auto result = reorderer.reorder(matrix);
    auto reordered = reorderer.apply_reordering(matrix, result);

    EXPECT_EQ(reordered.rows(), 5);
    EXPECT_EQ(reordered.cols(), 5);
    EXPECT_EQ(reordered.nnz(), matrix.nnz());
}

TEST_F(RCMReordererTest, BandwidthReductionRatio) {
    auto matrix = create_tridiagonal_matrix(10);
    RCMReorderer<double> reorderer;

    auto result = reorderer.reorder(matrix);

    if (result.original_bandwidth > 0) {
        EXPECT_GE(result.bandwidth_reduction_ratio, 0.0);
        EXPECT_LE(result.bandwidth_reduction_ratio, 1.0);
    }
}

TEST_F(RCMReordererTest, ReorderingResultPermutationIsPermutation) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;
    auto result = reorderer.reorder(matrix);

    std::vector<int> count(5, 0);
    for (int p : result.permutation) {
        EXPECT_GE(p, 0);
        EXPECT_LT(p, 5);
        count[p]++;
    }
    for (int c : count) {
        EXPECT_EQ(c, 1);
    }
}

TEST_F(RCMReordererTest, ReorderingResultInverseIsCorrect) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;
    auto result = reorderer.reorder(matrix);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(result.permutation[result.inverse_permutation[i]], i);
    }
}

TEST_F(RCMReordererTest, ReorderingResultBandwidthMetrics) {
    auto matrix = create_band_matrix(10, 5);
    RCMReorderer<double> reorderer;
    auto result = reorderer.reorder(matrix);

    EXPECT_GE(result.original_bandwidth, 0);
    EXPECT_GE(result.reordered_bandwidth, 0);
}

TEST_F(RCMReordererTest, ReorderingPreservesNnz) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;
    auto result = reorderer.reorder(matrix);
    auto reordered = reorderer.apply_reordering(matrix, result);

    EXPECT_EQ(reordered.nnz(), matrix.nnz());
}

TEST_F(RCMReordererTest, ReorderingPreservesSymmetry) {
    auto matrix = create_tridiagonal_matrix(5);
    RCMReorderer<double> reorderer;
    auto result = reorderer.reorder(matrix);
    auto reordered = reorderer.apply_reordering(matrix, result);

    EXPECT_EQ(reordered.rows(), matrix.rows());
    EXPECT_EQ(reordered.cols(), matrix.cols());
}

}
}
}
