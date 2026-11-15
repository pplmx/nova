/**
 * @file hyb_matrix_test.cpp
 * @brief Contracts for the ELL+COO hybrid sparse format
 *
 * SparseMatrixHYB had no test coverage at all. These pin the split it promises
 * (regular rows in ELL, irregular rows in COO), that neither half loses or
 * duplicates an entry, and that the multiply matches a dense reference for
 * every split point the threshold divisor can produce.
 */

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cuda/sparse/hyb_matrix.hpp>
#pragma GCC diagnostic pop

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nova::sparse::test {

namespace {

using CSR = SparseMatrixCSR<double>;

// 4x4 with row nnz counts 4, 1, 4, 1:
//   row 0 -> (0,1) (1,2) (2,3) (3,4)
//   row 1 -> (1,5)
//   row 2 -> (0,6) (1,7) (2,8) (3,9)
//   row 3 -> (2,10)
CSR make_irregular_matrix() {
    std::vector<double> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> row_offsets = {0, 4, 5, 9, 10};
    std::vector<int> col_indices = {0, 1, 2, 3, 1, 0, 1, 2, 3, 2};
    return CSR(values, row_offsets, col_indices, 4, 4);
}

std::vector<double> dense_reference(const CSR& csr, const std::vector<double>& x) {
    std::vector<double> y(static_cast<size_t>(csr.num_rows()), 0.0);
    for (int r = 0; r < csr.num_rows(); ++r) {
        for (int k = csr.row_offsets()[r]; k < csr.row_offsets()[r + 1]; ++k) {
            y[static_cast<size_t>(r)] += csr.values()[k] * x[static_cast<size_t>(csr.col_indices()[k])];
        }
    }
    return y;
}

}  // namespace

TEST(HybMatrixTest, DefaultConstructionIsEmpty) {
    SparseMatrixHYB<double> hyb;

    EXPECT_EQ(hyb.num_rows(), 0);
    EXPECT_EQ(hyb.num_cols(), 0);
    EXPECT_EQ(hyb.nnz(), 0);
    EXPECT_EQ(hyb.ell_row_count(), 0);
    EXPECT_EQ(hyb.coo_row_count(), 0);
}

TEST(HybMatrixTest, FromEmptyCsrStaysEmpty) {
    auto hyb = SparseMatrixHYB<double>::FromCSR(CSR{});

    EXPECT_EQ(hyb.num_rows(), 0);
    EXPECT_EQ(hyb.nnz(), 0);
    EXPECT_EQ(hyb.ell_row_count(), 0);
}

TEST(HybMatrixTest, ZeroNnzCsrKeepsItsShape) {
    CSR csr(std::vector<double>{}, std::vector<int>{0, 0, 0}, std::vector<int>{}, 2, 2);
    auto hyb = SparseMatrixHYB<double>::FromCSR(csr);

    EXPECT_EQ(hyb.num_rows(), 2);
    EXPECT_EQ(hyb.num_cols(), 2);
    EXPECT_EQ(hyb.nnz(), 0);
    EXPECT_EQ(hyb.ell_row_count(), 0);
    EXPECT_EQ(hyb.coo_row_count(), 2);
}

// Regression: threshold_ = max_nnz_per_row / threshold_divisor, an integer
// division the divisor went straight into. FromCSR(csr, 0) therefore raised
// SIGFPE and killed the process instead of reporting the bad argument.
TEST(HybMatrixTest, ZeroThresholdDivisorIsRejected) {
    auto csr = make_irregular_matrix();

    EXPECT_THROW(SparseMatrixHYB<double>::FromCSR(csr, 0), std::invalid_argument);
    EXPECT_THROW(SparseMatrixHYB<double>::FromCSR(csr, -1), std::invalid_argument);
}

// The split must be lossless: every input entry lands in exactly one half.
TEST(HybMatrixTest, SplitPreservesEveryEntryExactlyOnce) {
    auto csr = make_irregular_matrix();
    auto hyb = SparseMatrixHYB<double>::FromCSR(csr);

    EXPECT_EQ(hyb.nnz(), csr.nnz());

    // Reconstruct the matrix from both halves and compare with the input.
    std::vector<std::vector<double>> dense(4, std::vector<double>(4, 0.0));
    // ELL rows are the CSR rows with more than threshold() entries, in row
    // order, so the ELL index of a row is its rank among those rows.
    std::vector<int> ell_rows;
    for (int r = 0; r < csr.num_rows(); ++r) {
        if (hyb.is_ell_row(r)) {
            ell_rows.push_back(r);
        }
    }
    ASSERT_EQ(static_cast<int>(ell_rows.size()), hyb.ell_row_count());

    const int max_nnz = hyb.max_nnz_per_row();
    for (int ell_row = 0; ell_row < hyb.ell_row_count(); ++ell_row) {
        const int row = ell_rows[static_cast<size_t>(ell_row)];
        ASSERT_EQ(hyb.get_ell_row_index(row), ell_row);
        for (int j = 0; j < max_nnz; ++j) {
            const int col = hyb.ell_col_indices()[ell_row * max_nnz + j];
            if (col >= 0) {
                dense[static_cast<size_t>(row)][static_cast<size_t>(col)] +=
                    hyb.ell_values()[ell_row * max_nnz + j];
            }
        }
    }
    for (int i = 0; i < hyb.coo_nnz(); ++i) {
        dense[static_cast<size_t>(hyb.coo_row_indices()[i])]
             [static_cast<size_t>(hyb.coo_col_indices()[i])] += hyb.coo_values()[i];
    }

    for (int r = 0; r < csr.num_rows(); ++r) {
        for (int c = 0; c < csr.num_cols(); ++c) {
            double expected = 0.0;
            for (int k = csr.row_offsets()[r]; k < csr.row_offsets()[r + 1]; ++k) {
                if (csr.col_indices()[k] == c) {
                    expected += csr.values()[k];
                }
            }
            EXPECT_DOUBLE_EQ(dense[static_cast<size_t>(r)][static_cast<size_t>(c)], expected)
                << "row " << r << " col " << c;
        }
    }
}

TEST(HybMatrixTest, MultiplyMatchesDenseReferenceAtEverySplitPoint) {
    auto csr = make_irregular_matrix();
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const auto expected = dense_reference(csr, x);

    // divisor 1 puts everything in ELL, 2/3 split, 100 puts everything in ELL
    // again (threshold 0), so this walks every branch of the split rule.
    for (int divisor : {1, 2, 3, 100}) {
        auto hyb = SparseMatrixHYB<double>::FromCSR(csr, divisor);

        std::vector<double> y(4, -1.0);
        sparse_mv(hyb, x.data(), y.data());

        for (int i = 0; i < 4; ++i) {
            EXPECT_DOUBLE_EQ(y[static_cast<size_t>(i)], expected[static_cast<size_t>(i)])
                << "divisor " << divisor << " row " << i;
        }
    }
}

// A row that ends up in COO must not also be written by the ELL half: the ELL
// loop zero-initialises y only for its own rows, and a double write would show
// up as a doubled entry.
TEST(HybMatrixTest, EllAndCooHalvesAreDisjoint) {
    auto csr = make_irregular_matrix();
    auto hyb = SparseMatrixHYB<double>::FromCSR(csr);

    ASSERT_EQ(hyb.ell_row_count() + hyb.coo_row_count(), hyb.num_rows());

    for (int i = 0; i < hyb.coo_nnz(); ++i) {
        EXPECT_FALSE(hyb.is_ell_row(hyb.coo_row_indices()[i]))
            << "row " << hyb.coo_row_indices()[i] << " is in both halves";
    }
    for (int r = 0; r < hyb.num_rows(); ++r) {
        if (hyb.is_ell_row(r)) {
            EXPECT_GE(hyb.get_ell_row_index(r), 0);
            EXPECT_LT(hyb.get_ell_row_index(r), hyb.ell_row_count());
        } else {
            EXPECT_EQ(hyb.get_ell_row_index(r), -1);
        }
    }
}

TEST(HybMatrixTest, DiagonalMatrixGoesEntirelyToEll) {
    CSR csr(std::vector<double>{1, 1, 1}, std::vector<int>{0, 1, 2, 3},
            std::vector<int>{0, 1, 2}, 3, 3);
    auto hyb = SparseMatrixHYB<double>::FromCSR(csr, 2);

    EXPECT_EQ(hyb.nnz(), 3);
    EXPECT_EQ(hyb.coo_nnz(), 0);

    const std::vector<double> x = {2.0, 3.0, 4.0};
    std::vector<double> y(3, -1.0);
    sparse_mv(hyb, x.data(), y.data());
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
    EXPECT_DOUBLE_EQ(y[2], 4.0);
}

}  // namespace nova::sparse::test
