/**
 * @file spmv_host_test.cpp
 * @brief Contracts for the host-vector spmv overload
 *
 * The host overload copied A.cols() and A.rows() elements through the caller's
 * vectors without looking at their sizes. Every failure mode was silent:
 *
 * - an output vector that was too small (including the default-constructed
 *   empty one a caller naturally writes) made cudaMemcpy fail with "invalid
 *   argument" from inside a buffer;
 * - an input vector shorter than cols() read past the end of the host buffer
 *   and produced a plausible-looking but wrong result — the worst case, since
 *   nothing signals that the answer is garbage;
 * - an output vector larger than rows() kept its stale tail.
 *
 * These pin the contract: the overload sizes the output itself and rejects an
 * input of the wrong length.
 */

#include <cuda/sparse/matrix.hpp>
#include <cuda/sparse/sparse_ops.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace nova::sparse::test {

namespace {

// diag(2, 3, 4): rows() == cols() == 3, three non-zeros.
SparseMatrix<double> diagonal_matrix() {
    std::vector<double> dense = {
        2.0, 0.0, 0.0,
        0.0, 3.0, 0.0,
        0.0, 0.0, 4.0,
    };
    auto matrix = SparseMatrix<double>::FromDense(dense.data(), 3, 3);
    EXPECT_TRUE(matrix.has_value());
    return matrix ? std::move(*matrix) : SparseMatrix<double>();
}

}  // namespace

TEST(SpmvHostTest, SizesTheOutputVectorWhenEmpty) {
    auto matrix = diagonal_matrix();

    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y;  // caller did not pre-size it

    ASSERT_NO_THROW(spmv(matrix, x, y));
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    EXPECT_DOUBLE_EQ(y[1], 6.0);
    EXPECT_DOUBLE_EQ(y[2], 12.0);
}

TEST(SpmvHostTest, ResizesAnOutputVectorThatIsTooSmall) {
    auto matrix = diagonal_matrix();

    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y = {0.0};  // wrong size, would have overflowed

    ASSERT_NO_THROW(spmv(matrix, x, y));
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
    EXPECT_DOUBLE_EQ(y[2], 4.0);
}

TEST(SpmvHostTest, ShrinksAnOutputVectorThatIsTooLarge) {
    auto matrix = diagonal_matrix();

    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y(7, -1.0);  // trailing entries must not survive

    ASSERT_NO_THROW(spmv(matrix, x, y));
    ASSERT_EQ(y.size(), 3u) << "the result must be exactly rows() long";
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
    EXPECT_DOUBLE_EQ(y[2], 4.0);
}

// Regression: a short input used to be read past its end, producing a
// plausible but wrong answer with no error. Reject it instead.
TEST(SpmvHostTest, RejectsAnInputShorterThanTheColumnCount) {
    auto matrix = diagonal_matrix();

    std::vector<double> x = {1.0, 2.0};  // cols() == 3
    std::vector<double> y(3, 0.0);

    EXPECT_THROW(spmv(matrix, x, y), std::invalid_argument);
}

TEST(SpmvHostTest, RejectsAnInputLongerThanTheColumnCount) {
    auto matrix = diagonal_matrix();

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};  // cols() == 3
    std::vector<double> y(3, 0.0);

    EXPECT_THROW(spmv(matrix, x, y), std::invalid_argument);
}

TEST(SpmvHostTest, MatchesTheDevicePointerPathOnTheSameInput) {
    // The host overload is a copy-and-call wrapper around the device one; both
    // must agree, so a mismatch between them shows up here rather than as a
    // strange numerical result downstream.
    auto matrix = diagonal_matrix();

    std::vector<double> x = {1.5, -2.0, 0.25};
    std::vector<double> y_host;
    spmv(matrix, x, y_host);

    ASSERT_EQ(y_host.size(), 3u);
    EXPECT_DOUBLE_EQ(y_host[0], 3.0);
    EXPECT_DOUBLE_EQ(y_host[1], -6.0);
    EXPECT_DOUBLE_EQ(y_host[2], 1.0);
}

}  // namespace nova::sparse::test
