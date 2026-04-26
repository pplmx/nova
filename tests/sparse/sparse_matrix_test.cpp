#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace nova {
namespace sparse {
namespace test {

class SparseMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}

    std::vector<float> create_dense_3x3() {
        return {
            1.0f, 0.0f, 2.0f,
            0.0f, 3.0f, 0.0f,
            4.0f, 0.0f, 5.0f
        };
    }

    std::vector<float> create_dense_with_zero_row() {
        return {
            1.0f, 0.0f, 2.0f,
            0.0f, 0.0f, 0.0f,
            4.0f, 0.0f, 5.0f
        };
    }
};

TEST_F(SparseMatrixTest, FromDenseCreatesCSR) {
    auto dense = create_dense_3x3();
    auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(csr.has_value());
    EXPECT_EQ(csr->num_rows(), 3);
    EXPECT_EQ(csr->num_cols(), 3);
    EXPECT_EQ(csr->nnz(), 4);
}

TEST_F(SparseMatrixTest, CSRStoresCorrectValues) {
    auto dense = create_dense_3x3();
    auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(csr.has_value());

    EXPECT_EQ(csr->values()[0], 1.0f);
    EXPECT_EQ(csr->values()[1], 2.0f);
    EXPECT_EQ(csr->values()[2], 3.0f);
    EXPECT_EQ(csr->values()[3], 4.0f);
    EXPECT_EQ(csr->values()[4], 5.0f);

    EXPECT_EQ(csr->col_indices()[0], 0);
    EXPECT_EQ(csr->col_indices()[1], 2);
    EXPECT_EQ(csr->col_indices()[2], 1);
    EXPECT_EQ(csr->col_indices()[3], 0);
    EXPECT_EQ(csr->col_indices()[4], 2);
}

TEST_F(SparseMatrixTest, FromDenseReturnsNulloptForAllZeros) {
    std::vector<float> zeros(9, 0.0f);
    auto csr = SparseMatrixCSR<float>::FromDense(zeros.data(), 3, 3);

    EXPECT_FALSE(csr.has_value());
}

TEST_F(SparseMatrixTest, ToCSCConvertsCorrectly) {
    auto dense = create_dense_3x3();
    auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(csr.has_value());
    auto csc = SparseMatrixCSC<float>::FromCSR(*csr);

    EXPECT_EQ(csc.num_rows(), 3);
    EXPECT_EQ(csc.num_cols(), 3);
    EXPECT_EQ(csc.nnz(), 5);
}

TEST_F(SparseMatrixTest, SpMVProducesCorrectResult) {
    auto dense = create_dense_3x3();
    auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(csr.has_value());

    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y(3, 0.0f);

    sparse_mv(*csr, x.data(), y.data());

    EXPECT_FLOAT_EQ(y[0], 1.0f * 1.0f + 2.0f * 3.0f);
    EXPECT_FLOAT_EQ(y[1], 3.0f * 2.0f);
    EXPECT_FLOAT_EQ(y[2], 4.0f * 1.0f + 5.0f * 3.0f);
}

TEST_F(SparseMatrixTest, SpMMProducesCorrectResult) {
    auto dense = create_dense_3x3();
    auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(csr.has_value());

    std::vector<float> B = {1.0f, 2.0f, 3.0f,
                            4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f};
    std::vector<float> C(9, 0.0f);

    sparse_mm(*csr, B.data(), C.data(), 3);

    EXPECT_FLOAT_EQ(C[0], 1.0f * 1.0f + 2.0f * 4.0f);
    EXPECT_FLOAT_EQ(C[3], 3.0f * 5.0f);
    EXPECT_FLOAT_EQ(C[6], 4.0f * 1.0f + 5.0f * 7.0f);
}

} // namespace test
} // namespace sparse
} // namespace nova
