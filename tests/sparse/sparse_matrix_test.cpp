#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace nova {
    namespace sparse {
        namespace test {

            class SparseMatrixFormatTest : public ::testing::Test {
            protected:
                void SetUp() override {}

                std::vector<float> create_dense_3x3() { return {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f}; }

                std::vector<float> create_dense_with_zero_row() { return {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 5.0f}; }

                std::vector<float> create_dense_varying_density() {
                    return {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f,  3.0f, 0.0f,  0.0f,  0.0f,  4.0f,  5.0f, 6.0f,
                            0.0f, 0.0f, 7.0f, 8.0f, 9.0f, 10.0f, 0.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
                }
            };

            TEST_F(SparseMatrixFormatTest, FromDenseCreatesCSR) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

                ASSERT_TRUE(csr.has_value());
                EXPECT_EQ(csr->num_rows(), 3);
                EXPECT_EQ(csr->num_cols(), 3);
                EXPECT_EQ(csr->nnz(), 5);
            }

            TEST_F(SparseMatrixFormatTest, CSRStoresCorrectValues) {
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

            TEST_F(SparseMatrixFormatTest, FromDenseReturnsNulloptForAllZeros) {
                std::vector<float> zeros(9, 0.0f);
                auto csr = SparseMatrixCSR<float>::FromDense(zeros.data(), 3, 3);

                EXPECT_FALSE(csr.has_value());
            }

            TEST_F(SparseMatrixFormatTest, ToCSCConvertsCorrectly) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

                ASSERT_TRUE(csr.has_value());
                auto csc = SparseMatrixCSC<float>::FromCSR(*csr);

                EXPECT_EQ(csc.num_rows(), 3);
                EXPECT_EQ(csc.num_cols(), 3);
                EXPECT_EQ(csc.nnz(), 5);
            }

            TEST_F(SparseMatrixFormatTest, SpMVProducesCorrectResult) {
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

            TEST_F(SparseMatrixFormatTest, SpMMProducesCorrectResult) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);

                ASSERT_TRUE(csr.has_value());

                std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
                std::vector<float> C(9, 0.0f);

                sparse_mm(*csr, B.data(), C.data(), 3);

                EXPECT_FLOAT_EQ(C[0], 1.0f * 1.0f + 2.0f * 7.0f);
                EXPECT_FLOAT_EQ(C[3], 3.0f * 4.0f);
                EXPECT_FLOAT_EQ(C[6], 4.0f * 1.0f + 5.0f * 7.0f);
            }

            TEST_F(SparseMatrixFormatTest, ToELLConvertsCorrectly) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto ell = SparseMatrixELL<float>::FromCSR(*csr);

                EXPECT_EQ(ell.num_rows(), 3);
                EXPECT_EQ(ell.num_cols(), 3);
                EXPECT_EQ(ell.max_nnz_per_row(), 2);
                EXPECT_EQ(ell.padded_nnz(), 6);
            }

            TEST_F(SparseMatrixFormatTest, ELLPreservesValuesAndIndices) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto ell = SparseMatrixELL<float>::FromCSR(*csr);

                EXPECT_EQ(ell.values()[0], 1.0f);
                EXPECT_EQ(ell.values()[1], 2.0f);
                EXPECT_EQ(ell.col_indices()[0], 0);
                EXPECT_EQ(ell.col_indices()[1], 2);

                EXPECT_EQ(ell.values()[2], 3.0f);
                EXPECT_EQ(ell.col_indices()[2], 1);
                EXPECT_EQ(ell.col_indices()[3], -1);

                EXPECT_EQ(ell.values()[4], 4.0f);
                EXPECT_EQ(ell.values()[5], 5.0f);
                EXPECT_EQ(ell.col_indices()[4], 0);
                EXPECT_EQ(ell.col_indices()[5], 2);
            }

            TEST_F(SparseMatrixFormatTest, ELLSpMVMatchesCSR) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto ell = SparseMatrixELL<float>::FromCSR(*csr);

                std::vector<float> x = {1.0f, 2.0f, 3.0f};
                std::vector<float> y_csr(3, 0.0f);
                std::vector<float> y_ell(3, 0.0f);

                sparse_mv(*csr, x.data(), y_csr.data());
                sparse_mv(ell, x.data(), y_ell.data());

                EXPECT_FLOAT_EQ(y_ell[0], y_csr[0]);
                EXPECT_FLOAT_EQ(y_ell[1], y_csr[1]);
                EXPECT_FLOAT_EQ(y_ell[2], y_csr[2]);
            }

            TEST_F(SparseMatrixFormatTest, ToSELLConvertsCorrectly) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto sell = SparseMatrixSELL<float>::FromCSR(*csr, 2);

                EXPECT_EQ(sell.num_rows(), 3);
                EXPECT_EQ(sell.num_cols(), 3);
                EXPECT_EQ(sell.slice_height(), 2);
                EXPECT_EQ(sell.padded_nnz(), 8);
            }

            TEST_F(SparseMatrixFormatTest, SELLPreservesValuesAndIndices) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto sell = SparseMatrixSELL<float>::FromCSR(*csr, 2);

                int slice0_base = sell.slice_ptr()[0];
                EXPECT_EQ(sell.values()[slice0_base + 0], 1.0f);
                EXPECT_EQ(sell.col_indices()[slice0_base + 0], 0);
                EXPECT_EQ(sell.values()[slice0_base + 1], 2.0f);
                EXPECT_EQ(sell.col_indices()[slice0_base + 1], 2);

                EXPECT_EQ(sell.values()[slice0_base + 2], 3.0f);
                EXPECT_EQ(sell.col_indices()[slice0_base + 2], 1);

                int slice1_base = sell.slice_ptr()[1];
                EXPECT_EQ(sell.values()[slice1_base + 0], 4.0f);
                EXPECT_EQ(sell.col_indices()[slice1_base + 0], 0);
                EXPECT_EQ(sell.values()[slice1_base + 1], 5.0f);
                EXPECT_EQ(sell.col_indices()[slice1_base + 1], 2);
            }

            TEST_F(SparseMatrixFormatTest, SELLDefaultSliceHeight) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto sell = SparseMatrixSELL<float>::FromCSR(*csr);

                EXPECT_EQ(sell.slice_height(), 32);
            }

            TEST_F(SparseMatrixFormatTest, SELLSpMVMatchesCSR) {
                auto dense = create_dense_3x3();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 3, 3);
                ASSERT_TRUE(csr.has_value());

                auto sell = SparseMatrixSELL<float>::FromCSR(*csr, 2);

                std::vector<float> x = {1.0f, 2.0f, 3.0f};
                std::vector<float> y_csr(3, 0.0f);
                std::vector<float> y_sell(3, 0.0f);

                sparse_mv(*csr, x.data(), y_csr.data());
                sparse_mv(sell, x.data(), y_sell.data());

                EXPECT_FLOAT_EQ(y_sell[0], y_csr[0]);
                EXPECT_FLOAT_EQ(y_sell[1], y_csr[1]);
                EXPECT_FLOAT_EQ(y_sell[2], y_csr[2]);
            }

            TEST_F(SparseMatrixFormatTest, ELLEdgeCaseVaryingRowDensity) {
                auto dense = create_dense_varying_density();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 5, 5);
                ASSERT_TRUE(csr.has_value());

                auto ell = SparseMatrixELL<float>::FromCSR(*csr);

                EXPECT_EQ(ell.max_nnz_per_row(), 5);
                EXPECT_EQ(ell.padded_nnz(), 25);
            }

            TEST_F(SparseMatrixFormatTest, SELLSpMVVaryingDensityMatchesCSR) {
                auto dense = create_dense_varying_density();
                auto csr = SparseMatrixCSR<float>::FromDense(dense.data(), 5, 5);
                ASSERT_TRUE(csr.has_value());

                auto sell = SparseMatrixSELL<float>::FromCSR(*csr, 2);

                std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                std::vector<float> y_csr(5, 0.0f);
                std::vector<float> y_sell(5, 0.0f);

                sparse_mv(*csr, x.data(), y_csr.data());
                sparse_mv(sell, x.data(), y_sell.data());

                for (int i = 0; i < 5; ++i) {
                    EXPECT_FLOAT_EQ(y_sell[i], y_csr[i]) << "Mismatch at row " << i;
                }
            }

            TEST_F(SparseMatrixFormatTest, ELLEdgeCaseSingleRow) {
                std::vector<float> single_row = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f};
                auto csr = SparseMatrixCSR<float>::FromDense(single_row.data(), 1, 5);
                ASSERT_TRUE(csr.has_value());

                auto ell = SparseMatrixELL<float>::FromCSR(*csr);

                EXPECT_EQ(ell.num_rows(), 1);
                EXPECT_EQ(ell.max_nnz_per_row(), 2);
                EXPECT_EQ(ell.padded_nnz(), 2);
            }

            TEST_F(SparseMatrixFormatTest, SELLEdgeCaseSingleRow) {
                std::vector<float> single_row = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f};
                auto csr = SparseMatrixCSR<float>::FromDense(single_row.data(), 1, 5);
                ASSERT_TRUE(csr.has_value());

                auto sell = SparseMatrixSELL<float>::FromCSR(*csr, 2);

                EXPECT_EQ(sell.num_rows(), 1);
                EXPECT_EQ(sell.slice_height(), 2);
                EXPECT_EQ(sell.slice_ptr()[0], 0);
                EXPECT_EQ(sell.slice_ptr()[1], 4);
            }

        }  // namespace test
    }  // namespace sparse
}  // namespace nova
