#include <gtest/gtest.h>
#include <cuda/sparse/preconditioner.hpp>
#include <cuda/sparse/matrix.hpp>
#include <vector>
#include <cmath>

namespace nova {
namespace sparse {
namespace test {

class JacobiPreconditionerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    static std::vector<double> generate_laplacian(int n) {
        std::vector<double> values;
        std::vector<int> row_offsets(n + 1, 0);
        std::vector<int> col_indices;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    values.push_back(4.0);
                    col_indices.push_back(j);
                } else if (std::abs(i - j) == 1) {
                    values.push_back(-1.0);
                    col_indices.push_back(j);
                }
            }
            row_offsets[i + 1] = static_cast<int>(col_indices.size());
        }

        return values;
    }

    static SparseMatrix<double> create_laplacian_matrix(int n) {
        auto values = generate_laplacian(n);
        std::vector<int> row_offsets(n + 1);
        std::vector<int> col_indices;

        for (int i = 0; i < n; ++i) {
            row_offsets[i] = 0;
            for (int j = 0; j < n; ++j) {
                if (i == j || std::abs(i - j) == 1) {
                    col_indices.push_back(j);
                }
            }
            row_offsets[i + 1] = static_cast<int>(col_indices.size());
        }

        col_indices.clear();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    col_indices.push_back(j);
                } else if (std::abs(i - j) == 1) {
                    col_indices.push_back(j);
                }
            }
        }

        auto matrix = SparseMatrix<double>::FromHostData(
            values, row_offsets, col_indices, n, n);
        return matrix;
    }
};

TEST_F(JacobiPreconditionerTest, ConstructorDefaultOmega) {
    JacobiPreconditioner<double> prec;
    EXPECT_DOUBLE_EQ(prec.omega(), 1.0);
}

TEST_F(JacobiPreconditionerTest, ConstructorCustomOmega) {
    JacobiPreconditioner<double> prec(0.5);
    EXPECT_DOUBLE_EQ(prec.omega(), 0.5);

    JacobiPreconditioner<double> prec2(1.5);
    EXPECT_DOUBLE_EQ(prec2.omega(), 1.5);

    JacobiPreconditioner<double> prec3(2.0);
    EXPECT_DOUBLE_EQ(prec3.omega(), 2.0);
}

TEST_F(JacobiPreconditionerTest, ConstructorInvalidOmegaZero) {
    EXPECT_THROW(JacobiPreconditioner<double>(0.0), PreconditionerError);
}

TEST_F(JacobiPreconditionerTest, ConstructorInvalidOmegaNegative) {
    EXPECT_THROW(JacobiPreconditioner<double>(-1.0), PreconditionerError);
}

TEST_F(JacobiPreconditionerTest, ConstructorInvalidOmegaGreaterThanTwo) {
    EXPECT_THROW(JacobiPreconditioner<double>(2.5), PreconditionerError);
}

TEST_F(JacobiPreconditionerTest, SetupDiagonalMatrix) {
    std::vector<double> values = {2.0, 3.0, 4.0};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 3, 3);

    JacobiPreconditioner<double> prec;
    EXPECT_NO_THROW(prec.setup(matrix));
}

TEST_F(JacobiPreconditionerTest, SetupLaplacianMatrix) {
    auto matrix = create_laplacian_matrix(5);
    JacobiPreconditioner<double> prec;
    EXPECT_NO_THROW(prec.setup(matrix));
}

TEST_F(JacobiPreconditionerTest, SetupZeroDiagonal) {
    std::vector<double> values = {2.0, 0.0, 4.0};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 3, 3);

    JacobiPreconditioner<double> prec;
    EXPECT_THROW(prec.setup(matrix), PreconditionerError);
}

TEST_F(JacobiPreconditionerTest, ApplyIdentityMatrix) {
    std::vector<double> values = {1.0, 1.0, 1.0};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 3, 3);

    JacobiPreconditioner<double> prec;
    prec.setup(matrix);

    std::vector<double> in = {1.0, 2.0, 3.0};
    std::vector<double> out(3);

    prec.apply(in.data(), out.data());

    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 2.0);
    EXPECT_DOUBLE_EQ(out[2], 3.0);
}

TEST_F(JacobiPreconditionerTest, ApplyWeightedJacobi) {
    std::vector<double> values = {2.0, 2.0, 2.0};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 3, 3);

    JacobiPreconditioner<double> prec(0.5);
    prec.setup(matrix);

    std::vector<double> in = {1.0, 2.0, 3.0};
    std::vector<double> out(3);

    prec.apply(in.data(), out.data());

    EXPECT_DOUBLE_EQ(out[0], 0.25);
    EXPECT_DOUBLE_EQ(out[1], 0.5);
    EXPECT_DOUBLE_EQ(out[2], 0.75);
}

TEST_F(JacobiPreconditionerTest, ApplyBufferInterface) {
    std::vector<double> values = {2.0, 3.0, 4.0};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 3, 3);

    JacobiPreconditioner<double> prec;
    prec.setup(matrix);

    memory::Buffer<double> in(3);
    memory::Buffer<double> out(3);

    std::vector<double> h_in = {1.0, 2.0, 3.0};
    in.copy_from(h_in.data(), 3);

    prec.apply(in, out);

    std::vector<double> h_out(3);
    out.copy_to(h_out.data(), 3);

    EXPECT_DOUBLE_EQ(h_out[0], 0.5);
    EXPECT_DOUBLE_EQ(h_out[1], 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(h_out[2], 0.75);
}

TEST_F(JacobiPreconditionerTest, ApplyOnLaplacian) {
    auto matrix = create_laplacian_matrix(5);
    JacobiPreconditioner<double> prec;
    prec.setup(matrix);

    std::vector<double> in = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> out(5);

    prec.apply(in.data(), out.data());

    EXPECT_DOUBLE_EQ(out[0], 0.25);
    EXPECT_DOUBLE_EQ(out[1], 0.25);
    EXPECT_DOUBLE_EQ(out[2], 0.25);
    EXPECT_DOUBLE_EQ(out[3], 0.25);
    EXPECT_DOUBLE_EQ(out[4], 0.25);
}

TEST_F(JacobiPreconditionerTest, MultipleSetupCalls) {
    std::vector<double> values = {2.0, 3.0, 4.0};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    auto matrix = SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, 3, 3);

    JacobiPreconditioner<double> prec;
    EXPECT_NO_THROW(prec.setup(matrix));

    std::vector<double> values2 = {5.0, 6.0, 7.0};
    auto matrix2 = SparseMatrix<double>::FromHostData(values2, row_offsets, col_indices, 3, 3);
    EXPECT_NO_THROW(prec.setup(matrix2));
}

}
}
}
