#include <gtest/gtest.h>
#include <cuda/sparse/preconditioner.hpp>
#include <cuda/sparse/matrix.hpp>
#include <vector>

namespace nova {
namespace sparse {
namespace test {

class ILUPreconditionerTest : public ::testing::Test {
protected:
    static SparseMatrix<double> create_spd_matrix(int n) {
        std::vector<double> values;
        std::vector<int> row_offsets(n + 1, 0);
        std::vector<int> col_indices;

        for (int i = 0; i < n; ++i) {
            row_offsets[i] = static_cast<int>(col_indices.size());

            if (i > 0) {
                values.push_back(-1.0);
                col_indices.push_back(i - 1);
            }

            values.push_back(2.0);
            col_indices.push_back(i);

            if (i < n - 1) {
                values.push_back(-1.0);
                col_indices.push_back(i + 1);
            }
        }
        row_offsets[n] = static_cast<int>(col_indices.size());

        return SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, n, n);
    }

    static SparseMatrix<double> create_diagonally_dominant_matrix(int n) {
        std::vector<double> values;
        std::vector<int> row_offsets(n + 1, 0);
        std::vector<int> col_indices;

        for (int i = 0; i < n; ++i) {
            row_offsets[i] = static_cast<int>(col_indices.size());

            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    values.push_back(4.0 + static_cast<double>(i) * 0.1);
                    col_indices.push_back(j);
                } else if (std::abs(i - j) == 1) {
                    values.push_back(-1.0);
                    col_indices.push_back(j);
                }
            }
        }
        row_offsets[n] = static_cast<int>(col_indices.size());

        return SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, n, n);
    }
};

TEST_F(ILUPreconditionerTest, DefaultConstructor) {
    ILUPreconditioner<double> prec;
}

// ILU is not implemented: setup and both apply overloads fail fast with a
// PreconditionerError instead of silently pretending to factorize/solve (the
// old stub recorded n_ and returned the output buffer unchanged).
TEST_F(ILUPreconditionerTest, SetupFailsFastNotImplemented) {
    auto matrix = create_spd_matrix(5);
    ILUPreconditioner<double> prec;

    EXPECT_THROW(prec.setup(matrix), PreconditionerError);
}

TEST_F(ILUPreconditionerTest, SetupSmallMatrixFailsFast) {
    auto matrix = create_spd_matrix(2);
    ILUPreconditioner<double> prec;

    EXPECT_THROW(prec.setup(matrix), PreconditionerError);
}

TEST_F(ILUPreconditionerTest, ApplyFailsFastNotImplemented) {
    ILUPreconditioner<double> prec;

    std::vector<double> in = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> out(5);

    EXPECT_THROW(prec.apply(in.data(), out.data()), PreconditionerError);
}

TEST_F(ILUPreconditionerTest, BufferApplyFailsFastNotImplemented) {
    ILUPreconditioner<double> prec;

    memory::Buffer<double> in(5);
    memory::Buffer<double> out(5);

    EXPECT_THROW(prec.apply(in, out), PreconditionerError);
}

}
}
}
