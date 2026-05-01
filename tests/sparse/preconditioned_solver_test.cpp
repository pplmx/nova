#include <gtest/gtest.h>
#include <cuda/sparse/krylov.hpp>
#include <cuda/sparse/preconditioner.hpp>
#include <cuda/sparse/matrix.hpp>
#include <vector>
#include <cmath>

namespace nova {
namespace sparse {
namespace test {

class PreconditionedSolverTest : public ::testing::Test {
protected:
    static SparseMatrix<double> create_tridiagonal_matrix(int n) {
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

    static std::vector<double> create_rhs(int n) {
        std::vector<double> b(n, 1.0);
        return b;
    }

    static double compute_relative_error(const std::vector<double>& x, const std::vector<double>& x_exact) {
        double max_error = 0.0;
        for (int i = 0; i < static_cast<int>(x.size()); ++i) {
            max_error = std::max(max_error, std::abs(x[i] - x_exact[i]));
        }
        return max_error;
    }
};

TEST_F(PreconditionedSolverTest, CGWithNullPreconditioner) {
    auto A = create_tridiagonal_matrix(10);
    auto b = create_rhs(10);

    ConjugateGradient<double> cg;
    EXPECT_FALSE(cg.has_preconditioner());

    std::vector<double> x(A.cols(), 0.0);
    auto result = cg.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-6);
}

TEST_F(PreconditionedSolverTest, CGWithJacobiPreconditioner) {
    auto A = create_tridiagonal_matrix(10);
    auto b = create_rhs(10);

    ConjugateGradient<double> cg;
    auto prec = std::make_unique<JacobiPreconditioner<double>>();
    prec->setup(A);
    cg.set_preconditioner(std::move(prec));

    EXPECT_TRUE(cg.has_preconditioner());

    std::vector<double> x(A.cols(), 0.0);
    auto result = cg.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.relative_residual, 1e-6);
}

TEST_F(PreconditionedSolverTest, CGWithoutPreconditionerIterations) {
    auto A = create_tridiagonal_matrix(20);
    auto b = create_rhs(20);

    ConjugateGradient<double> cg_no_prec;
    std::vector<double> x_no_prec(A.cols(), 0.0);
    auto result_no_prec = cg_no_prec.solve(A, b.data(), x_no_prec.data());

    EXPECT_TRUE(result_no_prec.converged);

    ConjugateGradient<double> cg_with_prec;
    auto prec = std::make_unique<JacobiPreconditioner<double>>();
    prec->setup(A);
    cg_with_prec.set_preconditioner(std::move(prec));

    std::vector<double> x_with_prec(A.cols(), 0.0);
    auto result_with_prec = cg_with_prec.solve(A, b.data(), x_with_prec.data());

    EXPECT_TRUE(result_with_prec.converged);

    EXPECT_LE(result_with_prec.iterations, result_no_prec.iterations);
}

TEST_F(PreconditionedSolverTest, CGPreconditionerSolutionQuality) {
    auto A = create_tridiagonal_matrix(10);
    std::vector<double> x_exact(A.cols(), 1.0);

    memory::Buffer<double> d_x(A.cols());
    memory::Buffer<double> d_b(A.cols());
    std::vector<double> h_b(A.cols());

    spmv(A, x_exact.data(), h_b.data());
    d_b.copy_from(h_b.data(), A.cols());

    ConjugateGradient<double> cg;
    auto prec = std::make_unique<JacobiPreconditioner<double>>();
    prec->setup(A);
    cg.set_preconditioner(std::move(prec));

    std::vector<double> x(A.cols(), 0.0);
    auto result = cg.solve(A, h_b.data(), x.data());

    EXPECT_TRUE(result.converged);

    double rel_error = compute_relative_error(x, x_exact);
    EXPECT_LT(rel_error, 1e-6);
}

TEST_F(PreconditionedSolverTest, SolverConfigPreserved) {
    auto A = create_tridiagonal_matrix(5);

    SolverConfig<double> config;
    config.relative_tolerance = 1e-8;
    config.max_iterations = 100;

    ConjugateGradient<double> cg(config);
    auto prec = std::make_unique<JacobiPreconditioner<double>>(0.5);
    prec->setup(A);
    cg.set_preconditioner(std::move(prec));

    std::vector<double> b(A.cols(), 1.0);
    std::vector<double> x(A.cols(), 0.0);

    auto result = cg.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
}

}
}
}
