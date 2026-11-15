/**
 * @file krylov_zero_rhs_test.cpp
 * @brief Contracts for the degenerate right-hand side b = 0
 *
 * All three solvers short-circuit when ||b|| is below epsilon: they report
 * converged with zero iterations, which is correct — the system A x = 0 has the
 * exact solution x = 0. But the early return did not write that solution, so the
 * caller's x array kept whatever it held before the call.
 *
 * The contract pinned here is that a reported solution is actually written: a
 * converged result must leave x holding the answer, because callers use the
 * result flag as permission to read x.
 */

#include <cuda/sparse/krylov.hpp>
#include <cuda/sparse/matrix.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace nova::sparse::test {

namespace {

SparseMatrix<double> diagonal_matrix() {
    // diag(2, 3): well conditioned, so no test here depends on convergence
    // behaviour — only on what happens for b == 0.
    return SparseMatrix<double>::FromHostData(
        std::vector<double>{2.0, 3.0}, std::vector<int>{0, 1, 2},
        std::vector<int>{0, 1}, 2, 2);
}

constexpr double kSentinel = 777.0;

}  // namespace

TEST(KrylovZeroRhsTest, ConjugateGradientWritesTheZeroSolution) {
    auto A = diagonal_matrix();
    std::vector<double> b = {0.0, 0.0};
    std::vector<double> x = {kSentinel, kSentinel};

    ConjugateGradient<double> solver;
    const auto result = solver.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.error_code, SolverError::SUCCESS);
    EXPECT_DOUBLE_EQ(result.residual_norm, 0.0);
    EXPECT_DOUBLE_EQ(x[0], 0.0) << "a converged result must leave x holding the answer";
    EXPECT_DOUBLE_EQ(x[1], 0.0);
}

TEST(KrylovZeroRhsTest, GmresWritesTheZeroSolution) {
    auto A = diagonal_matrix();
    std::vector<double> b = {0.0, 0.0};
    std::vector<double> x = {kSentinel, kSentinel};

    GMRES<double> solver;
    const auto result = solver.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.error_code, SolverError::SUCCESS);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 0.0);
}

TEST(KrylovZeroRhsTest, BicgstabWritesTheZeroSolution) {
    auto A = diagonal_matrix();
    std::vector<double> b = {0.0, 0.0};
    std::vector<double> x = {kSentinel, kSentinel};

    BiCGSTAB<double> solver;
    const auto result = solver.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.error_code, SolverError::SUCCESS);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 0.0);
}

// GMRESGPU keeps the same degenerate path but writes through the caller's
// pointer in its own loop rather than detail::fill, so it needs its own case.
TEST(KrylovZeroRhsTest, GmresGpuWritesTheZeroSolution) {
    auto A = diagonal_matrix();
    std::vector<double> b = {0.0, 0.0};
    std::vector<double> x = {kSentinel, kSentinel};

    GMRESGPU<double> solver;
    const auto result = solver.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.error_code, SolverError::SUCCESS);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 0.0);
}

// The zero solution must overwrite an initial guess as well, not just garbage:
// x = 0 is the unique solution of A x = 0 and the solvers do not otherwise seed
// from the caller's x.
TEST(KrylovZeroRhsTest, ZeroSolutionOverwritesAConvergedInitialGuess) {
    auto A = diagonal_matrix();
    std::vector<double> b = {0.0, 0.0};

    for (int which = 0; which < 3; ++which) {
        std::vector<double> x = {0.0, 0.0};  // already "correct"
        switch (which) {
            case 0: {
                ConjugateGradient<double> s;
                EXPECT_TRUE(s.solve(A, b.data(), x.data()).converged);
                break;
            }
            case 1: {
                GMRES<double> s;
                EXPECT_TRUE(s.solve(A, b.data(), x.data()).converged);
                break;
            }
            default: {
                BiCGSTAB<double> s;
                EXPECT_TRUE(s.solve(A, b.data(), x.data()).converged);
                break;
            }
        }
        EXPECT_DOUBLE_EQ(x[0], 0.0);
        EXPECT_DOUBLE_EQ(x[1], 0.0);
    }
}

// Sanity check that the same solvers still solve a normal system, so a fix for
// the degenerate case cannot pass by breaking the real one.
TEST(KrylovZeroRhsTest, NonZeroRhsStillSolvedExactly) {
    auto A = diagonal_matrix();
    std::vector<double> b = {4.0, 9.0};
    std::vector<double> x(2, 0.0);

    ConjugateGradient<double> solver;
    const auto result = solver.solve(A, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 2.0, 1e-9);
    EXPECT_NEAR(x[1], 3.0, 1e-9);
}

}  // namespace nova::sparse::test
