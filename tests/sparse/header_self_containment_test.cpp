/**
 * @file header_self_containment_test.cpp
 * @brief Each sparse header must compile on its own
 *
 * Regression: solver_workspace.hpp named SolverError in TimedSolverResult but
 * never included its definition. Every existing translation unit happened to
 * pull in krylov.hpp first, so the breakage was invisible — until a caller
 * wanted the workspace or diagnostics types alone, which failed to compile.
 *
 * The includes below are deliberately in dependency-hostile order and there is
 * nothing else in this file: if any header stops being self-contained, this
 * file stops compiling.
 */

#include <cuda/sparse/solver_workspace.hpp>

#include <gtest/gtest.h>

#include <string>

namespace nova::sparse::test {

TEST(SparseHeaderTest, SolverWorkspaceIsSelfContained) {
    SolverWorkspace<double> workspace(4);
    EXPECT_EQ(workspace.size(), 4);
    EXPECT_EQ(static_cast<int>(workspace.r().size()), 4);

    workspace.reset();
    EXPECT_DOUBLE_EQ(workspace.r()[0], 0.0);

    TimedSolverResult<double> result;
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.error_code, SolverError::SUCCESS);
}

TEST(SparseHeaderTest, WorkspaceResizeReallocsEveryVector) {
    SolverWorkspace<double> workspace(2);
    workspace.resize(5);

    EXPECT_EQ(workspace.size(), 5);
    EXPECT_EQ(static_cast<int>(workspace.r().size()), 5);
    EXPECT_EQ(static_cast<int>(workspace.p().size()), 5);
    EXPECT_EQ(static_cast<int>(workspace.Ap().size()), 5);
    EXPECT_EQ(static_cast<int>(workspace.r_tilde().size()), 5);
    EXPECT_EQ(static_cast<int>(workspace.p_hat().size()), 5);
    EXPECT_EQ(static_cast<int>(workspace.s().size()), 5);
    EXPECT_EQ(static_cast<int>(workspace.t().size()), 5);
}

TEST(SparseHeaderTest, SolverDiagnosticsConvergenceRate) {
    SolverDiagnostics diagnostics;
    diagnostics.compute_convergence_rate({1.0, 0.5, 0.25, 0.125});
    EXPECT_DOUBLE_EQ(diagnostics.convergence_rate, 0.5);

    // Fewer than two samples carries no rate information: report zero rather
    // than dividing by an empty interval.
    SolverDiagnostics empty;
    empty.compute_convergence_rate({});
    EXPECT_DOUBLE_EQ(empty.convergence_rate, 0.0);

    SolverDiagnostics single;
    single.compute_convergence_rate({3.0});
    EXPECT_DOUBLE_EQ(single.convergence_rate, 0.0);

    EXPECT_NE(diagnostics.to_string().find("Convergence rate"), std::string::npos);
}

}  // namespace nova::sparse::test
