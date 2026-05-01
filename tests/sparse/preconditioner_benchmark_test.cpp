#include <gtest/gtest.h>
#include <cuda/sparse/krylov.hpp>
#include <cuda/sparse/preconditioner.hpp>
#include <cuda/sparse/matrix.hpp>
#include <chrono>
#include <vector>
#include <iostream>

namespace nova {
namespace sparse {
namespace test {

class PreconditionerBenchmarkTest : public ::testing::Test {
protected:
    static SparseMatrix<double> create_laplacian_matrix(int n) {
        std::vector<double> values;
        std::vector<int> row_offsets(n + 1, 0);
        std::vector<int> col_indices;

        for (int i = 0; i < n; ++i) {
            row_offsets[i] = static_cast<int>(col_indices.size());

            if (i > 0) {
                values.push_back(-1.0);
                col_indices.push_back(i - 1);
            }

            values.push_back(4.0);
            col_indices.push_back(i);

            if (i < n - 1) {
                values.push_back(-1.0);
                col_indices.push_back(i + 1);
            }
        }
        row_offsets[n] = static_cast<int>(col_indices.size());

        return SparseMatrix<double>::FromHostData(values, row_offsets, col_indices, n, n);
    }
};

TEST_F(PreconditionerBenchmarkTest, CGIterationCountComparison) {
    const int n = 50;
    auto A = create_laplacian_matrix(n);

    std::vector<double> b(n, 1.0);
    std::vector<double> x(n, 0.0);

    ConjugateGradient<double> cg_no_prec;
    auto result_no_prec = cg_no_prec.solve(A, b.data(), x.data());
    int iter_no_prec = result_no_prec.iterations;

    x.assign(n, 0.0);
    ConjugateGradient<double> cg_jacobi;
    auto jacobi = std::make_unique<JacobiPreconditioner<double>>();
    jacobi->setup(A);
    cg_jacobi.set_preconditioner(std::move(jacobi));
    auto result_jacobi = cg_jacobi.solve(A, b.data(), x.data());
    int iter_jacobi = result_jacobi.iterations;

    std::cout << "CG without preconditioner: " << iter_no_prec << " iterations\n";
    std::cout << "CG with Jacobi: " << iter_jacobi << " iterations\n";

    if (iter_no_prec > 0) {
        double reduction = 100.0 * (1.0 - static_cast<double>(iter_jacobi) / iter_no_prec);
        std::cout << "Iteration reduction: " << reduction << "%\n";
    }
}

TEST_F(PreconditionerBenchmarkTest, PreconditionerSetupTime) {
    const int n = 100;
    auto A = create_laplacian_matrix(n);

    auto start = std::chrono::high_resolution_clock::now();
    JacobiPreconditioner<double> jacobi;
    jacobi.setup(A);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Jacobi setup time (n=" << n << "): " << duration.count() << " us\n";
}

TEST_F(PreconditionerBenchmarkTest, ConvergenceOnLargerMatrix) {
    const int n = 100;
    auto A = create_laplacian_matrix(n);

    memory::Buffer<double> d_b(n), d_x(n);
    std::vector<double> x_exact(n, 1.0);
    std::vector<double> h_b(n);

    spmv(A, x_exact.data(), h_b.data());
    d_b.copy_from(h_b.data(), n);

    std::vector<double> x(n, 0.0);

    ConjugateGradient<double> cg;
    auto prec = std::make_unique<JacobiPreconditioner<double>>();
    prec->setup(A);
    cg.set_preconditioner(std::move(prec));

    auto start = std::chrono::high_resolution_clock::now();
    auto result = cg.solve(A, h_b.data(), x.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_TRUE(result.converged);
    std::cout << "CG+Jacobi solve time (n=" << n << "): " << duration.count() << " us\n";
    std::cout << "Final relative residual: " << result.relative_residual << "\n";
}

TEST_F(PreconditionerBenchmarkTest, JacobiWeightedVsStandard) {
    const int n = 50;
    auto A = create_laplacian_matrix(n);

    std::vector<double> b(n, 1.0);

    std::vector<double> x1(n, 0.0);
    ConjugateGradient<double> cg1;
    auto jacobi1 = std::make_unique<JacobiPreconditioner<double>>(1.0);
    jacobi1->setup(A);
    cg1.set_preconditioner(std::move(jacobi1));
    auto result1 = cg1.solve(A, b.data(), x1.data());

    std::vector<double> x2(n, 0.0);
    ConjugateGradient<double> cg2;
    auto jacobi2 = std::make_unique<JacobiPreconditioner<double>>(0.8);
    jacobi2->setup(A);
    cg2.set_preconditioner(std::move(jacobi2));
    auto result2 = cg2.solve(A, b.data(), x2.data());

    std::cout << "Jacobi (omega=1.0): " << result1.iterations << " iterations\n";
    std::cout << "Jacobi (omega=0.8): " << result2.iterations << " iterations\n";
}

}
}
}
