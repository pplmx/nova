#include <gtest/gtest.h>
#include <cuda/sparse/matrix.hpp>
#include <cuda/sparse/krylov.hpp>
#include <cuda/sparse/sparse_matrix.hpp>
#include <cuda/memory/buffer.h>

namespace nova::sparse::test {

template<typename T>
bool approx_equal(T a, T b, T tol = T{1e-4}) {
    return std::abs(a - b) < tol;
}

TEST(IntegrationTest, CSRToSparseMatrixConversion) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> row_offsets = {0, 2, 4};
    std::vector<int> col_indices = {0, 1, 1, 2};

    SparseMatrixCSR<float> csr(values, row_offsets, col_indices, 2, 3);
    EXPECT_EQ(csr.num_rows(), 2);
    EXPECT_EQ(csr.num_cols(), 3);
    EXPECT_EQ(csr.nnz(), 4);

    SparseMatrix<float> gpu_matrix = ToSparseMatrix(csr);

    EXPECT_EQ(gpu_matrix.rows(), 2);
    EXPECT_EQ(gpu_matrix.cols(), 3);
    EXPECT_EQ(gpu_matrix.nnz(), 4);

    std::vector<float> out_values;
    std::vector<int> out_row_offsets, out_col_indices;
    gpu_matrix.copy_to_host(out_values, out_row_offsets, out_col_indices);

    EXPECT_EQ(out_values, values);
    EXPECT_EQ(out_row_offsets, row_offsets);
    EXPECT_EQ(out_col_indices, col_indices);
}

TEST(IntegrationTest, SpmvResultConsistency) {
    std::vector<float> dense = {
        4.0f, 1.0f, 0.0f,
        1.0f, 3.0f, 1.0f,
        0.0f, 1.0f, 2.0f
    };

    auto gpu_matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);
    ASSERT_TRUE(gpu_matrix.has_value());

    std::vector<float> x = {1.0f, 2.0f, 3.0f};

    memory::Buffer<float> d_x(3), d_y(3);
    d_x.copy_from(x.data(), 3);

    spmv(*gpu_matrix, d_x.data(), d_y.data());

    std::vector<float> h_y(3);
    d_y.copy_to(h_y.data(), 3);

    std::vector<float> expected = {
        4.0f * 1.0f + 1.0f * 2.0f + 0.0f * 3.0f,
        1.0f * 1.0f + 3.0f * 2.0f + 1.0f * 3.0f,
        0.0f * 1.0f + 1.0f * 2.0f + 2.0f * 3.0f
    };

    EXPECT_TRUE(approx_equal(h_y[0], expected[0]));
    EXPECT_TRUE(approx_equal(h_y[1], expected[1]));
    EXPECT_TRUE(approx_equal(h_y[2], expected[2]));
}

TEST(IntegrationTest, CGSolverWithConvertedMatrix) {
    std::vector<float> dense = {
        4.0f, 1.0f,
        1.0f, 3.0f
    };

    auto gpu_matrix = SparseMatrix<float>::FromDense(dense.data(), 2, 2);
    ASSERT_TRUE(gpu_matrix.has_value());

    std::vector<float> b = {1.0f, 2.0f};
    std::vector<float> x(2, 0.0f);

    SolverConfig<float> config;
    config.relative_tolerance = 1e-8f;
    config.max_iterations = 100;

    ConjugateGradient<float> solver(config);
    auto result = solver.solve(*gpu_matrix, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 50);

    memory::Buffer<float> d_x(2), d_ax(2);
    d_x.copy_from(x.data(), 2);
    spmv(*gpu_matrix, d_x.data(), d_ax.data());
    std::vector<float> h_ax(2);
    d_ax.copy_to(h_ax.data(), 2);

    EXPECT_TRUE(approx_equal(h_ax[0], b[0], 1e-4f));
    EXPECT_TRUE(approx_equal(h_ax[1], b[1], 1e-4f));
}

TEST(IntegrationTest, LargeSparseMatrixSpMV) {
    const int n = 1000;
    const float sparsity = 0.99f;

    std::vector<float> dense(n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = 4.0f;
        if (i > 0) dense[i * n + i - 1] = -1.0f;
        if (i < n - 1) dense[i * n + i + 1] = -1.0f;
    }

    auto matrix = SparseMatrix<float>::FromDense(dense.data(), n, n, sparsity);
    ASSERT_TRUE(matrix.has_value());

    EXPECT_GT(matrix->nnz(), 0);
    EXPECT_LT(static_cast<float>(matrix->nnz()) / (n * n), 0.1f);

    std::vector<float> x(n, 1.0f);
    memory::Buffer<float> d_x(n), d_y(n);
    d_x.copy_from(x.data(), n);

    spmv(*matrix, d_x.data(), d_y.data());

    std::vector<float> h_y(n);
    d_y.copy_to(h_y.data(), n);

    for (int i = 0; i < n; ++i) {
        float expected = (i > 0 ? -1.0f : 0.0f) + 4.0f + (i < n - 1 ? -1.0f : 0.0f);
        EXPECT_TRUE(approx_equal(h_y[i], expected, 1e-3f));
    }
}

TEST(IntegrationTest, DoublePrecisionSolver) {
    std::vector<double> dense = {
        4.0, 1.0,
        1.0, 3.0
    };

    auto matrix = SparseMatrix<double>::FromDense(dense.data(), 2, 2);
    ASSERT_TRUE(matrix.has_value());

    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x(2, 0.0);

    SolverConfig<double> config;
    config.relative_tolerance = 1e-10;
    config.max_iterations = 100;

    ConjugateGradient<double> solver(config);
    auto result = solver.solve(*matrix, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 50);

    EXPECT_TRUE(approx_equal(x[0], 1.0/11.0, 1e-8));
    EXPECT_TRUE(approx_equal(x[1], 7.0/11.0, 1e-8));
}

TEST(IntegrationTest, BiCGSTABConvergence) {
    const int n = 50;
    std::vector<float> dense(n * n, 0.0f);

    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = 3.0f;
        if (i > 0) dense[i * n + i - 1] = -1.0f;
        if (i < n - 1) dense[i * n + i + 1] = -1.0f;
    }

    auto matrix = SparseMatrix<float>::FromDense(dense.data(), n, n);
    ASSERT_TRUE(matrix.has_value());

    std::vector<float> b(n, 1.0f);
    std::vector<float> x(n, 0.0f);

    SolverConfig<float> config;
    config.relative_tolerance = 1e-8f;
    config.max_iterations = 200;

    BiCGSTAB<float> solver(config);
    auto result = solver.solve(*matrix, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 100);
    EXPECT_LT(result.relative_residual, config.relative_tolerance);

    memory::Buffer<float> d_x(n), d_ax(n);
    d_x.copy_from(x.data(), n);
    spmv(*matrix, d_x.data(), d_ax.data());
    std::vector<float> h_ax(n);
    d_ax.copy_to(h_ax.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(approx_equal(h_ax[i], 1.0f, 1e-3f));
    }
}

TEST(IntegrationTest, GMRESRestart) {
    const int n = 30;
    std::vector<float> dense(n * n, 0.0f);

    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = 3.0f;
        if (i > 0) dense[i * n + i - 1] = -1.0f;
        if (i < n - 1) dense[i * n + i + 1] = -1.0f;
    }

    auto matrix = SparseMatrix<float>::FromDense(dense.data(), n, n);
    ASSERT_TRUE(matrix.has_value());

    std::vector<float> b(n, 1.0f);
    std::vector<float> x(n, 0.0f);

    SolverConfig<float> config;
    config.relative_tolerance = 1e-8f;
    config.max_iterations = 200;

    GMRESGPU<float> solver(config, 10);
    auto result = solver.solve(*matrix, b.data(), x.data());

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 150);

    memory::Buffer<float> d_x(n), d_ax(n);
    d_x.copy_from(x.data(), n);
    spmv(*matrix, d_x.data(), d_ax.data());
    std::vector<float> h_ax(n);
    d_ax.copy_to(h_ax.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(approx_equal(h_ax[i], 1.0f, 1e-3f));
    }
}

}  // namespace nova::sparse::test
