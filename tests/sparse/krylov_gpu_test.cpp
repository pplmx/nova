#include <gtest/gtest.h>
#include <cuda/sparse/krylov.hpp>
#include <cuda/sparse/matrix.hpp>
#include <cuda/memory/buffer.h>

namespace nova::sparse::test {

    template <typename T>
    bool approx_equal(T a, T b, T tol = T {1e-4}) {
        return std::abs(a - b) < tol;
    }

    template <typename T>
    class KrylovGPUTest : public ::testing::Test {};

    using KrylovTypes = ::testing::Types<double>;
    TYPED_TEST_SUITE(KrylovGPUTest, KrylovTypes);

    TYPED_TEST(KrylovGPUTest, CGTrivial) {
        using T = TypeParam;
        std::vector<T> dense = {T {4}, T {1}, T {1}, T {3}};
        auto matrix = SparseMatrix<T>::FromDense(dense.data(), 2, 2);
        ASSERT_TRUE(matrix.has_value());

        std::vector<T> b = {T {1}, T {2}};
        std::vector<T> x(2, T {0});

        SolverConfig<T> config;
        config.relative_tolerance = T {1e-8};
        config.max_iterations = 100;
        config.verbose = false;

        ConjugateGradient<T> solver(config);
        auto result = solver.solve(*matrix, b.data(), x.data());

        EXPECT_TRUE(result.converged);
        EXPECT_LT(result.iterations, 50);

        T expected_x0 = T {1} / T {11};
        T expected_x1 = T {7} / T {11};

        EXPECT_TRUE(approx_equal(x[0], expected_x0, T {1e-4}));
        EXPECT_TRUE(approx_equal(x[1], expected_x1, T {1e-4}));
    }

    TYPED_TEST(KrylovGPUTest, CGLaplacian) {
        using T = TypeParam;
        const int n = 10;
        std::vector<T> dense(n * n, T {0});

        for (int i = 0; i < n; ++i) {
            dense[i * n + i] = T {4};
            if (i > 0) {
                dense[i * n + i - 1] = T {-1};
            }
            if (i < n - 1) {
                dense[i * n + i + 1] = T {-1};
            }
        }

        auto matrix = SparseMatrix<T>::FromDense(dense.data(), n, n);
        ASSERT_TRUE(matrix.has_value());

        std::vector<T> b(n, T {1});
        std::vector<T> x(n, T {0});

        SolverConfig<T> config;
        config.relative_tolerance = T {1e-8};
        config.max_iterations = 500;

        ConjugateGradient<T> solver(config);
        auto result = solver.solve(*matrix, b.data(), x.data());

        EXPECT_TRUE(result.converged);
        EXPECT_LT(result.iterations, 100);

        cuda::memory::Buffer<T> d_x(n), d_temp(n);
        d_x.copy_from(x.data(), n);
        spmv(*matrix, d_x.data(), d_temp.data());
        std::vector<T> h_ax(n);
        d_temp.copy_to(h_ax.data(), n);

        for (int i = 0; i < n; ++i) {
            EXPECT_TRUE(approx_equal(h_ax[i], T {1}, T {1e-3}));
        }
    }

    TYPED_TEST(KrylovGPUTest, BiCGSTABTridiagonal) {
        using T = TypeParam;
        const int n = 20;
        std::vector<T> dense(n * n, T {0});

        for (int i = 0; i < n; ++i) {
            dense[i * n + i] = T {3};
            if (i > 0) {
                dense[i * n + i - 1] = T {-1};
            }
            if (i < n - 1) {
                dense[i * n + i + 1] = T {-1};
            }
        }

        auto matrix = SparseMatrix<T>::FromDense(dense.data(), n, n);
        ASSERT_TRUE(matrix.has_value());

        std::vector<T> b(n, T {1});
        std::vector<T> x(n, T {0});

        SolverConfig<T> config;
        config.relative_tolerance = T {1e-8};
        config.max_iterations = 200;

        BiCGSTAB<T> solver(config);
        auto result = solver.solve(*matrix, b.data(), x.data());

        EXPECT_TRUE(result.converged);
        EXPECT_LT(result.iterations, 100);

        cuda::memory::Buffer<T> d_x(n), d_ax(n);
        d_x.copy_from(x.data(), n);
        spmv(*matrix, d_x.data(), d_ax.data());
        std::vector<T> h_ax(n);
        d_ax.copy_to(h_ax.data(), n);

        for (int i = 0; i < n; ++i) {
            EXPECT_TRUE(approx_equal(h_ax[i], T {1}, T {1e-3}));
        }
    }

}  // namespace nova::sparse::test
