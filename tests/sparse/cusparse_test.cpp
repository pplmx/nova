#include <gtest/gtest.h>
#include <cuda/sparse/matrix.hpp>
#include <cuda/sparse/cusparse_context.hpp>
#include <cuda/memory/buffer.h>

namespace nova::sparse::test {

    using cuda::memory::Buffer;

    template <typename T>
    bool approx_equal(T a, T b, T tol = T{1e-4}) {
        return std::abs(a - b) < tol;
    }

    TEST(SparseOpsGPUTest, SpmvSimple) {
        std::vector<float> dense = {4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f};
        auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);
        ASSERT_TRUE(matrix.has_value());

        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        Buffer<float> d_x(3);
        Buffer<float> d_y(3);
        std::vector<float> h_y(3);

        d_x.copy_from(x.data(), 3);

        spmv(*matrix, d_x.data(), d_y.data());

        h_y.assign(3, 0.0f);
        d_y.copy_to(h_y.data(), 3);

        EXPECT_TRUE(approx_equal(h_y[0], 6.0f));
        EXPECT_TRUE(approx_equal(h_y[1], 10.0f));
        EXPECT_TRUE(approx_equal(h_y[2], 8.0f));
    }

    TEST(SparseOpsGPUTest, SpmvLarger) {
        const int n = 100;
        std::vector<float> dense(n * n, 0.0f);

        for (int i = 0; i < n; ++i) {
            dense[i * n + i] = 4.0f;
            if (i > 0) {
                dense[i * n + i - 1] = -1.0f;
            }
            if (i < n - 1) {
                dense[i * n + i + 1] = -1.0f;
            }
        }

        auto matrix = SparseMatrix<float>::FromDense(dense.data(), n, n);
        ASSERT_TRUE(matrix.has_value());

        std::vector<float> x(n, 1.0f);
        Buffer<float> d_x(n);
        Buffer<float> d_y(n);
        std::vector<float> h_y(n);

        d_x.copy_from(x.data(), n);

        spmv(*matrix, d_x.data(), d_y.data());

        h_y.assign(n, 0.0f);
        d_y.copy_to(h_y.data(), n);

        EXPECT_TRUE(approx_equal(h_y[0], 3.0f));
        EXPECT_TRUE(approx_equal(h_y[n / 2], 2.0f));
        EXPECT_TRUE(approx_equal(h_y[n - 1], 3.0f));
    }

    TEST(SparseOpsGPUTest, SpmvTranspose) {
        std::vector<float> dense = {4.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 2.0f};
        auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);
        ASSERT_TRUE(matrix.has_value());

        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        Buffer<float> d_x(3);
        Buffer<float> d_y(3);
        std::vector<float> h_y(3);

        d_x.copy_from(x.data(), 3);

        spmv_transpose(*matrix, d_x.data(), d_y.data());

        h_y.assign(3, 0.0f);
        d_y.copy_to(h_y.data(), 3);

        EXPECT_TRUE(approx_equal(h_y[0], 6.0f));
        EXPECT_TRUE(approx_equal(h_y[1], 10.0f));
        EXPECT_TRUE(approx_equal(h_y[2], 8.0f));
    }

    TEST(SparseOpsGPUTest, DoublePrecision) {
        std::vector<double> dense = {4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};
        auto matrix = SparseMatrix<double>::FromDense(dense.data(), 3, 3, 0.0);
        ASSERT_TRUE(matrix.has_value());

        std::vector<double> x = {1.0, 2.0, 3.0};
        Buffer<double> d_x(3);
        Buffer<double> d_y(3);
        std::vector<double> h_y(3);

        d_x.copy_from(x.data(), 3);

        spmv(*matrix, d_x.data(), d_y.data());

        h_y.assign(3, 0.0);
        d_y.copy_to(h_y.data(), 3);

        EXPECT_TRUE(approx_equal(h_y[0], 6.0, 1e-8));
        EXPECT_TRUE(approx_equal(h_y[1], 10.0, 1e-8));
        EXPECT_TRUE(approx_equal(h_y[2], 8.0, 1e-8));
    }

    TEST(SparseOpsGPUTest, CusparseContextSingleton) {
        auto& ctx1 = detail::CusparseContext::get();
        auto& ctx2 = detail::CusparseContext::get();
        EXPECT_EQ(&ctx1, &ctx2);
        EXPECT_NE(ctx1.handle(), nullptr);
    }

}  // namespace nova::sparse::test
