/**
 * @file cublas_gemm_test.cpp
 * @brief Contracts for the row-major cuBLAS matrix multiply
 *
 * src/cuda/device/cublas_context.cu implements cuda::algo::matrixMultiply, which
 * multiplies two ROW-MAJOR buffers via cuBLAS. It had no test coverage at all,
 * and the call was wrong:
 *
 * cuBLAS is column-major only. Reading row-major A as if it were column-major
 * means the buffer really holds A transpose, so C = A*B has to be expressed as
 * C^T = B^T * A^T: one call with the operands and the result dimensions SWAPPED.
 *
 * The old code passed the operands in order with lda=k, ldb=n, ldc=n. Two
 * consequences:
 *
 * - For m=2, n=4, k=3 it produced {4,5,0,0,4,5,0,0} instead of the correct
 *   {0,6,0,6,0,15,0,15} — a wrong answer, not a rounding difference.
 * - Whenever m > n it passed ldc = n < m, which cuBLAS rejects with
 *   "parameter number 10 had an illegal value", so the call could not complete.
 *
 * These tests compare against a direct row-major product computed on the host.
 */

#include <cuda/device/cublas_context.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace cuda::algo::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// Reference row-major product computed on the host.
        std::vector<float> reference_product(const std::vector<float>& a, const std::vector<float>& b, int m, int n, int k) {
            std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; ++p) {
                        sum += a[static_cast<size_t>(i) * k + p] * b[static_cast<size_t>(p) * n + j];
                    }
                    c[static_cast<size_t>(i) * n + j] = sum;
                }
            }
            return c;
        }

    }  // namespace

    // Regression: the old call returned a different matrix entirely.
    TEST(CublasGemmTest, MatchesTheRowMajorProduct) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A is 2x3, B is 3x4, so C is 2x4. n > m, which the old call at least did
        // not reject, yet still computed wrongly.
        const int m = 2, n = 4, k = 3;
        std::vector<float> a(static_cast<size_t>(m) * k);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(i + 1);
        }
        std::vector<float> b(static_cast<size_t>(k) * n);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = (i % 2 == 0) ? 0.0f : 1.0f;
        }

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(static_cast<size_t>(m) * n);
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixMultiply(d_a, d_b, d_c, m, n, k);
        (void)cudaDeviceSynchronize();

        std::vector<float> result(static_cast<size_t>(m) * n);
        d_c.copy_to(result.data(), result.size());

        const auto expected = reference_product(a, b, m, n, k);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(result[i], expected[i]) << "element " << i << " differs; the row-major convention is wrong";
        }
    }

    // Regression: m > n made ldc = n < m, which cuBLAS rejects outright.
    TEST(CublasGemmTest, WorksWhenRowsOutnumberColumns) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A is 4x2, B is 2x3, so C is 4x3 and m (4) exceeds n (3).
        const int m = 4, n = 3, k = 2;
        std::vector<float> a(static_cast<size_t>(m) * k);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
        }
        std::vector<float> b(static_cast<size_t>(k) * n);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<float>(i + 1);
        }

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(static_cast<size_t>(m) * n);
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        EXPECT_NO_THROW(matrixMultiply(d_a, d_b, d_c, m, n, k)) << "ldc must be max(1, m) under the column-major view";
        (void)cudaDeviceSynchronize();

        std::vector<float> result(static_cast<size_t>(m) * n);
        d_c.copy_to(result.data(), result.size());

        const auto expected = reference_product(a, b, m, n, k);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(result[i], expected[i], 1e-4f) << "element " << i;
        }
    }

    TEST(CublasGemmTest, SquareMatrices) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int size = 4;
        std::vector<float> a(static_cast<size_t>(size) * size, 1.0f);
        for (int i = 0; i < size; ++i) {
            a[static_cast<size_t>(i) * size + i] = static_cast<float>(i + 1);
        }
        std::vector<float> b(static_cast<size_t>(size) * size, 0.0f);
        for (int i = 0; i < size; ++i) {
            b[static_cast<size_t>(i) * size + i] = 1.0f;
        }

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(a.size());
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixMultiply(d_a, d_b, d_c, size, size, size);
        (void)cudaDeviceSynchronize();

        std::vector<float> result(a.size());
        d_c.copy_to(result.data(), result.size());

        // A * I == A.
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_FLOAT_EQ(result[i], a[i]) << "element " << i;
        }
    }

    TEST(CublasGemmTest, DoublePrecisionMatchesTheRowMajorProduct) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int m = 3, n = 2, k = 4;
        std::vector<double> a(static_cast<size_t>(m) * k);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<double>(i) * 0.25;
        }
        std::vector<double> b(static_cast<size_t>(k) * n);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<double>(i + 1);
        }

        // Host reference in double.
        std::vector<double> expected(static_cast<size_t>(m) * n, 0.0);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int p = 0; p < k; ++p) {
                    sum += a[static_cast<size_t>(i) * k + p] * b[static_cast<size_t>(p) * n + j];
                }
                expected[static_cast<size_t>(i) * n + j] = sum;
            }
        }

        memory::Buffer<double> d_a(a.size());
        memory::Buffer<double> d_b(b.size());
        memory::Buffer<double> d_c(static_cast<size_t>(m) * n);
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixMultiply(d_a, d_b, d_c, m, n, k);
        (void)cudaDeviceSynchronize();

        std::vector<double> result(static_cast<size_t>(m) * n);
        d_c.copy_to(result.data(), result.size());

        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_DOUBLE_EQ(result[i], expected[i]) << "element " << i;
        }
    }

    TEST(CublasGemmTest, DegenerateDimensionsAreANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        memory::Buffer<float> a(4);
        memory::Buffer<float> b(4);
        memory::Buffer<float> c(4);

        // A zero or negative dimension cannot describe a product; returning without
        // touching cuBLAS avoids its illegal-value error.
        EXPECT_NO_THROW(matrixMultiply(a, b, c, 0, 2, 2));
        EXPECT_NO_THROW(matrixMultiply(a, b, c, 2, 0, 2));
        EXPECT_NO_THROW(matrixMultiply(a, b, c, 2, 2, 0));
        EXPECT_NO_THROW(matrixMultiply(a, b, c, -1, 2, 2));
    }

}  // namespace cuda::algo::test
