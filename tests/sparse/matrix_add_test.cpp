/**
 * @file matrix_add_test.cpp
 * @brief Contracts for elementwise matrix addition
 *
 * src/matrix/add.cu implements cuda::algo::matrixAdd. It had zero test coverage,
 * and the interface could not be called at all:
 *
 *   void matrixAdd(memory::Buffer<float> a, memory::Buffer<float> b,
 *                  memory::Buffer<float> c, int numRows, int numCols);
 *
 * memory::Buffer deletes its copy constructor and copy assignment because it
 * owns device memory, so taking the parameters BY VALUE required a copy and no
 * call site compiled. Only a std::move at the call site made it build — and that
 * destroyed the caller's inputs to perform a read-only operation.
 *
 * The parameters are now const& for the inputs and & for the output, and the
 * function rejects a shape that does not fit the buffers instead of launching a
 * zero-sized grid or reading past the end.
 */

#include <matrix/add.h>
#include <cuda/memory/buffer.h>
#include <cuda/memory/buffer-inl.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
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

    }  // namespace

    // Regression: with by-value parameters this test did not compile, because
    // Buffer's copy constructor is deleted.
    TEST(MatrixAddTest, AddsTwoMatricesElementwise) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int rows = 3, cols = 4;
        std::vector<float> a(rows * cols), b(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i) * 10.0f;
        }

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(a.size());
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixAdd(d_a, d_b, d_c, rows, cols);
        (void)cudaDeviceSynchronize();

        std::vector<float> result(a.size());
        d_c.copy_to(result.data(), result.size());

        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_FLOAT_EQ(result[i], a[i] + b[i]) << "element " << i;
        }
    }

    // The inputs must survive the call: a read-only operation that consumes its
    // arguments is not usable.
    TEST(MatrixAddTest, InputsAreNotConsumed) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const int rows = 2, cols = 2;
        const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(a.size());
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixAdd(d_a, d_b, d_c, rows, cols);
        (void)cudaDeviceSynchronize();

        // Reuse the same input buffers for a second call; they must still hold their
        // original contents.
        memory::Buffer<float> d_c2(a.size());
        matrixAdd(d_a, d_b, d_c2, rows, cols);
        (void)cudaDeviceSynchronize();

        std::vector<float> first(a.size()), second(a.size());
        d_c.copy_to(first.data(), first.size());
        d_c2.copy_to(second.data(), second.size());

        EXPECT_EQ(first, second) << "a repeated call must produce the same result";
        EXPECT_FLOAT_EQ(second[0], 6.0f);
    }

    TEST(MatrixAddTest, HandlesNonSquareMatrices) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 1 row by 5 columns stresses the column dimension of the grid.
        const int rows = 1, cols = 5;
        std::vector<float> a(cols, 1.5f);
        std::vector<float> b(cols, 2.5f);

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(a.size());
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixAdd(d_a, d_b, d_c, rows, cols);
        (void)cudaDeviceSynchronize();

        std::vector<float> result(a.size());
        d_c.copy_to(result.data(), result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_FLOAT_EQ(result[i], 4.0f) << "element " << i;
        }
    }

    TEST(MatrixAddTest, HandlesManyRowsAndColumns) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Larger than one 16x16 block in both dimensions.
        const int rows = 40, cols = 37;
        std::vector<float> a(static_cast<size_t>(rows) * cols, 1.0f);
        std::vector<float> b(static_cast<size_t>(rows) * cols, 2.0f);

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(a.size());
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixAdd(d_a, d_b, d_c, rows, cols);
        (void)cudaDeviceSynchronize();

        std::vector<float> result(a.size());
        d_c.copy_to(result.data(), result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            ASSERT_FLOAT_EQ(result[i], 3.0f) << "element " << i;
        }
    }

    TEST(MatrixAddTest, ZeroDimensionsAreANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        memory::Buffer<float> d_a(4);
        memory::Buffer<float> d_b(4);
        memory::Buffer<float> d_c(4);

        // A zero-sized grid is rejected by CUDA, so this must return without
        // launching rather than throwing "invalid configuration argument".
        EXPECT_NO_THROW(matrixAdd(d_a, d_b, d_c, 0, 4));
        EXPECT_NO_THROW(matrixAdd(d_a, d_b, d_c, 4, 0));
        EXPECT_NO_THROW(matrixAdd(d_a, d_b, d_c, 0, 0));
        EXPECT_NO_THROW(matrixAdd(d_a, d_b, d_c, -1, 4));
        (void)cudaGetLastError();
    }

    TEST(MatrixAddTest, UndersizedBuffersAreRejected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Claim a 4x4 shape but supply buffers for only two elements: the kernel
        // would read and write past the end.
        memory::Buffer<float> d_a(2);
        memory::Buffer<float> d_b(2);
        memory::Buffer<float> d_c(2);

        EXPECT_THROW(matrixAdd(d_a, d_b, d_c, 4, 4), std::invalid_argument);
        (void)cudaGetLastError();
    }

    TEST(MatrixAddTest, ZeroInputValuesPropagate) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Guards against a kernel that skips the add (for example by initialising the
        // output instead of summing).
        const int rows = 2, cols = 3;
        std::vector<float> a(rows * cols, 0.0f);
        std::vector<float> b(rows * cols, 7.0f);

        memory::Buffer<float> d_a(a.size());
        memory::Buffer<float> d_b(b.size());
        memory::Buffer<float> d_c(a.size());
        d_a.copy_from(a.data(), a.size());
        d_b.copy_from(b.data(), b.size());

        matrixAdd(d_a, d_b, d_c, rows, cols);
        (void)cudaDeviceSynchronize();

        std::vector<float> result(a.size());
        d_c.copy_to(result.data(), result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_FLOAT_EQ(result[i], 7.0f) << "element " << i;
        }
    }

}  // namespace cuda::algo::test
