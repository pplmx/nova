/**
 * @file spmv_output_initialisation_test.cpp
 * @brief Contracts for the SpMV output buffer's initial contents
 *
 * src/cuda/sparse/sparse_ops.cu calls cuSPARSE's SpMV with beta = 0, which means "y is
 * overwritten, its incoming contents do not matter". cuSPARSE does not honour that
 * literally: for small matrices it selects a vector_scalar_multiply fast path that
 * LOADS y before scaling by beta. Passing an uninitialised buffer therefore hands the
 * library uninitialised memory, which compute-sanitizer --tool initcheck reports:
 *
 *     Uninitialized __global__ memory read of size 4 bytes
 *         at cusparse::vector_scalar_multiply_kernel<...>
 *         by thread (0,0,0) in block (0,0,0)
 *
 * 22 such reads across the sparse suites, all from this one cause. Both spmv_impl and
 * its transpose now clear y before the call.
 *
 * The result was always correct, so no ordinary assertion could catch this — it is an
 * input-hygiene defect, and the tests below pin the behaviour that matters: the output
 * never depends on what the caller's buffer held.
 */

#include <cuda/sparse/matrix.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace nova::sparse::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Owns a device float array filled with `fill`, which gives a caller-defined
/// starting state rather than whatever the allocator hands back.
class DeviceArray {
public:
    DeviceArray(size_t count, float fill) : count_(count) {
        cudaMalloc(&ptr_, (count > 0 ? count : 1) * sizeof(float));
        std::vector<float> host(count, fill);
        cudaMemcpy(ptr_, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~DeviceArray() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    float* data() { return ptr_; }

    std::vector<float> read() const {
        std::vector<float> host(count_, 0.0f);
        cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost);
        return host;
    }

private:
    float* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace

// The result must not depend on what y held beforehand.
TEST(SpmvOutputInitialisationTest, ResultIsIndependentOfTheOutputBuffersPriorContents) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A 3x3 tridiagonal matrix; the small size is what selects cuSPARSE's
    // vector_scalar_multiply path, and it is where the reads were observed.
    const std::vector<float> dense = {
        4.0f, 1.0f, 0.0f,
        1.0f, 3.0f, 1.0f,
        0.0f, 1.0f, 2.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);
    ASSERT_TRUE(matrix.has_value());

    const std::vector<float> x = {1.0f, 2.0f, 3.0f};
    // 4*1 + 1*2 = 6; 1*1 + 3*2 + 1*3 = 10; 1*2 + 2*3 = 8.
    const std::vector<float> expected = {6.0f, 10.0f, 8.0f};

    // Two runs with deliberately hostile starting values.
    for (const float fill : {0.0f, -12345.0f}) {
        DeviceArray d_x(x.size(), 0.0f);
        DeviceArray d_y(x.size(), fill);
        ASSERT_EQ(cudaMemcpy(d_x.data(), x.data(), x.size() * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);

        spmv(*matrix, d_x.data(), d_y.data());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = d_y.read();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(result[i], expected[i], 1e-5f)
                << "run with y prefilled " << fill << ", element " << i;
        }
    }
}

TEST(SpmvOutputInitialisationTest, ATransposeSpMVAlsoIgnoresItsOutputBuffer) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A non-symmetric matrix, so a transposed product differs from the plain one.
    const std::vector<float> dense = {
        1.0f, 2.0f, 0.0f,
        0.0f, 3.0f, 4.0f,
        5.0f, 0.0f, 6.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);
    ASSERT_TRUE(matrix.has_value());

    const std::vector<float> x = {1.0f, 1.0f, 1.0f};

    std::vector<float> first;
    for (const float fill : {0.0f, 99999.0f}) {
        DeviceArray d_x(x.size(), 0.0f);
        DeviceArray d_y(x.size(), fill);
        ASSERT_EQ(cudaMemcpy(d_x.data(), x.data(), x.size() * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);

        spmv_transpose(*matrix, d_x.data(), d_y.data());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = d_y.read();
        if (first.empty()) {
            first = result;
            // Column sums: 1+0+5 = 6, 2+3+0 = 5, 0+4+6 = 10.
            EXPECT_NEAR(result[0], 6.0f, 1e-5f);
            EXPECT_NEAR(result[1], 5.0f, 1e-5f);
            EXPECT_NEAR(result[2], 10.0f, 1e-5f);
        } else {
            for (size_t i = 0; i < result.size(); ++i) {
                EXPECT_NEAR(result[i], first[i], 1e-5f)
                    << "the transposed result changed with the output buffer's "
                       "starting contents, element "
                    << i;
            }
        }
    }
}

TEST(SpmvOutputInitialisationTest, TheVectorOverloadSizesAndFillsTheOutput) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> dense = {
        2.0f, 0.0f,
        0.0f, 3.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 2, 2, 0.0f);
    ASSERT_TRUE(matrix.has_value());

    // The host-vector overload allocates its own device output, so this exercises the
    // same path the tests above reach from a differently-shaped entry point.
    const std::vector<float> x = {5.0f, 7.0f};
    std::vector<float> y;
    spmv(*matrix, x, y);

    ASSERT_EQ(y.size(), 2u);
    EXPECT_NEAR(y[0], 10.0f, 1e-5f);
    EXPECT_NEAR(y[1], 21.0f, 1e-5f);
}

}  // namespace cuda::sparse::test
