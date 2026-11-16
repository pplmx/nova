/**
 * @file matrix_elementwise_edge_test.cpp
 * @brief Contracts for the elementwise matrix entry points on empty extents
 *
 * src/matrix/ops.cu's transpose entry points return early for an empty extent, but the
 * three elementwise ones did not: they derived a grid from the caller's dimensions and
 * launched it. Measured before the fix:
 *
 *     matrixElementwiseAdd(rows=0)       -> CUDA error: invalid configuration argument
 *     matrixElementwiseAdd(rows=-2)      -> the same
 *     matrixScale(size=0)                -> the same
 *
 * A negative extent additionally wrapped when rows * cols was converted to size_t, so the
 * grid expression saw a huge count rather than a negative one.
 */

#include <matrix/ops.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace matrix_elementwise_edge_test {

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

// Regression: all three threw "invalid configuration argument".
TEST(MatrixElementwiseEdgeTest, EmptyExtentsAreNoOps) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    float scratch[1] = {1.0f};
    (void)cudaGetLastError();

    EXPECT_NO_THROW(matrixElementwiseAdd<float>(scratch, scratch, scratch, 0, 4));
    EXPECT_NO_THROW(matrixElementwiseAdd<float>(scratch, scratch, scratch, -2, 4));
    EXPECT_NO_THROW(matrixElementwiseAdd<float>(scratch, scratch, scratch, 4, 0));
    EXPECT_NO_THROW(matrixElementwiseMultiply<float>(scratch, scratch, scratch, 0, 4));
    EXPECT_NO_THROW(matrixElementwiseMultiply<float>(scratch, scratch, scratch, -1, -1));
    EXPECT_NO_THROW(matrixScale<float>(scratch, 2.0f, 0));
    EXPECT_NO_THROW(matrixScale<float>(scratch, 2.0f, -5));

    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "an empty elementwise call must not leave a launch error behind";
}

namespace {

/// Owns a device float array and copies to and from it.
class DeviceArray {
public:
    explicit DeviceArray(const std::vector<float>& host) : count_(host.size()) {
        cudaMalloc(&ptr_, (count_ > 0 ? count_ : 1) * sizeof(float));
        if (count_ > 0) {
            cudaMemcpy(ptr_, host.data(), count_ * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
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

TEST(MatrixElementwiseEdgeTest, AddAndMultiplyFollowTheDefinition) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // These entry points take DEVICE pointers (the parameters are named d_a, d_b, d_c);
    // passing std::vector data() fails with an illegal memory access. My first version
    // of this case did exactly that.
    const std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b_host = {10.0f, 20.0f, 30.0f, 40.0f};

    DeviceArray a(a_host);
    DeviceArray b(b_host);
    DeviceArray sum(std::vector<float>(4, 0.0f));
    DeviceArray product(std::vector<float>(4, 0.0f));

    matrixElementwiseAdd<float>(a.data(), b.data(), sum.data(), 2, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    matrixElementwiseMultiply<float>(a.data(), b.data(), product.data(), 2, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto sum_host = sum.read();
    const auto product_host = product.read();
    for (size_t i = 0; i < a_host.size(); ++i) {
        EXPECT_NEAR(sum_host[i], a_host[i] + b_host[i], 1e-5f) << "sum element " << i;
        EXPECT_NEAR(product_host[i], a_host[i] * b_host[i], 1e-5f)
            << "product element " << i;
    }
}

TEST(MatrixElementwiseEdgeTest, ScaleMultipliesEveryElement) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<float> original = {1.0f, -2.0f, 3.0f, -4.0f};
    DeviceArray matrix(original);

    matrixScale<float>(matrix.data(), 2.5f, 4);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto scaled = matrix.read();
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(scaled[i], original[i] * 2.5f, 1e-5f) << "element " << i;
    }
}

}  // namespace matrix_elementwise_edge_test
