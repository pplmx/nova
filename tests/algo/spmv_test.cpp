#include <gtest/gtest.h>
#include <vector>

#include "cuda/algo/spmv.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::spmv::test {

class SpMVTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SpMVTest, MultiplyCSRSimple) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> row_offsets = {0, 2, 3, 4};
    std::vector<int> col_indices = {0, 1, 1, 2};
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y(3);

    cuda::memory::Buffer<float> d_values(values.size());
    cuda::memory::Buffer<int> d_row_offsets(row_offsets.size());
    cuda::memory::Buffer<int> d_col_indices(col_indices.size());
    cuda::memory::Buffer<float> d_x(x.size());
    cuda::memory::Buffer<float> d_y(3);

    cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets.data(), row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices.data(), col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x.data(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    cuda::algo::spmv::multiply_csr(d_values.data(), d_row_offsets.data(),
                                    d_col_indices.data(), d_x.data(), d_y.data(), 3);

    cudaMemcpy(y.data(), d_y.data(), 3 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(y[0], 1.0f * 1.0f + 2.0f * 2.0f, 1e-5f);
    EXPECT_NEAR(y[1], 3.0f * 2.0f, 1e-5f);
    EXPECT_NEAR(y[2], 4.0f * 3.0f, 1e-5f);
}

TEST_F(SpMVTest, MultiplyCSRDouble) {
    std::vector<double> values = {1.0, 2.0};
    std::vector<int> row_offsets = {0, 1, 2};
    std::vector<int> col_indices = {0, 1};
    std::vector<double> x = {2.0, 3.0};
    std::vector<double> y(2);

    cuda::memory::Buffer<double> d_values(values.size());
    cuda::memory::Buffer<int> d_row_offsets(row_offsets.size());
    cuda::memory::Buffer<int> d_col_indices(col_indices.size());
    cuda::memory::Buffer<double> d_x(x.size());
    cuda::memory::Buffer<double> d_y(2);

    cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets.data(), row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices.data(), col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x.data(), x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);

    cuda::algo::spmv::multiply_csr(d_values.data(), d_row_offsets.data(),
                                    d_col_indices.data(), d_x.data(), d_y.data(), 2);

    cudaMemcpy(y.data(), d_y.data(), 2 * sizeof(double), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(y[0], 1.0 * 2.0, 1e-10);
    EXPECT_NEAR(y[1], 2.0 * 3.0, 1e-10);
}

TEST_F(SpMVTest, MultiplyGeneric) {
    std::vector<float> values = {1.0f, 2.0f};
    std::vector<int> offsets = {0, 1, 2};
    std::vector<int> indices = {0, 1};
    std::vector<float> x = {2.0f, 3.0f};
    std::vector<float> y(2);

    cuda::memory::Buffer<float> d_values(values.size());
    cuda::memory::Buffer<int> d_offsets(offsets.size());
    cuda::memory::Buffer<int> d_indices(indices.size());
    cuda::memory::Buffer<float> d_x(x.size());
    cuda::memory::Buffer<float> d_y(2);

    cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets.data(), offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices.data(), indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x.data(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    cuda::algo::spmv::multiply(d_values.data(), d_offsets.data(), d_indices.data(),
                                d_x.data(), d_y.data(), 2, Format::CSR);

    cudaMemcpy(y.data(), d_y.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(y[0], 2.0f, 1e-5f);
    EXPECT_NEAR(y[1], 6.0f, 1e-5f);
}

TEST_F(SpMVTest, ConfigSetters) {
    SpMVConfig config;
    config.vectorization_width = 8;
    config.row_chunk_size = 128;

    cuda::algo::spmv::set_config(config);

    auto retrieved = cuda::algo::spmv::get_config();
    EXPECT_EQ(retrieved.vectorization_width, 8);
    EXPECT_EQ(retrieved.row_chunk_size, 128);
}

TEST_F(SpMVTest, MultiplyCSCSimple) {
    // A = [[1, 0, 2],
    //      [0, 3, 0],
    //      [4, 0, 5]]
    // CSC: col_offsets = [0, 2, 3, 5]
    //      row_indices = [0, 2, 1, 0, 2]
    //      values      = [1, 4, 3, 2, 5]
    // x = [1, 2, 3]
    // y = A * x = [1*1+2*3, 3*2, 4*1+5*3] = [7, 6, 19]
    std::vector<float> values = {1.0f, 4.0f, 3.0f, 2.0f, 5.0f};
    std::vector<int> col_offsets = {0, 2, 3, 5};
    std::vector<int> row_indices = {0, 2, 1, 0, 2};
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {0.0f, 0.0f, 0.0f};

    cuda::memory::Buffer<float> d_values(values.size());
    cuda::memory::Buffer<int> d_col_offsets(col_offsets.size());
    cuda::memory::Buffer<int> d_row_indices(row_indices.size());
    cuda::memory::Buffer<float> d_x(x.size());
    cuda::memory::Buffer<float> d_y(y.size());

    d_values.copy_from(values.data(), values.size());
    d_col_offsets.copy_from(col_offsets.data(), col_offsets.size());
    d_row_indices.copy_from(row_indices.data(), row_indices.size());
    d_x.copy_from(x.data(), x.size());

    cuda::algo::spmv::multiply_csc(d_values.data(), d_col_offsets.data(),
                                    d_row_indices.data(), d_x.data(), d_y.data(), 3);

    d_y.copy_to(y.data(), y.size());

    EXPECT_NEAR(y[0], 7.0f, 1e-5f);
    EXPECT_NEAR(y[1], 6.0f, 1e-5f);
    EXPECT_NEAR(y[2], 19.0f, 1e-5f);
}

}  // namespace cuda::algo::spmv::test
