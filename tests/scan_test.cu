#include <gtest/gtest.h>
#include "parallel/scan.h"
#include "cuda/device/device_utils.h"
#include <vector>

class ScanTest : public ::testing::Test {
protected:
    size_t size_ = 8;
    std::vector<int> h_input_, h_output_;

    void SetUp() override {
        h_input_.resize(size_);
        h_output_.resize(size_);
        CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output_, size_ * sizeof(int)));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(d_input_));
        CUDA_CHECK(cudaFree(d_output_));
    }

    int *d_input_, *d_output_;
};

TEST_F(ScanTest, BasicPrefixSum) {
    h_input_ = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> expected = {0, 3, 4, 8, 9, 14, 23, 25};

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScan(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output_, expected);
}

TEST_F(ScanTest, SingleElement) {
    int single_input = 42;
    int single_output = -1;

    CUDA_CHECK(cudaMemcpy(d_input_, &single_input, sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScan(d_input_, d_output_, 1);
    CUDA_CHECK(cudaMemcpy(&single_output, d_output_, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(single_output, 0);
}

TEST_F(ScanTest, AllZeros) {
    h_input_.assign(size_, 0);
    std::vector<int> expected(size_, 0);

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScan(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output_, expected);
}

TEST_F(ScanTest, OptimizedVersion) {
    h_input_ = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> expected = {0, 3, 4, 8, 9, 14, 23, 25};

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScanOptimized(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output_, expected);
}

TEST_F(ScanTest, BasicAndOptimizedConsistency) {
    h_input_ = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> output_basic(size_), output_opt(size_);

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));

    exclusiveScan(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(output_basic.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    exclusiveScanOptimized(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(output_opt.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output_basic, output_opt);
}

TEST_F(ScanTest, InclusiveScan) {
    h_input_ = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> expected = {3, 4, 8, 9, 14, 23, 25, 31};

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    inclusiveScan(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output_, expected);
}

TEST_F(ScanTest, LargeArray) {
    size_ = 1024;
    h_input_.resize(size_);
    h_output_.resize(size_);

    CUDA_CHECK(cudaFree(d_input_));
    CUDA_CHECK(cudaFree(d_output_));
    CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_, size_ * sizeof(int)));

    for (size_t i = 0; i < size_; ++i) {
        h_input_[i] = 1;
    }

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScan(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_EQ(h_output_[i], static_cast<int>(i));
    }
}

TEST_F(ScanTest, AlternatingPattern) {
    size_ = 8;
    h_input_ = {1, 0, 1, 0, 1, 0, 1, 0};
    h_output_.resize(size_);

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScan(d_input_, d_output_, size_);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size_ * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> expected = {0, 1, 1, 2, 2, 3, 3, 4};
    EXPECT_EQ(h_output_, expected);
}

TEST_F(ScanTest, EmptyArray) {
    EXPECT_NO_THROW(exclusiveScan(d_input_, d_output_, 0));
}

TEST_F(ScanTest, MaximumSize) {
    size_t maxSize = 1024;
    h_input_.resize(maxSize);
    h_output_.resize(maxSize);

    CUDA_CHECK(cudaFree(d_input_));
    CUDA_CHECK(cudaFree(d_output_));
    CUDA_CHECK(cudaMalloc(&d_input_, maxSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_, maxSize * sizeof(int)));

    for (size_t i = 0; i < maxSize; ++i) {
        h_input_[i] = 1;
    }
    std::vector<int> expected(maxSize);
    for (size_t i = 0; i < maxSize; ++i) {
        expected[i] = static_cast<int>(i);
    }

    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), maxSize * sizeof(int), cudaMemcpyHostToDevice));
    exclusiveScan(d_input_, d_output_, maxSize);
    CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, maxSize * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output_, expected);
}
