#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include "parallel/scan.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include <vector>
#include <numeric>

using cuda::memory::Buffer;

class ScanTest : public ::testing::Test {
protected:
    size_t size_ = 8;

    void SetUp() override {
        d_input_ = cuda::memory::Buffer<int>(size_);
        d_output_ = cuda::memory::Buffer<int>(size_);
    }

    cuda::memory::Buffer<int> d_input_;
    cuda::memory::Buffer<int> d_output_;
};

TEST_F(ScanTest, BasicPrefixSum) {
    std::vector<int> h_input = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> expected = {0, 3, 4, 8, 9, 14, 23, 25};

    d_input_.copy_from(h_input.data(), h_input.size());
    cuda::algo::exclusiveScan(d_input_, d_output_, size_);

    std::vector<int> h_output(size_);
    d_output_.copy_to(h_output.data(), size_);

    EXPECT_EQ(h_output, expected);
}

TEST_F(ScanTest, SingleElement) {
    std::vector<int> h_input = {42};
    std::vector<int> h_output(1);

    d_input_.copy_from(h_input.data(), 1);
    cuda::algo::exclusiveScan(d_input_, d_output_, 1);
    d_output_.copy_to(h_output.data(), 1);

    EXPECT_EQ(h_output[0], 0);
}

TEST_F(ScanTest, AllZeros) {
    std::vector<int> h_input(size_, 0);
    std::vector<int> expected(size_, 0);

    d_input_.copy_from(h_input.data(), size_);
    cuda::algo::exclusiveScan(d_input_, d_output_, size_);

    std::vector<int> h_output(size_);
    d_output_.copy_to(h_output.data(), size_);

    EXPECT_EQ(h_output, expected);
}

TEST_F(ScanTest, OptimizedVersion) {
    std::vector<int> h_input = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> expected = {0, 3, 4, 8, 9, 14, 23, 25};

    d_input_.copy_from(h_input.data(), size_);
    cuda::algo::exclusiveScanOptimized(d_input_, d_output_, size_);

    std::vector<int> h_output(size_);
    d_output_.copy_to(h_output.data(), size_);

    EXPECT_EQ(h_output, expected);
}

TEST_F(ScanTest, BasicAndOptimizedConsistency) {
    std::vector<int> h_input = {3, 1, 4, 1, 5, 9, 2, 6};

    d_input_.copy_from(h_input.data(), size_);

    cuda::memory::Buffer<int> output_basic(size_);
    cuda::algo::exclusiveScan(d_input_, output_basic, size_);

    cuda::algo::exclusiveScanOptimized(d_input_, d_output_, size_);

    std::vector<int> h_basic(size_), h_opt(size_);
    output_basic.copy_to(h_basic.data(), size_);
    d_output_.copy_to(h_opt.data(), size_);

    EXPECT_EQ(h_basic, h_opt);
}

TEST_F(ScanTest, InclusiveScan) {
    std::vector<int> h_input = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> expected = {3, 4, 8, 9, 14, 23, 25, 31};

    d_input_.copy_from(h_input.data(), size_);
    cuda::algo::inclusiveScan(d_input_, d_output_, size_);

    std::vector<int> h_output(size_);
    d_output_.copy_to(h_output.data(), size_);

    EXPECT_EQ(h_output, expected);
}

TEST_F(ScanTest, LargeArray) {
    size_t large_size = 1024;
    d_input_ = cuda::memory::Buffer<int>(large_size);
    d_output_ = cuda::memory::Buffer<int>(large_size);

    std::vector<int> h_input(large_size, 1);
    d_input_.copy_from(h_input.data(), large_size);

    cuda::algo::exclusiveScan(d_input_, d_output_, large_size);

    std::vector<int> h_output(large_size);
    d_output_.copy_to(h_output.data(), large_size);

    for (size_t i = 0; i < large_size; ++i) {
        EXPECT_EQ(h_output[i], static_cast<int>(i));
    }
}

TEST_F(ScanTest, AlternatingPattern) {
    std::vector<int> h_input = {1, 0, 1, 0, 1, 0, 1, 0};
    std::vector<int> expected = {0, 1, 1, 2, 2, 3, 3, 4};

    d_input_.copy_from(h_input.data(), size_);
    cuda::algo::exclusiveScan(d_input_, d_output_, size_);

    std::vector<int> h_output(size_);
    d_output_.copy_to(h_output.data(), size_);

    EXPECT_EQ(h_output, expected);
}

TEST_F(ScanTest, EmptyArray) {
    cuda::memory::Buffer<int> empty_input(1);
    cuda::memory::Buffer<int> empty_output(1);
    EXPECT_NO_THROW(cuda::algo::exclusiveScan(empty_input, empty_output, 0));
}

TEST_F(ScanTest, ExceedsMaxSize) {
    size_t largeSize = MAX_SCAN_SIZE + 1;
    cuda::memory::Buffer<int> large_input(largeSize);
    cuda::memory::Buffer<int> large_output(largeSize);

    std::vector<int> h_large(largeSize, 1);
    large_input.copy_from(h_large.data(), largeSize);

    EXPECT_THROW(cuda::algo::exclusiveScan(large_input, large_output, largeSize), ScanSizeException);
    EXPECT_THROW(cuda::algo::inclusiveScan(large_input, large_output, largeSize), ScanSizeException);
}

TEST_F(ScanTest, MaximumSize) {
    size_t maxSize = 1024;
    d_input_ = cuda::memory::Buffer<int>(maxSize);
    d_output_ = cuda::memory::Buffer<int>(maxSize);

    std::vector<int> h_input(maxSize, 1);
    std::vector<int> expected(maxSize);
    for (size_t i = 0; i < maxSize; ++i) {
        expected[i] = static_cast<int>(i);
    }

    d_input_.copy_from(h_input.data(), maxSize);
    cuda::algo::exclusiveScan(d_input_, d_output_, maxSize);

    std::vector<int> h_output(maxSize);
    d_output_.copy_to(h_output.data(), maxSize);

    EXPECT_EQ(h_output, expected);
}
