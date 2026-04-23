#include <gtest/gtest.h>
#include "cuda/device/device_utils.h"
#include "cuda/memory/buffer.h"
#include "cuda/algo/reduce.h"
#include <numeric>

class ReduceTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    cuda::memory::Buffer<int> d_input_;

    void SetUp() override {
        h_input_.resize(size_);
        d_input_ = cuda::memory::Buffer<int>(size_);
    }

    void TearDown() override {
        d_input_.release();
    }
};

TEST_F(ReduceTest, SumBasic) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_sum(d_input_.data(), size_);
    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;

    EXPECT_EQ(result, expected);
}

TEST_F(ReduceTest, SumOptimized) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_sum_optimized(d_input_.data(), size_);
    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;

    EXPECT_EQ(result, expected);
}

TEST_F(ReduceTest, SumConsistency) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    d_input_.copy_from(h_input_.data(), size_);

    int basic = cuda::algo::reduce_sum(d_input_.data(), size_);
    int optimized = cuda::algo::reduce_sum_optimized(d_input_.data(), size_);

    EXPECT_EQ(basic, optimized);
}

TEST_F(ReduceTest, MaxTest) {
    h_input_.assign(size_, 0);
    h_input_[500] = 999;
    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_max(d_input_.data(), size_);
    EXPECT_EQ(result, 999);
}

TEST_F(ReduceTest, MinTest) {
    for (int i = 0; i < static_cast<int>(size_); ++i) h_input_[i] = i + 100;
    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_min(d_input_.data(), size_);
    EXPECT_EQ(result, 100);
}

TEST_F(ReduceTest, EmptyInput) {
    int result = cuda::algo::reduce_sum(d_input_.data(), 0);
    EXPECT_EQ(result, 0);
}

TEST_F(ReduceTest, SingleElement) {
    h_input_[0] = 42;
    d_input_.copy_from(h_input_.data(), 1);
    int result = cuda::algo::reduce_sum(d_input_.data(), 1);
    EXPECT_EQ(result, 42);
}

TEST_F(ReduceTest, NonPowerOfTwo) {
    h_input_.resize(1000);
    for (int i = 0; i < 1000; ++i) h_input_[i] = i + 1;
    d_input_ = cuda::memory::Buffer<int>(1000);
    d_input_.copy_from(h_input_.data(), 1000);
    int result = cuda::algo::reduce_sum(d_input_.data(), 1000);
    EXPECT_EQ(result, 1000 * 1001 / 2);
}

TEST_F(ReduceTest, LargeArray) {
    size_ = 1 << 20;
    h_input_.resize(size_);

    d_input_ = cuda::memory::Buffer<int>(size_);

    for (size_t i = 0; i < size_; ++i) h_input_[i] = static_cast<int>(i + 1);

    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_sum(d_input_.data(), size_);
    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;

    EXPECT_EQ(result, expected);
}

TEST_F(ReduceTest, NegativeNumbers) {
    h_input_.resize(100);
    for (int i = 0; i < 100; ++i) h_input_[i] = i - 50;
    d_input_ = cuda::memory::Buffer<int>(100);
    d_input_.copy_from(h_input_.data(), 100);

    int result = cuda::algo::reduce_sum(d_input_.data(), 100);
    int expected = -50;

    EXPECT_EQ(result, expected);
}

TEST_F(ReduceTest, MinTestEdge) {
    h_input_.assign(size_, 1000);
    h_input_[42] = -999;
    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_min(d_input_.data(), size_);
    EXPECT_EQ(result, -999);
}
