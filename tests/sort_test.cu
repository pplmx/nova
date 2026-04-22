#include <gtest/gtest.h>
#include "sort.h"
#include "cuda_utils.h"
#include <algorithm>

class SortTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    std::vector<int> h_output_;
    int *d_input_ = nullptr;
    int *d_output_ = nullptr;

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

    void runSort(size_t size) {
        CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        bitonicSort(d_input_, d_output_, size);
        CUDA_CHECK(cudaMemcpy(h_output_.data(), d_output_, size * sizeof(int), cudaMemcpyDeviceToHost));
    }
};

TEST_F(SortTest, RandomArray) {
    h_input_ = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    h_output_.resize(h_input_.size());

    runSort(h_input_.size());

    std::vector<int> expected = h_input_;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(h_output_, expected);
}

TEST_F(SortTest, AlreadySorted) {
    h_input_ = {1, 2, 3, 4, 5, 6, 7, 8};
    h_output_.resize(h_input_.size());

    runSort(h_input_.size());

    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(h_output_, expected);
}

TEST_F(SortTest, ReverseSorted) {
    h_input_ = {8, 7, 6, 5, 4, 3, 2, 1};
    h_output_.resize(h_input_.size());

    runSort(h_input_.size());

    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(h_output_, expected);
}

TEST_F(SortTest, SingleElement) {
    h_input_ = {42};
    h_output_.resize(1);

    runSort(1);

    EXPECT_EQ(h_output_[0], 42);
}

TEST_F(SortTest, Duplicates) {
    h_input_ = {3, 1, 4, 1, 5, 9, 2, 6, 3, 3};
    h_output_.resize(h_input_.size());

    runSort(h_input_.size());

    for (size_t i = 1; i < h_output_.size(); ++i) {
        EXPECT_LE(h_output_[i-1], h_output_[i]);
    }
}
