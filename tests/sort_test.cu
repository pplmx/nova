#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include "parallel/sort.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include <algorithm>
#include <numeric>

using cuda::memory::Buffer;

namespace {

    bool isPermutation(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        std::vector<int> sorted_a = a, sorted_b = b;
        std::sort(sorted_a.begin(), sorted_a.end());
        std::sort(sorted_b.begin(), sorted_b.end());
        return sorted_a == sorted_b;
    }

}  // namespace

class SortTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    std::vector<int> h_output_;
    cuda::memory::Buffer<int> d_input_;
    cuda::memory::Buffer<int> d_output_;

    void SetUp() override {
        h_input_.resize(size_);
        h_output_.resize(size_);
        d_input_ = cuda::memory::Buffer<int>(size_);
        d_output_ = cuda::memory::Buffer<int>(size_);
    }

    void runSort(size_t size) {
        d_input_.copy_from(h_input_.data(), size);
        cuda::parallel::bitonicSort(d_input_, d_output_, size);
        d_output_.copy_to(h_output_.data(), size);
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
        EXPECT_LE(h_output_[i - 1], h_output_[i]);
    }
    EXPECT_TRUE(isPermutation(h_input_, h_output_));
}

TEST_F(SortTest, LargeArray) {
    size_t size = 512;
    h_input_.resize(size);
    h_output_.resize(size);
    d_input_ = cuda::memory::Buffer<int>(size);
    d_output_ = cuda::memory::Buffer<int>(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<int>(size - i);
    }

    runSort(size);

    for (size_t i = 1; i < size; ++i) {
        EXPECT_LE(h_output_[i - 1], h_output_[i]);
    }
    EXPECT_TRUE(isPermutation(h_input_, h_output_));
}

TEST_F(SortTest, AllSame) {
    size_t size = 100;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = 42;
    }

    runSort(size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(h_output_[i], 42);
    }
}

class OddEvenSortTest : public ::testing::Test {
protected:
    std::vector<int> h_input_, h_output_;
    cuda::memory::Buffer<int> d_input_;
    cuda::memory::Buffer<int> d_output_;

    void SetUp() override {
        d_input_ = cuda::memory::Buffer<int>(1024);
        d_output_ = cuda::memory::Buffer<int>(1024);
    }

    void runAndDownload(size_t size) {
        d_input_.copy_from(h_input_.data(), size);
        cuda::parallel::oddEvenSort(d_input_, d_output_, size);
        d_output_.copy_to(h_output_.data(), size);
    }
};

TEST_F(OddEvenSortTest, RandomArray) {
    h_input_ = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    h_output_.resize(h_input_.size());

    runAndDownload(h_input_.size());

    std::vector<int> expected = h_input_;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(h_output_, expected);
}

TEST_F(OddEvenSortTest, LargeArray) {
    size_t size = 100;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<int>((i * 17 + 31) % 100);
    }

    runAndDownload(size);

    for (size_t i = 1; i < size; ++i) {
        EXPECT_LE(h_output_[i - 1], h_output_[i]);
    }
    EXPECT_TRUE(isPermutation(h_input_, h_output_));
}

TEST_F(OddEvenSortTest, AllSame) {
    size_t size = 100;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = 42;
    }

    runAndDownload(size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(h_output_[i], 42);
    }
}
