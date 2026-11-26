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

// The padding launch covered the wrong range. padWithMaxKernel writes where
// `i >= size && i < paddedSize`, so it needs threads over the WHOLE padded array:
//
//     launcher.grid(detail::calc_grid_1d(paddedSize - size, {256, 1, 1}));
//     launcher.launch(padWithMaxKernel<T>, d_data.data(), size, paddedSize, maxVal);
//
// The grid was sized from the PAD COUNT, so the threads it created had indices 0..pad-1,
// every one of them below `size` -- the kernel's condition never held and not a single
// pad slot was written. Those slots kept whatever cudaMalloc returned, and the bitonic
// network then sorted them into the result.
//
// The other cases here use fewer than 256 elements, where the pad count is smaller than
// one block and the wrong grid still happened to cover enough indices. 1000 elements pad
// to 1024, a pad of 24.
TEST_F(SortTest, SortsASizeThatNeedsPadding) {
    size_t size = 1000;
    h_input_.resize(size);
    h_output_.resize(size);
    d_input_ = cuda::memory::Buffer<int>(size);
    d_output_ = cuda::memory::Buffer<int>(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<int>((i * 29 + 7) % 1000);
    }

    runSort(size);

    std::vector<int> expected = h_input_;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(h_output_, expected) << "the padded tail was left uninitialised and sorted into the result";
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

// bitonicSort launches one thread per element:
//
//     launcher.block({paddedSize, 1, 1});
//     launcher.grid({1, 1, 1});
//
// where paddedSize is the input rounded up to a power of two. A block cannot have
// more than 1024 threads, so any input above 1024 either fails to launch or is
// silently clamped depending on the launch path. Every case in this file used 1024
// or fewer -- exactly the boundary that hides it -- and the fuzz target caps its
// input at 1024 for the same reason.
//
// 2048 is the smallest power of two above the limit, so it exercises the failure
// with the smallest allocation.
TEST_F(SortTest, SortsMoreElementsThanABlockCanHold) {
    const size_t size = 4096;  // smallest input that pads above the 1024-thread limit
    h_input_.resize(size);
    h_output_.resize(size);
    d_input_ = cuda::memory::Buffer<int>(size);
    d_output_ = cuda::memory::Buffer<int>(size);

    for (size_t i = 0; i < size; ++i) {
        // A permutation, so a partially sorted result cannot pass by accident.
        h_input_[i] = static_cast<int>((i * 7919u) % size);
    }

    runSort(size);

    for (size_t i = 1; i < size; ++i) {
        ASSERT_LE(h_output_[i - 1], h_output_[i]) << "first inversion at index " << i;
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

// The launch grid for the compare-exchange pass divided by the block size twice:
//
//     size_t elementsPerBlock = blockSize * 2;                                   // 512
//     launcher.grid(calc_grid_1d((size + elementsPerBlock - 1) / elementsPerBlock, {blockSize, 1, 1}));
//
// The inner expression is already a BLOCK COUNT, and calc_grid_1d divides its argument by
// block.x again. For size = 1024 that is ceil(2 / 256) = 1 block, so 256 threads compared the
// first 512 elements and the remaining half was never touched -- and the function still
// returned normally. The existing cases used 9 and 100 elements, both under the 512 that one
// block can reach, which is why the bug survived them.
TEST_F(OddEvenSortTest, SortsMoreThanOneBlockCanCover) {
    size_t size = 1024;
    h_input_.resize(size);
    h_output_.resize(size);

    for (size_t i = 0; i < size; ++i) {
        h_input_[i] = static_cast<int>((i * 37 + 11) % 1024);
    }

    runAndDownload(size);

    std::vector<int> expected = h_input_;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(h_output_, expected) << "elements past the first 512 were never compared";
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
