#include <gtest/gtest.h>
#include "cuda/algo/sort.h"
#include "cuda/memory/buffer.h"
#include <algorithm>
#include <random>
#include <vector>

using cuda::memory::Buffer;
using cuda::sort::Order;
using cuda::sort::radix_sort_keys;
using cuda::sort::radix_sort_pair;
using cuda::sort::select_top_k;
using cuda::sort::binary_search;
using cuda::sort::BinarySearchResult;
using cuda::sort::SearchResult;

namespace {

bool isSorted(const std::vector<int>& arr, Order order) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (order == Order::Ascending) {
            if (arr[i - 1] > arr[i]) return false;
        } else {
            if (arr[i - 1] < arr[i]) return false;
        }
    }
    return true;
}

bool isSorted(const std::vector<float>& arr, Order order) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (order == Order::Ascending) {
            if (arr[i - 1] > arr[i]) return false;
        } else {
            if (arr[i - 1] < arr[i]) return false;
        }
    }
    return true;
}

}  // namespace

class RadixSortTest : public ::testing::Test {
protected:
    std::vector<int> h_keys_;
    cuda::memory::Buffer<int> d_keys_;
    size_t size_ = 1024;

    void SetUp() override {
        h_keys_.resize(size_);
        d_keys_ = cuda::memory::Buffer<int>(size_);
    }

    void generateRandomKeys(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 10000);

        h_keys_.resize(size);
        for (size_t i = 0; i < size; ++i) {
            h_keys_[i] = dis(gen);
        }
    }
};

TEST_F(RadixSortTest, AscendingSort) {
    h_keys_ = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), d_keys_.size(), Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Ascending));
}

TEST_F(RadixSortTest, DescendingSort) {
    h_keys_ = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), d_keys_.size(), Order::Descending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Descending));
}

TEST_F(RadixSortTest, AlreadySorted) {
    h_keys_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), d_keys_.size(), Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Ascending));
}

TEST_F(RadixSortTest, ReverseSorted) {
    h_keys_ = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), d_keys_.size(), Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Ascending));
}

TEST_F(RadixSortTest, SingleElement) {
    h_keys_ = {42};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), 1, Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_EQ(h_keys_[0], 42);
}

TEST_F(RadixSortTest, Duplicates) {
    h_keys_ = {3, 1, 4, 1, 5, 9, 2, 6, 3, 3};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), d_keys_.size(), Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Ascending));
}

TEST_F(RadixSortTest, AllSame) {
    h_keys_ = std::vector<int>(100, 42);
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys_.data(), d_keys_.size(), Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());

    for (const auto& val : h_keys_) {
        EXPECT_EQ(val, 42) << "All values should be 42 after sorting";
    }
}

TEST_F(RadixSortTest, LargeArray) {
    generateRandomKeys(100000);
    cuda::memory::Buffer<int> d_keys(h_keys_.size());
    d_keys.copy_from(h_keys_.data(), h_keys_.size());

    radix_sort_keys(d_keys.data(), d_keys.size(), Order::Ascending);

    d_keys.copy_to(h_keys_.data(), h_keys_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Ascending));
}

class KeyValueSortTest : public ::testing::Test {
protected:
    std::vector<int> h_keys_;
    std::vector<float> h_values_;
    cuda::memory::Buffer<int> d_keys_;
    cuda::memory::Buffer<float> d_values_;
    size_t size_ = 1024;

    void SetUp() override {
        h_keys_.resize(size_);
        h_values_.resize(size_);
        d_keys_ = cuda::memory::Buffer<int>(size_);
        d_values_ = cuda::memory::Buffer<float>(size_);
    }
};

TEST_F(KeyValueSortTest, SortPairsAscending) {
    h_keys_ = {5, 2, 8, 1, 9};
    h_values_ = {50.0f, 20.0f, 80.0f, 10.0f, 90.0f};
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());
    d_values_.copy_from(h_values_.data(), h_values_.size());

    radix_sort_pair(d_keys_.data(), d_values_.data(), d_keys_.size(), Order::Ascending);

    d_keys_.copy_to(h_keys_.data(), h_keys_.size());
    d_values_.copy_to(h_values_.data(), h_values_.size());

    EXPECT_TRUE(isSorted(h_keys_, Order::Ascending));
    for (size_t i = 1; i < h_keys_.size(); ++i) {
        if (h_keys_[i] == h_keys_[i - 1]) {
            continue;
        }
        EXPECT_LT(h_keys_[i - 1], h_keys_[i]);
    }
}

class TopKTest : public ::testing::Test {
protected:
    std::vector<float> h_keys_;
    std::vector<float> h_values_;
    cuda::memory::Buffer<float> d_keys_;
    cuda::memory::Buffer<float> d_values_;
};

TEST_F(TopKTest, SelectTopKDescending) {
    h_keys_ = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 7.0f, 4.0f, 6.0f, 10.0f};
    h_values_ = {50.0f, 20.0f, 80.0f, 10.0f, 90.0f, 30.0f, 70.0f, 40.0f, 60.0f, 100.0f};

    d_keys_ = cuda::memory::Buffer<float>(h_keys_.size());
    d_values_ = cuda::memory::Buffer<float>(h_values_.size());
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());
    d_values_.copy_from(h_values_.data(), h_values_.size());

    size_t k = 3;
    auto result = select_top_k(d_keys_.data(), d_values_.data(), h_keys_.size(), k, Order::Descending);

    ASSERT_EQ(result.actual_k, k);

    std::vector<float> result_keys(k);
    result.keys.copy_to(result_keys.data(), k);

    EXPECT_EQ(result_keys[0], 10.0f);
    EXPECT_EQ(result_keys[1], 9.0f);
    EXPECT_EQ(result_keys[2], 8.0f);
}

TEST_F(TopKTest, SelectTopKAscending) {
    h_keys_ = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f};
    h_values_ = {50.0f, 20.0f, 80.0f, 10.0f, 90.0f};

    d_keys_ = cuda::memory::Buffer<float>(h_keys_.size());
    d_values_ = cuda::memory::Buffer<float>(h_values_.size());
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());
    d_values_.copy_from(h_values_.data(), h_values_.size());

    size_t k = 2;
    auto result = select_top_k(d_keys_.data(), d_values_.data(), h_keys_.size(), k, Order::Ascending);

    ASSERT_EQ(result.actual_k, k);

    std::vector<float> result_keys(k);
    result.keys.copy_to(result_keys.data(), k);

    EXPECT_EQ(result_keys[0], 1.0f);
    EXPECT_EQ(result_keys[1], 2.0f);
}

TEST_F(TopKTest, KGreaterThanSize) {
    h_keys_ = {5.0f, 2.0f, 8.0f};
    h_values_ = {50.0f, 20.0f, 80.0f};

    d_keys_ = cuda::memory::Buffer<float>(h_keys_.size());
    d_values_ = cuda::memory::Buffer<float>(h_values_.size());
    d_keys_.copy_from(h_keys_.data(), h_keys_.size());
    d_values_.copy_from(h_values_.data(), h_values_.size());

    auto result = select_top_k(d_keys_.data(), d_values_.data(), h_keys_.size(), 10, Order::Descending);

    EXPECT_EQ(result.actual_k, 3);
}

class BinarySearchTest : public ::testing::Test {
protected:
    std::vector<int> h_sorted_;
    cuda::memory::Buffer<int> d_sorted_;
};

TEST_F(BinarySearchTest, Found) {
    h_sorted_ = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    d_sorted_.copy_from(h_sorted_.data(), h_sorted_.size());

    auto result = binary_search(d_sorted_.data(), d_sorted_.size(), 7);

    EXPECT_EQ(result.status, SearchResult::Found);
    EXPECT_EQ(result.index, 3);
}

TEST_F(BinarySearchTest, NotFound) {
    h_sorted_ = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    d_sorted_.copy_from(h_sorted_.data(), h_sorted_.size());

    auto result = binary_search(d_sorted_.data(), d_sorted_.size(), 8);

    EXPECT_EQ(result.status, SearchResult::NotFound);
}

TEST_F(BinarySearchTest, FirstElement) {
    h_sorted_ = {1, 3, 5, 7, 9};
    d_sorted_.copy_from(h_sorted_.data(), h_sorted_.size());

    auto result = binary_search(d_sorted_.data(), d_sorted_.size(), 1);

    EXPECT_EQ(result.status, SearchResult::Found);
    EXPECT_EQ(result.index, 0);
}

TEST_F(BinarySearchTest, LastElement) {
    h_sorted_ = {1, 3, 5, 7, 9};
    d_sorted_.copy_from(h_sorted_.data(), h_sorted_.size());

    auto result = binary_search(d_sorted_.data(), d_sorted_.size(), 9);

    EXPECT_EQ(result.status, SearchResult::Found);
    EXPECT_EQ(result.index, 4);
}

TEST_F(BinarySearchTest, SingleElementFound) {
    h_sorted_ = {42};
    d_sorted_.copy_from(h_sorted_.data(), 1);

    auto result = binary_search(d_sorted_.data(), 1, 42);

    EXPECT_EQ(result.status, SearchResult::Found);
    EXPECT_EQ(result.index, 0);
}

TEST_F(BinarySearchTest, SingleElementNotFound) {
    h_sorted_ = {42};
    d_sorted_.copy_from(h_sorted_.data(), 1);

    auto result = binary_search(d_sorted_.data(), 1, 43);

    EXPECT_EQ(result.status, SearchResult::NotFound);
}

TEST_F(BinarySearchTest, EmptyArray) {
    d_sorted_ = cuda::memory::Buffer<int>(0);

    auto result = binary_search(d_sorted_.data(), 0, 1);

    EXPECT_EQ(result.status, SearchResult::NotFound);
}

TEST_F(BinarySearchTest, FloatArray) {
    std::vector<float> h_sorted_float = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    cuda::memory::Buffer<float> d_sorted_float;
    d_sorted_float.copy_from(h_sorted_float.data(), h_sorted_float.size());

    auto result = binary_search(d_sorted_float.data(), d_sorted_float.size(), 3.5f);

    EXPECT_EQ(result.status, SearchResult::Found);
    EXPECT_EQ(result.index, 2);
}
