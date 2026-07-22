#include <gtest/gtest.h>
#include <vector>

#include "cuda/algo/sample_sort.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::sample_sort::test {

class SampleSortTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SampleSortTest, SortAscending) {
    std::vector<int> input = {5, 3, 1, 4, 2};
    std::vector<int> expected = {1, 2, 3, 4, 5};

    cuda::memory::Buffer<int> d_input(input.size());
    cudaMemcpy(d_input.data(), input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);

    auto result = cuda::algo::sample_sort::sort(d_input.data(), input.size(),
                                                 cuda::algo::sample_sort::Order::Ascending);

    std::vector<int> output(result.actual_count);
    cudaMemcpy(output.data(), result.data.data(), result.actual_count * sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(output, expected);
}

TEST_F(SampleSortTest, SortDescending) {
    std::vector<float> input = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    std::vector<float> expected = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    cuda::memory::Buffer<float> d_input(input.size());
    cudaMemcpy(d_input.data(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    auto result = cuda::algo::sample_sort::sort(d_input.data(), input.size(),
                                                 cuda::algo::sample_sort::Order::Descending);

    std::vector<float> output(result.actual_count);
    cudaMemcpy(output.data(), result.data.data(), result.actual_count * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(output, expected);
}

TEST_F(SampleSortTest, SortLargeDataset) {
    const size_t count = 100000;
    std::vector<double> input(count);

    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<double>(count - i);
    }

    cuda::memory::Buffer<double> d_input(count);
    cudaMemcpy(d_input.data(), input.data(), count * sizeof(double), cudaMemcpyHostToDevice);

    auto result = cuda::algo::sample_sort::sort_large_dataset(d_input.data(), count,
                                                               cuda::algo::sample_sort::Order::Ascending,
                                                               1024);

    std::vector<double> output(count);
    cudaMemcpy(output.data(), result.data.data(), count * sizeof(double), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result.actual_count, count);
    for (size_t i = 1; i < count; ++i) {
        EXPECT_LE(output[i - 1], output[i]);
    }
}

TEST_F(SampleSortTest, SortInPlace) {
    std::vector<int> input = {5, 3, 1, 4, 2};
    std::vector<int> expected = {1, 2, 3, 4, 5};

    cuda::memory::Buffer<int> d_input(input.size());
    cudaMemcpy(d_input.data(), input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);

    cuda::algo::sample_sort::sort_inplace(d_input.data(), input.size(),
                                           cuda::algo::sample_sort::Order::Ascending);

    std::vector<int> output(input.size());
    cudaMemcpy(output.data(), d_input.data(), input.size() * sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(output, expected);
}

TEST_F(SampleSortTest, ConfigSetters) {
    SampleSortConfig config;
    config.threshold_large_dataset = 1000000;
    config.default_sample_rate = 2048;

    cuda::algo::sample_sort::set_config(config);

    auto retrieved = cuda::algo::sample_sort::get_config();
    EXPECT_EQ(retrieved.threshold_large_dataset, 1000000);
    EXPECT_EQ(retrieved.default_sample_rate, 2048);
}

}  // namespace cuda::algo::sample_sort::test
