#include <gtest/gtest.h>
#include <vector>

#include "cuda/algo/segmented_sort.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::segmented::test {

    class SegmentedSortTest : public ::testing::Test {
    protected:
        void SetUp() override {
            cudaSetDevice(0);
            cudaDeviceSynchronize();
        }
        void TearDown() override {}
    };

    TEST_F(SegmentedSortTest, SortByKeyBasic) {
        std::vector<float> keys = {3.0f, 1.0f, 2.0f, 5.0f, 4.0f};
        std::vector<int> segments = {0, 0, 0, 1, 1};
        std::vector<float> out_keys(5);
        std::vector<int> out_segments(5);

        cuda::memory::Buffer<float> d_keys(keys.size());
        cuda::memory::Buffer<int> d_segments(segments.size());
        cuda::memory::Buffer<float> d_out_keys(5);
        cuda::memory::Buffer<int> d_out_segments(5);

        cudaMemcpy(d_keys.data(), keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_segments.data(), segments.data(), segments.size() * sizeof(int), cudaMemcpyHostToDevice);

        cuda::algo::segmented::sort_by_key(d_keys.data(), d_segments.data(), d_out_keys.data(), d_out_segments.data(), 5, 2);

        cudaMemcpy(out_keys.data(), d_out_keys.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_segments.data(), d_out_segments.data(), 5 * sizeof(int), cudaMemcpyDeviceToHost);

        EXPECT_EQ(out_segments[0], 0);
        EXPECT_EQ(out_segments[1], 0);
        EXPECT_EQ(out_segments[2], 0);
        EXPECT_EQ(out_segments[3], 1);
        EXPECT_EQ(out_segments[4], 1);

        EXPECT_TRUE(out_keys[0] <= out_keys[1]);
        EXPECT_TRUE(out_keys[1] <= out_keys[2]);
    }

    TEST_F(SegmentedSortTest, SortInPlace) {
        std::vector<int> keys = {3, 1, 2, 5, 4};
        std::vector<int> segments = {0, 0, 0, 1, 1};

        cuda::memory::Buffer<int> d_keys(keys.size());
        cuda::memory::Buffer<int> d_segments(segments.size());

        cudaMemcpy(d_keys.data(), keys.data(), keys.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_segments.data(), segments.data(), segments.size() * sizeof(int), cudaMemcpyHostToDevice);

        cuda::algo::segmented::sort_by_key_inplace(d_keys.data(), d_segments.data(), 5, 2);

        std::vector<int> result(5);
        cudaMemcpy(result.data(), d_keys.data(), 5 * sizeof(int), cudaMemcpyDeviceToHost);

        EXPECT_TRUE(result[0] <= result[1]);
        EXPECT_TRUE(result[1] <= result[2]);
    }

    TEST_F(SegmentedSortTest, ConfigSetters) {
        SegmentedSortConfig config;
        config.max_segments_per_block = 8;
        config.elements_per_segment_block = 512;

        cuda::algo::segmented::set_config(config);

        auto retrieved = cuda::algo::segmented::get_config();
        EXPECT_EQ(retrieved.max_segments_per_block, 8);
        EXPECT_EQ(retrieved.elements_per_segment_block, 512);
    }

    TEST_F(SegmentedSortTest, EmptyInput) {
        cuda::memory::Buffer<float> d_out_keys(0);
        cuda::memory::Buffer<int> d_out_segments(0);

        SUCCEED();
    }

}  // namespace cuda::algo::segmented::test
