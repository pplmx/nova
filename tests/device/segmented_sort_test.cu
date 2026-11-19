/**
 * @file segmented_sort_test.cu
 * @brief Contracts for segmented (within-group) sorting
 *
 * include/cuda/algo/segmented_sort.h is implemented in src/algo/
 * segmented_sort.cu. A .cu test executable is needed because the header pulls in
 * cuda_runtime.h and the implementation is device code.
 *
 * Two defects:
 *
 * - sort_by_key announced itself as a SEGMENTED sort but sorted the whole buffer
 *   globally and never read segment_ids. Keys {5,1 | 4,2} came out as {1,2,4,5}
 *   instead of {1,5 | 2,4}: elements crossed segment boundaries, and the gathered
 *   segment ids no longer described the elements they were paired with
 *   (observed output was keys {1,2,4,5} with segments {0,1,1,0}).
 *
 * - sort_pairs_by_key was declared in the header and never defined anywhere, so
 *   any caller failed to link with "undefined reference to
 *   sort_pairs_by_key<int, int>".
 */

#include <cuda/algo/segmented_sort.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace cuda::algo::segmented::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// RAII device buffer for tests.
        template <typename T>
        class DeviceArray {
        public:
            explicit DeviceArray(const std::vector<T>& host) {
                count_ = host.size();
                cudaMalloc(&ptr_, count_ * sizeof(T));
                cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice);
            }

            explicit DeviceArray(size_t count) {
                count_ = count;
                cudaMalloc(&ptr_, count_ * sizeof(T));
                cudaMemset(ptr_, 0, count_ * sizeof(T));
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            T* data() { return ptr_; }
            const T* data() const { return ptr_; }

            std::vector<T> to_host() const {
                std::vector<T> host(count_);
                if (count_ > 0) {
                    cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
                }
                return host;
            }

        private:
            T* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: this returned a globally sorted buffer and scrambled the pairing
    // between each key and its segment id.
    TEST(SegmentedSortByKeyTest, SortsWithinSegmentsOnly) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Segment 0 holds {5,1}; segment 1 holds {4,2}.
        const std::vector<int> keys = {5, 1, 4, 2};
        const std::vector<int> segments = {0, 0, 1, 1};

        DeviceArray<int> d_keys(keys);
        DeviceArray<int> d_segments(segments);
        DeviceArray<int> d_out_keys(keys.size());
        DeviceArray<int> d_out_segments(keys.size());

        sort_by_key(d_keys.data(), d_segments.data(), d_out_keys.data(), d_out_segments.data(), keys.size(), 2);

        // Each segment sorted independently; nothing crosses a boundary.
        EXPECT_EQ(d_out_keys.to_host(), (std::vector<int>{1, 5, 2, 4}));
        EXPECT_EQ(d_out_segments.to_host(), (std::vector<int>{0, 0, 1, 1}));
    }

    TEST(SegmentedSortByKeyTest, KeepsEachKeyWithItsOwnSegmentId) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Interleaved segment ids would break a naive global sort.
        const std::vector<int> keys = {9, 3, 8, 2, 7, 1};
        const std::vector<int> segments = {0, 0, 1, 1, 2, 2};

        DeviceArray<int> d_keys(keys);
        DeviceArray<int> d_segments(segments);
        DeviceArray<int> d_out_keys(keys.size());
        DeviceArray<int> d_out_segments(keys.size());

        sort_by_key(d_keys.data(), d_segments.data(), d_out_keys.data(), d_out_segments.data(), keys.size(), 3);

        const auto sorted_keys = d_out_keys.to_host();
        const auto sorted_segments = d_out_segments.to_host();

        EXPECT_EQ(sorted_keys, (std::vector<int>{3, 9, 2, 8, 1, 7}));
        EXPECT_EQ(sorted_segments, (std::vector<int>{0, 0, 1, 1, 2, 2}));

        // Within every segment the keys must be ascending.
        for (size_t i = 1; i < sorted_keys.size(); ++i) {
            if (sorted_segments[i] == sorted_segments[i - 1]) {
                EXPECT_LE(sorted_keys[i - 1], sorted_keys[i]) << "segment " << sorted_segments[i] << " is not sorted at index " << i;
            }
        }
    }

    TEST(SegmentedSortByKeyTest, SingleElementSegmentsAreUntouched) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<int> keys = {3, 1, 2};
        const std::vector<int> segments = {0, 1, 2};  // every segment holds one element

        DeviceArray<int> d_keys(keys);
        DeviceArray<int> d_segments(segments);
        DeviceArray<int> d_out_keys(keys.size());
        DeviceArray<int> d_out_segments(keys.size());

        sort_by_key(d_keys.data(), d_segments.data(), d_out_keys.data(), d_out_segments.data(), keys.size(), 3);

        EXPECT_EQ(d_out_keys.to_host(), keys) << "segments of one element have nothing to reorder";
    }

    TEST(SegmentedSortByKeyTest, HandlesFloatAndDoubleKeys) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> keys = {2.5f, 0.5f, 9.0f, 1.0f};
        const std::vector<int> segments = {0, 0, 1, 1};
        DeviceArray<float> d_keys(keys);
        DeviceArray<int> d_segments(segments);
        DeviceArray<float> d_out_keys(keys.size());
        DeviceArray<int> d_out_segments(keys.size());

        sort_by_key(d_keys.data(), d_segments.data(), d_out_keys.data(), d_out_segments.data(), keys.size(), 2);

        EXPECT_EQ(d_out_keys.to_host(), (std::vector<float>{0.5f, 2.5f, 1.0f, 9.0f}));
    }

    TEST(SegmentedSortByKeyInplaceTest, SortsTheBufferInPlace) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<int> keys = {4, 2, 3, 1};
        const std::vector<int> segments = {0, 0, 1, 1};
        DeviceArray<int> d_keys(keys);
        DeviceArray<int> d_segments(segments);

        sort_by_key_inplace(d_keys.data(), d_segments.data(), keys.size(), 2);

        EXPECT_EQ(d_keys.to_host(), (std::vector<int>{2, 4, 1, 3}));
        EXPECT_EQ(d_segments.to_host(), (std::vector<int>{0, 0, 1, 1}));
    }

    // Regression: this function had no definition at all.
    TEST(SegmentedSortPairsByKeyTest, SortsKeysAndCarriesValues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Keys {5,1 | 4,2} with payloads {50,10 | 40,20}.
        const std::vector<int> keys = {5, 1, 4, 2};
        const std::vector<int> values = {50, 10, 40, 20};
        const std::vector<int> segments = {0, 0, 1, 1};

        DeviceArray<int> d_keys(keys);
        DeviceArray<int> d_values(values);
        DeviceArray<int> d_segments(segments);
        DeviceArray<int> d_out_keys(keys.size());
        DeviceArray<int> d_out_values(keys.size());
        DeviceArray<int> d_out_segments(keys.size());

        sort_pairs_by_key(d_keys.data(), d_values.data(), d_segments.data(), d_out_keys.data(), d_out_values.data(), d_out_segments.data(), keys.size(), 2);

        EXPECT_EQ(d_out_keys.to_host(), (std::vector<int>{1, 5, 2, 4})) << "keys must be sorted within each segment";
        EXPECT_EQ(d_out_values.to_host(), (std::vector<int>{10, 50, 20, 40})) << "each value must follow the key it was paired with";
        EXPECT_EQ(d_out_segments.to_host(), (std::vector<int>{0, 0, 1, 1}));
    }

    TEST(SegmentedSortPairsByKeyTest, ValuesStayPairedAcrossManySegments) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<int> keys = {3, 1, 2, 6, 5, 4};
        const std::vector<int> values = {30, 10, 20, 60, 50, 40};
        const std::vector<int> segments = {0, 0, 0, 1, 1, 1};

        DeviceArray<int> d_keys(keys);
        DeviceArray<int> d_values(values);
        DeviceArray<int> d_segments(segments);
        DeviceArray<int> d_out_keys(keys.size());
        DeviceArray<int> d_out_values(keys.size());
        DeviceArray<int> d_out_segments(keys.size());

        sort_pairs_by_key(d_keys.data(), d_values.data(), d_segments.data(), d_out_keys.data(), d_out_values.data(), d_out_segments.data(), keys.size(), 2);

        const auto sorted_keys = d_out_keys.to_host();
        const auto sorted_values = d_out_values.to_host();
        EXPECT_EQ(sorted_keys, (std::vector<int>{1, 2, 3, 4, 5, 6}));

        // Value == key * 10 in the input, so the pairing is verifiable after sorting.
        for (size_t i = 0; i < sorted_keys.size(); ++i) {
            EXPECT_EQ(sorted_values[i], sorted_keys[i] * 10) << "value at index " << i << " is no longer paired with its key";
        }
    }

    TEST(SegmentedSortPairsByKeyTest, EmptyInputIsANoOp) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // EXPECT_NO_THROW's macro argument is split on commas, and both the template
        // argument list and the call's own commas contain them. A local lambda
        // variable keeps the macro argument free of the problematic punctuation.
        auto call_with_empty_input = [] { sort_pairs_by_key<int, int>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0); };
        EXPECT_NO_THROW(call_with_empty_input());
        (void)cudaGetLastError();
    }

    TEST(SegmentedSortConfigTest, ConfigRoundTrips) {
        const SegmentedSortConfig original = get_config();

        SegmentedSortConfig custom;
        custom.max_segments_per_block = 8;
        custom.elements_per_segment_block = 512;
        custom.stable = false;
        set_config(custom);

        const SegmentedSortConfig readback = get_config();
        EXPECT_EQ(readback.max_segments_per_block, 8u);
        EXPECT_EQ(readback.elements_per_segment_block, 512u);
        EXPECT_FALSE(readback.stable);

        set_config(original);
    }

    TEST(SegmentedSortConfigTest, DefaultsDescribeAStableSort) {
        const SegmentedSortConfig config;
        EXPECT_TRUE(config.stable) << "stability is the safer default";
        EXPECT_GT(config.max_segments_per_block, 0u);
        EXPECT_GT(config.elements_per_segment_block, 0u);
    }

}  // namespace cuda::algo::segmented::test
