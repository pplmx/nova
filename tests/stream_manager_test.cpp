#include <gtest/gtest.h>
#include "cuda/async/stream_manager.h"

class StreamManagerTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(StreamManagerTest, GetStreamReturnsValidHandle) {
    cuda::async::StreamManager manager;

    cudaStream_t stream = manager.get_stream(0);

    EXPECT_NE(stream, nullptr);
}

TEST_F(StreamManagerTest, GetStreamPriorityRangeReturnsValidRange) {
    auto range = cuda::async::get_stream_priority_range();

    // std::clamp(v, lo, hi) requires lo <= hi, so the bounds have to be ordered that way. This case
    // used to assert `max_priority - min_priority <= 0`, i.e. the opposite: the accessor forwarded
    // cudaDeviceGetStreamPriorityRange's two out parameters straight into a struct named min/max,
    // but that call reports the *least* priority first, and it is the numerically larger value.
    // Measured: min_priority = 0, max_priority = -5, so get_stream()'s std::clamp had lo > hi and
    // its precondition was violated.
    EXPECT_LE(range.min_priority, range.max_priority) << "std::clamp needs lo <= hi; the accessor must order the bounds, not mirror CUDA's argument order";
    EXPECT_LE(range.min_priority, 0);
    EXPECT_LE(range.max_priority, 0);
}

// With the bounds ordered correctly, clamping is a no-op for in-range priorities and pulls
// out-of-range ones inside. Before the fix the bounds were inverted, so std::clamp's precondition was
// violated and the request was passed through untested -- an extreme priority reached the driver
// unchanged instead of being clamped.
TEST_F(StreamManagerTest, GetStreamClampsAPriorityOutsideTheRange) {
    auto range = cuda::async::get_stream_priority_range();
    ASSERT_LE(range.min_priority, range.max_priority);

    cuda::async::StreamManager manager;

    // Wildly out of range in both directions; both must produce a usable stream rather than
    // forwarding the value.
    EXPECT_NE(manager.get_stream(range.min_priority - 100), nullptr);
    EXPECT_NE(manager.get_stream(range.max_priority + 100), nullptr);
    EXPECT_NE(manager.get_stream(range.min_priority), nullptr);
    EXPECT_NE(manager.get_stream(range.max_priority), nullptr);
    CUDA_CHECK(cudaGetLastError());
}

TEST_F(StreamManagerTest, InitializeCreatesMultipleStreams) {
    cuda::async::StreamManager manager;

    manager.initialize(4);

    EXPECT_GE(manager.num_streams(), 4);
}

TEST_F(StreamManagerTest, SynchronizeAllCompletes) {
    cuda::async::StreamManager manager;
    manager.initialize(2);

    EXPECT_NO_THROW(manager.synchronize_all());
}

TEST_F(StreamManagerTest, GetHighAndLowPriorityStreams) {
    cuda::async::StreamManager manager;

    cudaStream_t high = manager.get_high_priority_stream();
    cudaStream_t low = manager.get_low_priority_stream();

    EXPECT_NE(high, nullptr);
    EXPECT_NE(low, nullptr);
}

TEST_F(StreamManagerTest, GetSameStreamTwice) {
    cuda::async::StreamManager manager;

    cudaStream_t stream1 = manager.get_stream(0);
    cudaStream_t stream2 = manager.get_stream(0);

    EXPECT_EQ(stream1, stream2);
}

TEST_F(StreamManagerTest, QueryAllReturnsResults) {
    cuda::async::StreamManager manager;
    manager.initialize(2);

    auto results = manager.query_all();

    EXPECT_EQ(results.size(), manager.num_streams());
}

TEST_F(StreamManagerTest, GlobalStreamManagerReturnsSingleton) {
    auto& global1 = cuda::async::global_stream_manager();
    auto& global2 = cuda::async::global_stream_manager();

    EXPECT_EQ(&global1, &global2);
}
