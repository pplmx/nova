#include <gtest/gtest.h>
#include "cuda/async/stream_manager.h"

class StreamManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(StreamManagerTest, GetStreamReturnsValidHandle) {
    cuda::async::StreamManager manager;

    cudaStream_t stream = manager.get_stream(0);

    EXPECT_NE(stream, nullptr);
}

TEST_F(StreamManagerTest, GetStreamPriorityRangeReturnsValidRange) {
    auto range = cuda::async::get_stream_priority_range();

    int range_size = range.max_priority - range.min_priority;
    EXPECT_TRUE(range_size <= 0);
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
