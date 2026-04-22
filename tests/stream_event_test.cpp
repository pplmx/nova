#include <gtest/gtest.h>
#include "cuda/api/stream.h"
#include "cuda/device/error.h"
#include <vector>
#include <algorithm>
#include <numeric>

namespace {

void reset() {
    cudaDeviceReset();
}

}

class StreamTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(StreamTest, DefaultConstruction) {
    cuda::api::Stream stream;
    EXPECT_NE(stream.get(), nullptr);
}

TEST_F(StreamTest, CopyConstructionDeleted) {
    cuda::api::Stream stream;
    EXPECT_FALSE(std::is_copy_constructible_v<cuda::api::Stream>);
}

TEST_F(StreamTest, CopyAssignmentDeleted) {
    cuda::api::Stream stream;
    EXPECT_FALSE(std::is_copy_assignable_v<cuda::api::Stream>);
}

TEST_F(StreamTest, MoveConstruction) {
    cuda::api::Stream stream1;
    auto handle = stream1.get();

    cuda::api::Stream stream2(std::move(stream1));
    EXPECT_EQ(stream2.get(), handle);
}

TEST_F(StreamTest, MoveAssignment) {
    cuda::api::Stream stream1;
    cuda::api::Stream stream2;
    auto handle1 = stream1.get();

    stream2 = std::move(stream1);
    EXPECT_EQ(stream2.get(), handle1);
}

TEST_F(StreamTest, GetReturnsStreamHandle) {
    cuda::api::Stream stream;
    EXPECT_NE(stream.get(), nullptr);
    EXPECT_NE(*stream, nullptr);
}

TEST_F(StreamTest, SynchronizeVerifiesDataTransfer) {
    cuda::api::Stream stream;
    constexpr int N = 1000;
    std::vector<int> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 1);
    std::vector<int> h_output(N, 0);

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_input.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream.get()));

    stream.synchronize();

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_output, h_input);

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(StreamTest, QueryReturnsTrue) {
    cuda::api::Stream stream;
    EXPECT_TRUE(stream.query());
}

TEST_F(StreamTest, QueryAfterAsyncWork) {
    cuda::api::Stream stream;
    constexpr int N = 100;
    std::vector<int> h_data(N, 42);
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream.get()));

    while (!stream.query()) {
    }

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(StreamTest, MakeStream) {
    auto stream = cuda::api::make_stream();
    EXPECT_NE(stream->get(), nullptr);
}

class EventTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(EventTest, DefaultConstruction) {
    cuda::api::Event event;
    EXPECT_NE(event.get(), nullptr);
}

TEST_F(EventTest, CopyConstructionDeleted) {
    cuda::api::Event event;
    EXPECT_FALSE(std::is_copy_constructible_v<cuda::api::Event>);
}

TEST_F(EventTest, CopyAssignmentDeleted) {
    cuda::api::Event event;
    EXPECT_FALSE(std::is_copy_assignable_v<cuda::api::Event>);
}

TEST_F(EventTest, MoveConstruction) {
    cuda::api::Event event1;
    auto handle = event1.get();

    cuda::api::Event event2(std::move(event1));
    EXPECT_EQ(event2.get(), handle);
}

TEST_F(EventTest, MoveAssignment) {
    cuda::api::Event event1;
    cuda::api::Event event2;
    auto handle1 = event1.get();

    event2 = std::move(event1);
    EXPECT_EQ(event2.get(), handle1);
}

TEST_F(EventTest, GetReturnsEventHandle) {
    cuda::api::Event event;
    EXPECT_NE(event.get(), nullptr);
    EXPECT_NE(*event, nullptr);
}

TEST_F(EventTest, RecordOnStreamAndQuery) {
    cuda::api::Stream stream;
    cuda::api::Event event;

    constexpr int N = 100;
    std::vector<int> h_data(N, 42);
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream.get()));

    event.record(stream);
    stream.synchronize();

    EXPECT_TRUE(event.query());

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(EventTest, SynchronizeBlocksUntilComplete) {
    cuda::api::Event event;

    event.synchronize();

    EXPECT_TRUE(event.query());
}

TEST_F(EventTest, QueryAfterCreation) {
    cuda::api::Event event;
    EXPECT_TRUE(event.query());
}

TEST_F(EventTest, QueryAfterRecordWithDataTransfer) {
    cuda::api::Stream stream;
    cuda::api::Event start_event, end_event;

    constexpr int N = 1000;
    std::vector<int> h_data(N);
    std::iota(h_data.begin(), h_data.end(), 1);

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    start_event.record(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream.get()));
    end_event.record(stream);

    stream.synchronize();

    EXPECT_TRUE(end_event.query());

    std::vector<int> h_output(N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_output, h_data);

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(EventTest, ElapsedTimeIsNonNegative) {
    cuda::api::Stream stream;
    cuda::api::Event start_event, end_event;

    constexpr int N = 10000;
    std::vector<int> h_data(N, 42);
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    start_event.record(stream);
    for (int i = 0; i < 100; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream.get()));
    }
    end_event.record(stream);

    stream.synchronize();

    float elapsed = cuda::api::Event::elapsed_time(start_event, end_event);
    EXPECT_GE(elapsed, 0.0f);
    EXPECT_LT(elapsed, 10000.0f);

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(EventTest, ElapsedTimeAccurateForSequentialOps) {
    cuda::api::Stream stream;
    cuda::api::Event start_event, end_event;

    constexpr int N = 5000;
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    start_event.record(stream);
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, nullptr, 0, cudaMemcpyHostToDevice, stream.get()));
    }
    end_event.record(stream);
    stream.synchronize();

    float elapsed = cuda::api::Event::elapsed_time(start_event, end_event);
    EXPECT_GE(elapsed, 0.0f);

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(EventTest, MakeEvent) {
    auto event = cuda::api::make_event();
    EXPECT_NE(event->get(), nullptr);
}

class StreamEventIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(StreamEventIntegrationTest, MeasureKernelTimeAndVerifyData) {
    cuda::api::Stream stream;
    cuda::api::Event start_event, end_event;

    constexpr int N = 10000;
    std::vector<int> h_input(N, 1);
    std::vector<int> h_output(N, 0);

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    start_event.record(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, N * sizeof(int),
                             cudaMemcpyDeviceToDevice, stream.get()));
    end_event.record(stream);

    stream.synchronize();

    float elapsed = cuda::api::Event::elapsed_time(start_event, end_event);
    EXPECT_GE(elapsed, 0.0f);
    EXPECT_LT(elapsed, 1000.0f);

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_output, h_input);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST_F(StreamEventIntegrationTest, MultipleStreamsIndependenceWithDataVerification) {
    cuda::api::Stream stream1, stream2;

    constexpr int N = 1000;
    std::vector<int> h_data(N, 42);
    std::vector<int> h_result1(N, 0);
    std::vector<int> h_result2(N, 0);

    int *d_data1, *d_data2;
    CUDA_CHECK(cudaMalloc(&d_data1, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_data2, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_data1, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream1.get()));
    CUDA_CHECK(cudaMemcpyAsync(d_data2, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream2.get()));

    stream1.synchronize();
    stream2.synchronize();

    EXPECT_TRUE(stream1.query());
    EXPECT_TRUE(stream2.query());

    CUDA_CHECK(cudaMemcpy(h_result1.data(), d_data1, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result2.data(), d_data2, N * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result1, h_data);
    EXPECT_EQ(h_result2, h_data);

    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
}

TEST_F(StreamEventIntegrationTest, EventOrderingWithDataVerification) {
    cuda::api::Stream stream;
    cuda::api::Event events[3];
    constexpr int N = 100;
    std::vector<int> h_data(N, 1);
    std::vector<int> h_result(N, 0);

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    events[0].record(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream.get()));
    events[1].record(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream.get()));
    events[2].record(stream);

    stream.synchronize();

    float time01 = cuda::api::Event::elapsed_time(events[0], events[1]);
    float time12 = cuda::api::Event::elapsed_time(events[1], events[2]);
    float time02 = cuda::api::Event::elapsed_time(events[0], events[2]);

    EXPECT_GE(time01, 0.0f);
    EXPECT_GE(time12, 0.0f);
    EXPECT_GE(time02, 0.0f);
    EXPECT_LE(time01 + time12, time02 + 0.1f);

    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_result, h_data);

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(StreamEventIntegrationTest, EventTimingConsistency) {
    cuda::api::Stream stream;
    cuda::api::Event start, middle, end;

    constexpr int N = 5000;
    std::vector<int> h_data(N, 0);
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    start.record(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice, stream.get()));
    middle.record(stream);
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice, stream.get()));
    end.record(stream);
    stream.synchronize();

    float t1 = cuda::api::Event::elapsed_time(start, middle);
    float t2 = cuda::api::Event::elapsed_time(middle, end);
    float t_total = cuda::api::Event::elapsed_time(start, end);

    EXPECT_GE(t1, 0.0f);
    EXPECT_GE(t2, 0.0f);
    EXPECT_GE(t_total, 0.0f);
    EXPECT_LE(t1 + t2, t_total + 0.01f);

    CUDA_CHECK(cudaFree(d_data));
}

TEST_F(StreamEventIntegrationTest, StreamPriorityIndependence) {
    cuda::api::Stream stream1, stream2;

    constexpr int N = 2000;
    std::vector<int> h_data(N, 7);
    std::vector<int> h_result1(N, 0);
    std::vector<int> h_result2(N, 0);

    int *d1, *d2;
    CUDA_CHECK(cudaMalloc(&d1, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d2, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d1, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream1.get()));
    CUDA_CHECK(cudaMemcpyAsync(d2, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice, stream2.get()));

    stream1.synchronize();
    stream2.synchronize();

    CUDA_CHECK(cudaMemcpy(h_result1.data(), d1, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result2.data(), d2, N * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result1, h_data);
    EXPECT_EQ(h_result2, h_data);

    CUDA_CHECK(cudaFree(d1));
    CUDA_CHECK(cudaFree(d2));
}
