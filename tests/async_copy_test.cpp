#include <gtest/gtest.h>
#include "cuda/async/async_copy.h"
#include "cuda/async/pinned_memory.h"
#include "cuda/stream/stream.h"

class AsyncCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(AsyncCopyTest, AsyncCopyH2DCopiesCorrectly) {
    constexpr size_t N = 1024;

    std::vector<int> host_data(N);
    for (int i = 0; i < static_cast<int>(N); ++i) {
        host_data[i] = i;
    }

    int* d_ptr;
    cudaMalloc(&d_ptr, N * sizeof(int));

    cuda::stream::Stream stream;
    cuda::async::async_copy_h2d(d_ptr, host_data.data(), N * sizeof(int), stream.get());
    stream.synchronize();

    std::vector<int> result(N);
    cudaMemcpy(result.data(), d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < static_cast<int>(N); ++i) {
        EXPECT_EQ(result[i], i);
    }

    cudaFree(d_ptr);
}

TEST_F(AsyncCopyTest, AsyncCopyD2HCopiesCorrectly) {
    constexpr size_t N = 1024;

    std::vector<int> host_expected(N);
    for (int i = 0; i < static_cast<int>(N); ++i) {
        host_expected[i] = i * 2;
    }

    int* d_ptr;
    cudaMalloc(&d_ptr, N * sizeof(int));
    cudaMemcpy(d_ptr, host_expected.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> host_result(N);
    cuda::stream::Stream stream;
    cuda::async::async_copy_d2h(host_result.data(), d_ptr, N * sizeof(int), stream.get());
    stream.synchronize();

    for (int i = 0; i < static_cast<int>(N); ++i) {
        EXPECT_EQ(host_result[i], host_expected[i]);
    }

    cudaFree(d_ptr);
}

TEST_F(AsyncCopyTest, AsyncCopyD2DCopiesCorrectly) {
    constexpr size_t N = 1024;

    std::vector<int> src_data(N);
    for (int i = 0; i < static_cast<int>(N); ++i) {
        src_data[i] = i + 100;
    }

    int *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(int));
    cudaMalloc(&d_dst, N * sizeof(int));
    cudaMemcpy(d_src, src_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    cuda::stream::Stream stream;
    cuda::async::async_copy_d2d(d_dst, d_src, N * sizeof(int), stream.get());
    stream.synchronize();

    std::vector<int> result(N);
    cudaMemcpy(result.data(), d_dst, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < static_cast<int>(N); ++i) {
        EXPECT_EQ(result[i], src_data[i]);
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_F(AsyncCopyTest, AsyncCopyWithDirectionWorks) {
    constexpr size_t N = 256;
    std::vector<float> h_data(N, 3.14f);

    float* d_ptr;
    cudaMalloc(&d_ptr, N * sizeof(float));

    cuda::stream::Stream stream;
    cuda::async::async_copy(d_ptr, h_data.data(), N * sizeof(float),
                            cuda::async::CopyDirection::HostToDevice, stream.get());
    stream.synchronize();

    std::vector<float> result(N);
    cuda::async::async_copy(result.data(), d_ptr, N * sizeof(float),
                            cuda::async::CopyDirection::DeviceToHost, stream.get());
    stream.synchronize();

    EXPECT_FLOAT_EQ(result[0], 3.14f);

    cudaFree(d_ptr);
}

TEST_F(AsyncCopyTest, AsyncCopyFromPinnedWorks) {
    constexpr size_t N = 512;
    auto pinned = cuda::async::make_pinned<int>(N);
    for (int i = 0; i < static_cast<int>(N); ++i) {
        pinned.data()[i] = i * 10;
    }

    int* d_ptr;
    cudaMalloc(&d_ptr, N * sizeof(int));

    cuda::stream::Stream stream;
    cuda::async::async_copy_from_pinned(d_ptr, pinned, 0, N, stream.get());
    stream.synchronize();

    std::vector<int> result(N);
    cudaMemcpy(result.data(), d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[100], 1000);

    cudaFree(d_ptr);
}

TEST_F(AsyncCopyTest, AsyncCopyToPinnedWorks) {
    constexpr size_t N = 256;
    std::vector<int> h_data(N);
    for (int i = 0; i < static_cast<int>(N); ++i) {
        h_data[i] = i * 5;
    }

    int* d_ptr;
    cudaMalloc(&d_ptr, N * sizeof(int));
    cudaMemcpy(d_ptr, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    cuda::async::PinnedBuffer<int> pinned(N);
    cuda::stream::Stream stream;
    cuda::async::async_copy_to_pinned(pinned, d_ptr, 0, N, stream.get());
    stream.synchronize();

    EXPECT_EQ(pinned.data()[50], 250);

    cudaFree(d_ptr);
}
