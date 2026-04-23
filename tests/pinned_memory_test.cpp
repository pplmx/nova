#include <gtest/gtest.h>
#include "cuda/async/pinned_memory.h"
#include "cuda/stream/stream.h"

class PinnedMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(PinnedMemoryTest, DefaultConstruction) {
    cuda::async::PinnedMemory<int> mem;

    EXPECT_EQ(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 0);
    EXPECT_FALSE(mem);
}

TEST_F(PinnedMemoryTest, ConstructionWithSize) {
    cuda::async::PinnedMemory<int> mem(100);

    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 100);
    EXPECT_TRUE(mem);
}

TEST_F(PinnedMemoryTest, DataAccess) {
    cuda::async::PinnedMemory<int> mem(10);

    for (int i = 0; i < 10; ++i) {
        mem.data()[i] = i;
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(mem.data()[i], i);
    }
}

TEST_F(PinnedMemoryTest, SizeBytes) {
    cuda::async::PinnedMemory<double> mem(50);

    EXPECT_EQ(mem.size_bytes(), 50 * sizeof(double));
}

TEST_F(PinnedMemoryTest, MoveSemantics) {
    cuda::async::PinnedMemory<int> mem1(100);
    mem1.data()[0] = 42;

    cuda::async::PinnedMemory<int> mem2(std::move(mem1));

    EXPECT_EQ(mem2.data()[0], 42);
    EXPECT_EQ(mem1.data(), nullptr);
}

TEST_F(PinnedMemoryTest, MoveAssignment) {
    cuda::async::PinnedMemory<int> mem1(100);
    mem1.data()[0] = 42;

    cuda::async::PinnedMemory<int> mem2;
    mem2 = std::move(mem1);

    EXPECT_EQ(mem2.data()[0], 42);
    EXPECT_EQ(mem1.data(), nullptr);
}

TEST_F(PinnedMemoryTest, Reset) {
    cuda::async::PinnedMemory<int> mem(100);

    mem.reset(50);

    EXPECT_EQ(mem.size(), 50);
}

TEST_F(PinnedMemoryTest, Release) {
    cuda::async::PinnedMemory<int> mem(100);

    int* ptr = mem.release();

    EXPECT_EQ(mem.data(), nullptr);
    EXPECT_EQ(mem.size(), 0);
    EXPECT_NE(ptr, nullptr);

    cudaFreeHost(ptr);
}

TEST_F(PinnedMemoryTest, MakePinned) {
    auto mem = cuda::async::make_pinned<int>(100);

    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 100);
}

class PinnedBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(PinnedBufferTest, CopyFromVector) {
    cuda::async::PinnedBuffer<int> buf(10);
    std::vector<int> data = {1, 2, 3, 4, 5};

    buf.copy_from(data);

    EXPECT_EQ(buf.data()[0], 1);
    EXPECT_EQ(buf.data()[4], 5);
}

TEST_F(PinnedBufferTest, CopyToVector) {
    cuda::async::PinnedBuffer<int> buf(5);
    for (int i = 0; i < 5; ++i) {
        buf.data()[i] = i * 10;
    }

    std::vector<int> data;
    buf.copy_to(data);

    EXPECT_EQ(data.size(), 5);
    EXPECT_EQ(data[2], 20);
}

TEST_F(PinnedBufferTest, CopyToDeviceAsync) {
    cuda::async::PinnedBuffer<int> buf(100);
    for (int i = 0; i < 100; ++i) {
        buf.data()[i] = i;
    }

    int* d_ptr;
    cudaMalloc(&d_ptr, 100 * sizeof(int));

    cuda::stream::Stream stream;
    buf.copy_to_device(d_ptr, stream.get());
    stream.synchronize();

    std::vector<int> result(100);
    cudaMemcpy(result.data(), d_ptr, 100 * sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[50], 50);

    cudaFree(d_ptr);
}

TEST_F(PinnedBufferTest, CopyFromDeviceAsync) {
    std::vector<int> h_data(100);
    for (int i = 0; i < 100; ++i) {
        h_data[i] = i;
    }

    int* d_ptr;
    cudaMalloc(&d_ptr, 100 * sizeof(int));
    cudaMemcpy(d_ptr, h_data.data(), 100 * sizeof(int), cudaMemcpyHostToDevice);

    cuda::async::PinnedBuffer<int> buf(100);
    cuda::stream::Stream stream;
    buf.copy_from_device(d_ptr, stream.get());
    stream.synchronize();

    EXPECT_EQ(buf.data()[50], 50);

    cudaFree(d_ptr);
}
