#include <gtest/gtest.h>
#include "cuda/async/pinned_memory.h"

#include "cuda_alloc_injection.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include "cuda/stream/stream.h"

class PinnedMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        (void)cudaDeviceSynchronize();
        CUDA_CHECK(cudaSetDevice(0));
        (void)cudaDeviceSynchronize();
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

// reset() frees the current buffer and then allocates a new one, without clearing the handle in
// between:
//
//     if (data_) { cudaFreeHost(data_); }        // data_ still points at the freed block
//     size_ = count;
//     if (count > 0) { CUDA_CHECK(cudaMallocHost(&data_, count * sizeof(T))); }   // throws here
//
// If the reallocation fails, the exception leaves data_ pointing at memory that was already
// released, and size_ already updated. The destructor then calls cudaFreeHost on that stale handle
// -- a double free -- and the object reports a size that does not match what it holds.
TEST_F(PinnedMemoryTest, ResetLeavesTheObjectEmptyWhenTheReallocationFails) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        (void)cudaGetLastError();
        GTEST_SKIP() << "no CUDA device";
    }
    (void)cudaGetLastError();

    // The allocation inside reset() is a template body compiled into this file, so --wrap does not
    // reach it (the same limitation recorded in tests/cuda_alloc_injection.h for
    // Benchmark::run()). Ask for a size the driver cannot satisfy instead: the failure is the
    // driver's, not an injected one, so it arrives through the same throw.
    cuda::async::PinnedMemory<float> buffer(16);
    ASSERT_TRUE(static_cast<bool>(buffer)) << "the initial allocation must have succeeded";

    const size_t impossible = static_cast<size_t>(-1) / sizeof(float) / 2;
    EXPECT_THROW(buffer.reset(impossible), std::exception) << "an impossible size must fail the reallocation";

    // Strong guarantee: the object keeps what it had. The defect was that it kept a handle to memory
    // it had already released, together with the size it failed to allocate; keeping the original
    // buffer intact is the behaviour that makes the failure recoverable.
    EXPECT_TRUE(static_cast<bool>(buffer)) << "a failed reset must leave the object usable, not holding a freed handle";
    EXPECT_EQ(buffer.size(), 16u) << "a failed reset must keep the previous size, not the one it failed to allocate";
    EXPECT_NE(buffer.get(), nullptr);

    (void)cudaGetLastError();
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
    void SetUp() override { (void)cudaDeviceSynchronize(); }
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
