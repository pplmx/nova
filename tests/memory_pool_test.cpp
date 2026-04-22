#include <gtest/gtest.h>
#include "cuda/memory/buffer.h"
#include "cuda/memory/memory_pool.h"
#include "cuda/memory/unique_ptr.h"
#include <vector>
#include <numeric>

namespace {

void reset() {
    cudaDeviceReset();
}

}

class BufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(BufferTest, DefaultConstruction) {
    cuda::memory::Buffer<int> buffer;
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
}

TEST_F(BufferTest, ConstructionWithSize) {
    cuda::memory::Buffer<int> buffer(100);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 100);
}

TEST_F(BufferTest, ConstructionWithZeroSize) {
    cuda::memory::Buffer<int> buffer(0);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
}

TEST_F(BufferTest, CopyConstructionDeleted) {
    cuda::memory::Buffer<int> buffer(10);
    EXPECT_FALSE(std::is_copy_constructible_v<cuda::memory::Buffer<int>>);
}

TEST_F(BufferTest, CopyAssignmentDeleted) {
    cuda::memory::Buffer<int> buffer(10);
    EXPECT_FALSE(std::is_copy_assignable_v<cuda::memory::Buffer<int>>);
}

TEST_F(BufferTest, MoveConstruction) {
    cuda::memory::Buffer<int> buffer1(10);
    auto* data = buffer1.data();
    cuda::memory::Buffer<int> buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.data(), data);
    EXPECT_EQ(buffer2.size(), 10);
    EXPECT_EQ(buffer1.data(), nullptr);
    EXPECT_EQ(buffer1.size(), 0);
}

TEST_F(BufferTest, MoveAssignment) {
    cuda::memory::Buffer<int> buffer1(10);
    cuda::memory::Buffer<int> buffer2(20);
    auto* data = buffer1.data();

    buffer2 = std::move(buffer1);

    EXPECT_EQ(buffer2.data(), data);
    EXPECT_EQ(buffer2.size(), 10);
    EXPECT_EQ(buffer1.data(), nullptr);
    EXPECT_EQ(buffer1.size(), 0);
}

TEST_F(BufferTest, MoveToSelf) {
    cuda::memory::Buffer<int> buffer(10);
    buffer = std::move(buffer);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 10);
}

TEST_F(BufferTest, CopyFromHost) {
    cuda::memory::Buffer<int> buffer(10);
    std::vector<int> host(10);
    std::iota(host.begin(), host.end(), 1);

    buffer.copy_from(host.data(), 10);

    std::vector<int> result(10);
    buffer.copy_to(result.data(), 10);

    EXPECT_EQ(result, host);
}

TEST_F(BufferTest, CopyToHost) {
    cuda::memory::Buffer<int> buffer(5);
    std::vector<int> expected = {10, 20, 30, 40, 50};
    buffer.copy_from(expected.data(), 5);

    std::vector<int> result(5);
    buffer.copy_to(result.data(), 5);

    EXPECT_EQ(result, expected);
}

TEST_F(BufferTest, Release) {
    cuda::memory::Buffer<int> buffer(10);
    auto* data = buffer.data();
    int* released = buffer.release();

    EXPECT_EQ(released, data);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);

    cudaFree(released);
}

TEST_F(BufferTest, DestructorFrees) {
    int* ptr = nullptr;
    {
        cuda::memory::Buffer<int> buffer(100);
        ptr = buffer.data();
    }

    void* dummy = nullptr;
    cudaError_t err = cudaMemcpy(&dummy, ptr, sizeof(void*), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        GTEST_FAIL() << "Memory should have been freed";
    }
}

TEST_F(BufferTest, VoidBufferConstruction) {
    cuda::memory::Buffer<void> buffer(256);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 256);
}

TEST_F(BufferTest, VoidBufferCopyFrom) {
    cuda::memory::Buffer<void> buffer(100);
    std::vector<char> host(100, 42);
    buffer.copy_from(host.data(), 100);
}

TEST_F(BufferTest, VoidBufferRelease) {
    cuda::memory::Buffer<void> buffer(100);
    void* ptr = buffer.release();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(buffer.data(), nullptr);
    cudaFree(ptr);
}

class MemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(MemoryPoolTest, DefaultConstruction) {
    cuda::memory::MemoryPool::Config config;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_blocks(), 0);
    EXPECT_EQ(pool.total_allocated(), 0);
    EXPECT_EQ(pool.total_available(), 0);
}

TEST_F(MemoryPoolTest, PreallocatedBlocks) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 4;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_blocks(), 4);
    EXPECT_EQ(pool.total_available(), 1024 * 4);
}

TEST_F(MemoryPoolTest, AllocateFromPool) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    auto* ptr1 = pool.allocate(100);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(pool.total_allocated(), 100);
    EXPECT_LT(pool.total_available(), 4096 * 2);

    auto* ptr2 = pool.allocate(200);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(pool.total_allocated(), 300);

    EXPECT_NE(ptr1, ptr2);
}

TEST_F(MemoryPoolTest, DeallocateReturnsMemory) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 1;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.total_available(), 4096);

    void* ptr = pool.allocate(100);
    size_t available_before = pool.total_available();

    pool.deallocate(ptr, 100);

    EXPECT_EQ(pool.total_allocated(), 0);
}

TEST_F(MemoryPoolTest, FirstFitAllocation) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);

    void* ptr1 = pool.allocate(1000);
    pool.deallocate(ptr1, 1000);

    void* ptr2 = pool.allocate(500);
    EXPECT_NE(ptr2, nullptr);
}

TEST_F(MemoryPoolTest, AllocateBeyondBlockSize) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 8;
    config.preallocate = false;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_blocks(), 0);

    void* ptr = pool.allocate(2048);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(pool.num_blocks(), 1);
}

TEST_F(MemoryPoolTest, ExhaustPoolThenAllocate) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 256;
    config.max_blocks = 8;
    config.preallocate = false;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_blocks(), 0);

    pool.allocate(100);
    pool.allocate(200);

    EXPECT_EQ(pool.num_blocks(), 2);
}

TEST_F(MemoryPoolTest, AllocateZeroBytes) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 1;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);

    void* ptr = pool.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryPoolTest, DeallocateNullptr) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    size_t allocated_before = pool.total_allocated();

    pool.deallocate(nullptr, 100);

    EXPECT_EQ(pool.total_allocated(), allocated_before);
}

TEST_F(MemoryPoolTest, ClearFreesAllBlocks) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    pool.allocate(1000);
    pool.allocate(2000);

    pool.clear();

    EXPECT_EQ(pool.num_blocks(), 0);
    EXPECT_EQ(pool.total_allocated(), 0);
    EXPECT_EQ(pool.total_available(), 0);
}

TEST_F(MemoryPoolTest, MoveConstructor) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = true;

    cuda::memory::MemoryPool pool1(config);
    pool1.allocate(100);

    cuda::memory::MemoryPool pool2(std::move(pool1));

    EXPECT_EQ(pool2.num_blocks(), 2);
    EXPECT_EQ(pool2.total_allocated(), 100);
}

TEST_F(MemoryPoolTest, MoveAssignment) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = true;

    cuda::memory::MemoryPool pool1(config);
    pool1.allocate(100);

    cuda::memory::MemoryPool pool2;
    pool2 = std::move(pool1);

    EXPECT_EQ(pool2.num_blocks(), 2);
    EXPECT_EQ(pool2.total_allocated(), 100);
}

TEST_F(MemoryPoolTest, NumAllocations) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_allocations(), 0);

    pool.allocate(100);
    EXPECT_EQ(pool.num_allocations(), 1);

    pool.allocate(200);
    EXPECT_EQ(pool.num_allocations(), 2);

    void* ptr = pool.allocate(300);
    pool.deallocate(ptr, 300);
    EXPECT_EQ(pool.num_allocations(), 2);
}

TEST_F(MemoryPoolTest, MultipleAllocationsSameBlock) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 1;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool.allocate(50));
    }

    EXPECT_EQ(pool.num_blocks(), 1);
    EXPECT_EQ(pool.num_allocations(), 10);

    for (auto* ptr : ptrs) {
        pool.deallocate(ptr, 50);
    }

    EXPECT_EQ(pool.num_allocations(), 0);
}

class UniquePtrTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(UniquePtrTest, DefaultConstruction) {
    cuda::memory::unique_ptr<int> ptr;
    EXPECT_EQ(ptr.get(), nullptr);
    EXPECT_FALSE(ptr);
}

TEST_F(UniquePtrTest, ConstructionWithSize) {
    cuda::memory::unique_ptr<int> ptr(100);
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_TRUE(ptr);
}

TEST_F(UniquePtrTest, GetReturnsPointer) {
    cuda::memory::unique_ptr<int> ptr(50);
    int* raw = ptr.get();
    EXPECT_NE(raw, nullptr);
}

TEST_F(UniquePtrTest, CopyConstructionDeleted) {
    cuda::memory::unique_ptr<int> ptr(10);
    EXPECT_FALSE(std::is_copy_constructible_v<cuda::memory::unique_ptr<int>>);
}

TEST_F(UniquePtrTest, MoveConstruction) {
    cuda::memory::unique_ptr<int> ptr1(100);
    auto* data = ptr1.get();

    cuda::memory::unique_ptr<int> ptr2(std::move(ptr1));

    EXPECT_EQ(ptr2.get(), data);
    EXPECT_EQ(ptr1.get(), nullptr);
}

TEST_F(UniquePtrTest, MoveAssignment) {
    cuda::memory::unique_ptr<int> ptr1(100);
    cuda::memory::unique_ptr<int> ptr2(50);
    auto* data = ptr1.get();

    ptr2 = std::move(ptr1);

    EXPECT_EQ(ptr2.get(), data);
    EXPECT_EQ(ptr1.get(), nullptr);
}

TEST_F(UniquePtrTest, Release) {
    cuda::memory::unique_ptr<int> ptr(100);
    auto* data = ptr.get();

    int* released = ptr.release();

    EXPECT_EQ(released, data);
    EXPECT_EQ(ptr.get(), nullptr);

    cudaFree(released);
}

TEST_F(UniquePtrTest, ResetWithNullptr) {
    cuda::memory::unique_ptr<int> ptr(100);
    ptr.reset(nullptr);
    EXPECT_EQ(ptr.get(), nullptr);
}

TEST_F(UniquePtrTest, BoolOperator) {
    cuda::memory::unique_ptr<int> empty;
    cuda::memory::unique_ptr<int> valid(100);

    EXPECT_FALSE(empty);
    EXPECT_TRUE(valid);
}

TEST_F(UniquePtrTest, DestructorFrees) {
    int* tracking_ptr = nullptr;
    {
        cuda::memory::unique_ptr<int> ptr(100);
        tracking_ptr = ptr.get();
    }

    void* dummy = nullptr;
    cudaError_t err = cudaMemcpy(&dummy, tracking_ptr, sizeof(void*), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        GTEST_FAIL() << "Memory should have been freed";
    }
}

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(ErrorHandlingTest, InvalidSizeTrows) {
    EXPECT_THROW({
        cuda::memory::Buffer<void> buffer(SIZE_MAX);
    }, cuda::device::CudaException);
}

TEST_F(ErrorHandlingTest, BufferVoidInvalidSizeTrows) {
    EXPECT_THROW({
        cuda::memory::Buffer<void> buffer(static_cast<size_t>(-1));
    }, cuda::device::CudaException);
}

class ScopedMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(ScopedMemoryPoolTest, DefaultConstruction) {
    cuda::memory::MemoryPool::Config config;
    config.preallocate = false;
    cuda::memory::ScopedMemoryPool scoped(config);
    auto& pool = scoped.get();
    EXPECT_EQ(pool.num_blocks(), 0);
}

TEST_F(ScopedMemoryPoolTest, AllocateTypedWithDataVerification) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.preallocate = true;
    cuda::memory::ScopedMemoryPool scoped(config);

    constexpr int N = 100;
    int* ptr = scoped.allocate<int>(N);
    ASSERT_NE(ptr, nullptr);

    std::vector<int> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 1);
    CUDA_CHECK(cudaMemcpy(ptr, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> h_output(N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), ptr, N * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_output, h_input);

    scoped.deallocate<int>(ptr, N);
}

TEST_F(ScopedMemoryPoolTest, AllocateVoidWithDataVerification) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 8192;
    config.preallocate = true;
    cuda::memory::ScopedMemoryPool scoped(config);

    constexpr size_t N = 1000;
    void* ptr = scoped.allocate(N);
    ASSERT_NE(ptr, nullptr);

    std::vector<int> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 42);
    CUDA_CHECK(cudaMemcpy(ptr, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> h_output(N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), ptr, N * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_output, h_input);

    scoped.deallocate(ptr, N);
}

TEST_F(ScopedMemoryPoolTest, AllocateAndReallocate) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.preallocate = true;
    cuda::memory::ScopedMemoryPool scoped(config);

    constexpr int N = 100;
    int* ptr1 = scoped.allocate<int>(N);
    ASSERT_NE(ptr1, nullptr);

    std::vector<int> h_data1(N, 77);
    CUDA_CHECK(cudaMemcpy(ptr1, h_data1.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int* ptr2 = scoped.allocate<int>(N);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr2, ptr1);

    std::vector<int> h_data2(N, 88);
    CUDA_CHECK(cudaMemcpy(ptr2, h_data2.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    scoped.deallocate<int>(ptr1, N);
    scoped.deallocate<int>(ptr2, N);
}

TEST_F(ScopedMemoryPoolTest, MoveConstructor) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.preallocate = true;
    cuda::memory::ScopedMemoryPool scoped1(config);
    scoped1.allocate(100);

    cuda::memory::ScopedMemoryPool scoped2(std::move(scoped1));

    EXPECT_EQ(scoped2.get().num_allocations(), 1);
}

TEST_F(ScopedMemoryPoolTest, MoveAssignment) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.preallocate = true;
    cuda::memory::ScopedMemoryPool scoped1(config);
    scoped1.allocate(100);

    cuda::memory::ScopedMemoryPool scoped2;
    scoped2 = std::move(scoped1);

    EXPECT_EQ(scoped2.get().num_allocations(), 1);
}

class BufferIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(BufferIntegrationTest, CopyFromPartialData) {
    cuda::memory::Buffer<int> buffer(10);
    std::vector<int> source = {1, 2, 3, 4, 5};
    buffer.copy_from(source.data(), 5);

    std::vector<int> result(10);
    buffer.copy_to(result.data(), 5);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(result[i], source[i]);
    }
}

TEST_F(BufferIntegrationTest, MultipleCopies) {
    cuda::memory::Buffer<int> buffer(100);

    for (int iter = 0; iter < 5; ++iter) {
        std::vector<int> input(100);
        std::iota(input.begin(), input.end(), iter * 100);
        buffer.copy_from(input.data(), 100);

        std::vector<int> output(100);
        buffer.copy_to(output.data(), 100);
        EXPECT_EQ(output, input);
    }
}

TEST_F(BufferIntegrationTest, LargeBufferTransfer) {
    constexpr size_t N = 1000000;
    cuda::memory::Buffer<int> buffer(N);

    std::vector<int> input(N);
    std::iota(input.begin(), input.end(), 1);
    buffer.copy_from(input.data(), N);

    std::vector<int> output(N);
    buffer.copy_to(output.data(), N);

    EXPECT_EQ(output, input);
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[N-1], N);
}

class MemoryPoolStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(MemoryPoolStressTest, ManySmallAllocations) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1 << 20;
    config.max_blocks = 4;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(pool.allocate(100));
        ASSERT_NE(ptrs.back(), nullptr);
    }

    EXPECT_EQ(pool.num_allocations(), 100);

    for (auto* ptr : ptrs) {
        pool.deallocate(ptr, 100);
    }

    EXPECT_EQ(pool.num_allocations(), 0);
}

TEST_F(MemoryPoolStressTest, DeallocateAllowsReallocation) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    void* ptr1 = pool.allocate(1000);
    ASSERT_NE(ptr1, nullptr);

    pool.deallocate(ptr1, 1000);

    void* ptr2 = pool.allocate(1000);
    ASSERT_NE(ptr2, nullptr);

    ptr1 = pool.allocate(1000);
    ASSERT_NE(ptr1, nullptr);

    pool.deallocate(ptr2, 1000);
    pool.deallocate(ptr1, 1000);

    EXPECT_EQ(pool.num_allocations(), 0);
}

TEST_F(MemoryPoolStressTest, MixedSizeAllocations) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = true;
    cuda::memory::MemoryPool pool(config);

    std::vector<std::pair<void*, size_t>> allocs;

    allocs.push_back({pool.allocate(100), 100});
    allocs.push_back({pool.allocate(500), 500});
    allocs.push_back({pool.allocate(1000), 1000});
    allocs.push_back({pool.allocate(200), 200});

    for (auto& [ptr, size] : allocs) {
        ASSERT_NE(ptr, nullptr);
    }

    pool.deallocate(allocs[1].first, allocs[1].second);
    pool.deallocate(allocs[3].first, allocs[3].second);

    allocs[1] = {pool.allocate(300), 300};
    allocs[3] = {pool.allocate(150), 150};

    EXPECT_GE(pool.num_allocations(), 4);

    for (auto& [ptr, size] : allocs) {
        pool.deallocate(ptr, size);
    }

    EXPECT_EQ(pool.num_allocations(), 0);
}

class UniquePtrIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(UniquePtrIntegrationTest, WriteAndReadData) {
    cuda::memory::unique_ptr<int> ptr(100);

    std::vector<int> input(100);
    std::iota(input.begin(), input.end(), 1);
    CUDA_CHECK(cudaMemcpy(ptr.get(), input.data(), 100 * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> output(100);
    CUDA_CHECK(cudaMemcpy(output.data(), ptr.get(), 100 * sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output, input);
}

TEST_F(UniquePtrIntegrationTest, SwapWorks) {
    cuda::memory::unique_ptr<int> ptr1(50);
    cuda::memory::unique_ptr<int> ptr2(100);
    auto* data1 = ptr1.get();
    auto* data2 = ptr2.get();

    swap(ptr1, ptr2);

    EXPECT_EQ(ptr1.get(), data2);
    EXPECT_EQ(ptr2.get(), data1);
}
