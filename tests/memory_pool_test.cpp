#include <gtest/gtest.h>
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/memory/memory_pool.h"
#include "cuda/memory/unique_ptr.h"
#include <atomic>
#include <numeric>
#include <thread>
#include <vector>
#include "cuda/device/error.h"

namespace {

    void reset() {
        // Was cudaDeviceReset(): that tears down the whole process context and leaves
        // every singleton's cached handles stale, which libcuda dereferences rather than
        // reporting when they are used again (measured: cuEventDestroy on a handle created
        // before the reset segfaults). Synchronising gives these fixtures what they
        // actually want -- no work still in flight from the previous case.
        (void)cudaDeviceSynchronize();
    }

}  // namespace

class BufferTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

// The device-wide-memory cases live in ...SerialTest suites so CMake can exclude them
// from the parallel discovery set and register them with RUN_SERIAL. cudaMemGetInfo
// aggregates every process, so these can only mean anything when nothing else is
// allocating on the device.
class BufferSerialTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
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
    // Deliberately self-moving: the case exists to prove the move-assignment operator's
    // self-check works. GCC diagnoses this unconditionally, hence the suppression.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
    buffer = std::move(buffer);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
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

TEST_F(BufferSerialTest, DestructorFrees) {
    // Verify the destructor releases device memory via cudaMemGetInfo accounting
    // instead of dereferencing the freed pointer. Reading through a freed device
    // pointer is undefined behaviour; on some drivers it even leaves the CUDA
    // context in a faulted state that breaks later thrust/cub dispatch.
    //
    // cudaMemGetInfo aggregates EVERY process on the device, so the case lives in a
    // ...SerialTest suite and CMake registers it with RUN_SERIAL: another process
    // allocating between the two reads is indistinguishable from this buffer not
    // being freed.
    //
    // Serialising is necessary but not sufficient. The driver's allocator keeps a
    // 64 KiB pool from the FIRST cudaMalloc and never returns it, so measured on a
    // fresh context the "after" snapshot is 65536 bytes below the "before" one even
    // though the buffer was freed correctly. Probe, repeated per allocation size:
    //
    //     400 bytes: net retained = 65536
    //   65536 bytes: net retained = 65536
    //   1 MiB      : net retained = 131072
    //  64 MiB      : net retained = 65536
    //
    // One warm-up allocation therefore happens before the baseline is taken, so the
    // only thing measured is the buffer under test.
    void* warmup = nullptr;
    ASSERT_EQ(cudaMalloc(&warmup, 4096), cudaSuccess);
    ASSERT_EQ(cudaFree(warmup), cudaSuccess);

    size_t free_before = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);

    {
        cuda::memory::Buffer<int> buffer(100);
        size_t free_in_use = 0;
        ASSERT_EQ(cudaMemGetInfo(&free_in_use, &total), cudaSuccess);
        EXPECT_LT(free_in_use, free_before) << "Allocating the buffer must consume device memory";
    }

    size_t free_after = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
    EXPECT_GE(free_after, free_before) << "Destructor must release the buffer's device memory";
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
    CUDA_CHECK(cudaDeviceSynchronize());

    // This case used to end after the copy, so it passed whether or not the copy did
    // anything at all: a no-op implementation, a silently failed memcpy, or a wrong
    // direction would all satisfy it. Reading the bytes back is what makes it a test.
    std::vector<char> readback(100, 0);
    buffer.copy_to(readback.data(), 100);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < readback.size(); ++i) {
        ASSERT_EQ(readback[i], 42) << "byte " << i << " did not survive the round trip";
    }
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
    void SetUp() override { reset(); }
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

// Regression: deallocate() subtracted the CALLER'S byte count from the block's bump offset
// without recording what allocate() had handed out. Releasing a 100-byte pointer as 200 bytes
// underflowed size_t -- total_allocated became 18446744073709551516 -- and because
// find_block_for_size() computes (block.size - block.offset), the wrapped offset read as
// thousands of free bytes. The next allocate() then returned a pointer BEFORE the block start,
// outside the cudaMalloc'd region.
//
// A caller confusing element count with byte count, or releasing twice with a different size,
// is exactly the mistake this must survive.
TEST_F(MemoryPoolTest, DeallocateUsesTheSizeThatWasAllocatedNotTheSizeReported) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 1;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* first = pool.allocate(100);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(pool.total_allocated(), 100u);
    EXPECT_EQ(pool.num_allocations(), 1u);

    // Overstated release size.
    pool.deallocate(first, 200);

    EXPECT_EQ(pool.total_allocated(), 0u) << "accounting must return to zero, not wrap";
    EXPECT_EQ(pool.total_available(), 4096u) << "the whole block is free again";
    EXPECT_EQ(pool.num_allocations(), 0u);

    // And the block must still be usable from its start, not from a wrapped offset.
    void* second = pool.allocate(4000);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second, first) << "a fully released block must serve the next request from its beginning";
    pool.deallocate(second, 4000);
    EXPECT_EQ(pool.total_allocated(), 0u);
}

// The other direction: reporting less than was allocated must not leave the block believing it
// has space it already gave away.
TEST_F(MemoryPoolTest, DeallocateToleratesAnUnderstatedSize) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 1;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* p = pool.allocate(1000);
    ASSERT_NE(p, nullptr);
    pool.deallocate(p, 10);  // caller under-reports

    EXPECT_EQ(pool.total_allocated(), 0u) << "the allocation is gone regardless of the reported size";
    EXPECT_EQ(pool.total_available(), 4096u);
}

TEST_F(MemoryPoolTest, DeallocateReturnsMemory) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 1;
    config.preallocate = true;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.total_available(), 4096);

    void* ptr = pool.allocate(100);
    const size_t available_before = pool.total_available();

    pool.deallocate(ptr, 100);

    EXPECT_EQ(pool.total_allocated(), 0);
    // The case is named DeallocateReturnsMemory, so the memory coming back is the thing
    // to assert; available_before was collected and never compared.
    EXPECT_EQ(pool.total_available(), 4096) << "deallocating the only allocation must restore the full block";
    EXPECT_EQ(pool.total_available(), available_before + 100);
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

// This case used to assert that a 2048-byte request against 1024-byte blocks returned a non-null
// pointer. That is the defect, not the contract: the pool handed out a pointer into a 1024-byte block
// and advanced the block's offset past its end, so the caller believed it owned 2048 bytes. Changing
// the assertion rather than the code was not an option -- the assertion described memory the
// allocation did not own.
//
// The contract now: a request larger than one block is refused, and the refusal is visible as a
// throw (throw_on_failure defaults to true) rather than as a silent overrun.
TEST_F(MemoryPoolTest, AllocateBeyondBlockSize) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 1024;
    config.max_blocks = 8;
    config.preallocate = false;

    cuda::memory::MemoryPool pool(config);
    EXPECT_EQ(pool.num_blocks(), 0);

    EXPECT_THROW(pool.allocate(2048), std::runtime_error) << "a request larger than the block size must be refused, not handed a short block";

    // Nothing was handed out, so nothing is accounted for.
    EXPECT_EQ(pool.total_allocated(), 0u);
    EXPECT_EQ(pool.total_available(), 0u) << "a refused request must not credit or debit availability";
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
    void SetUp() override { reset(); }
};

class UniquePtrSerialTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
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

TEST_F(UniquePtrSerialTest, DestructorFrees) {
    // Verify the destructor releases device memory via cudaMemGetInfo accounting
    // instead of dereferencing the freed pointer (UB that can fault the driver).
    //
    // The warm-up and the RUN_SERIAL registration are both required, for the reasons
    // spelled out on BufferSerialTest.DestructorFrees above: the driver retains a
    // 64 KiB pool from the first cudaMalloc, and cudaMemGetInfo aggregates every
    // process on the device.
    void* warmup = nullptr;
    ASSERT_EQ(cudaMalloc(&warmup, 4096), cudaSuccess);
    ASSERT_EQ(cudaFree(warmup), cudaSuccess);

    size_t free_before = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);

    {
        cuda::memory::unique_ptr<int> ptr(100);
        size_t free_in_use = 0;
        ASSERT_EQ(cudaMemGetInfo(&free_in_use, &total), cudaSuccess);
        EXPECT_LT(free_in_use, free_before) << "Allocating the unique_ptr must consume device memory";
    }

    size_t free_after = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
    EXPECT_GE(free_after, free_before) << "Destructor must release the unique_ptr's device memory";
}

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(ErrorHandlingTest, InvalidSizeTrows) {
    EXPECT_THROW({ cuda::memory::Buffer<void> buffer(SIZE_MAX); }, cuda::device::CudaException);
    // A failed giant allocation can leave a sticky CUDA error that later tests
    // observe; drain the device error state so this negative test cannot poison
    // subsequent thrust/cub dispatch in the same process.
    (void)cudaGetLastError();
    (void)cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(ErrorHandlingTest, BufferVoidInvalidSizeTrows) {
    EXPECT_THROW({ cuda::memory::Buffer<void> buffer(static_cast<size_t>(-1)); }, cuda::device::CudaException);
    (void)cudaGetLastError();
    (void)cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

class ScopedMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
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
    void SetUp() override { reset(); }
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
    EXPECT_EQ(output[N - 1], N);
}

class MemoryPoolStressTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
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
    void SetUp() override { reset(); }
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

// deallocate() used to rewind the block's bump offset unconditionally, so releasing anything but the
// most recent allocation made the next allocate() hand back memory that was still live. Measured
// with a 4096-byte block:
//
//   a = allocate(256) -> 0x...000
//   b = allocate(256) -> 0x...100
//   c = allocate(256) -> 0x...200
//   deallocate(a)                        // offset stepped back to 0x200
//   d = allocate(256) -> 0x...200        // exactly c
//
// Two live allocations sharing memory is data corruption the moment either is written. The block's
// offset now moves only when its last allocation is released.
TEST(MemoryPoolReuseTest, ReleasingAnInteriorAllocationDoesNotHandOutLiveMemory) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* a = pool.allocate(256);
    void* b = pool.allocate(256);
    void* c = pool.allocate(256);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(a, b);
    ASSERT_NE(b, c);

    pool.deallocate(a, 256);
    void* d = pool.allocate(256);
    ASSERT_NE(d, nullptr);
    EXPECT_NE(d, b) << "the new allocation landed on a live block";
    EXPECT_NE(d, c) << "the new allocation landed on a live block";

    pool.deallocate(b, 256);
    pool.deallocate(c, 256);
    pool.deallocate(d, 256);
}

// The reuse that IS safe must still happen: a block whose allocations have all been released serves
// the next request from its beginning, which is what pooling is for.
TEST(MemoryPoolReuseTest, AnEmptyBlockIsReusedFromItsBeginning) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 1;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* first = pool.allocate(256);
    ASSERT_NE(first, nullptr);
    pool.deallocate(first, 256);

    void* second = pool.allocate(4000);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second, first) << "an emptied block must serve the next request from offset zero";
    pool.deallocate(second, 4000);
}

// A request larger than a block cannot be served -- every block is config.block_size bytes -- and it
// used to be served anyway. find_block_for_size() correctly reported no fit, allocate_block()
// created a fresh block, and allocate() handed out a pointer into it while advancing offset past the
// end: measured with block_size 4096 and a request of 8192, one block allocated, total_allocated
// 8192, and total_available 2^64-4096 because the subtraction wrapped. The caller held a pointer to
// twice the memory that existed.
TEST(MemoryPoolReuseTest, ARequestLargerThanABlockIsRefused) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    EXPECT_THROW(pool.allocate(8192), std::runtime_error) << "a request bigger than one block must not be served from it";

    EXPECT_EQ(pool.total_available(), 0u) << "no block should be allocated for a request that cannot be served";
    EXPECT_EQ(pool.total_allocated(), 0u);
    EXPECT_TRUE(pool.get_allocations_by_stream().empty()) << "a refused request must not leave a live allocation";

    // A request that does fit must still be served.
    void* ok = pool.allocate(4096);
    ASSERT_NE(ok, nullptr);
    pool.deallocate(ok, 4096);
}

// The shape most callers actually use: allocate, release, allocate the same size again. The
// interior-release fix in this file stops the bump offset from rewinding while other allocations in
// the block are still live -- it must not stop the block from being reused when it is empty, and it
// must not force a second cudaMalloc for the second request.
TEST(MemoryPoolReuseTest, AllocateReleaseAllocateReusesTheSameBlock) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* first = pool.allocate(256);
    ASSERT_NE(first, nullptr);
    pool.deallocate(first, 256);

    void* second = pool.allocate(256);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second, first) << "the pool must hand back the address it just took";

    // One block was ever allocated: the release must not have cost a fresh cudaMalloc.
    const auto metrics = pool.get_metrics();
    EXPECT_EQ(metrics.misses, 1u) << "the first request missed; the second must not";
    EXPECT_EQ(metrics.hits, 1u) << "the second request must be served from the pool";
    EXPECT_EQ(pool.num_blocks(), 1u) << "no second block should have been allocated";

    pool.deallocate(second, 256);
}

// Releasing the same pointer twice must not corrupt the accounting or let one address be handed out
// to two live callers. deallocate() finds the allocation in allocation_map_ and returns when the
// entry is gone, so the second call is a no-op rather than a second credit.
TEST(MemoryPoolReuseTest, ReleasingTheSamePointerTwiceIsANoOp) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 4;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* first = pool.allocate(256);
    ASSERT_NE(first, nullptr);
    pool.deallocate(first, 256);
    pool.deallocate(first, 256);

    EXPECT_EQ(pool.total_allocated(), 0u) << "the second release must not drive the count negative";

    void* a = pool.allocate(256);
    void* b = pool.allocate(256);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b) << "a double release must not let one address serve two live allocations";

    pool.deallocate(a, 256);
    pool.deallocate(b, 256);
}

// defragment() frees and reallocates each used block, which moves the memory that every live
// allocation points into. Nothing in the API lets a caller learn that its pointer moved, so the
// operation is refused while anything is live. Measured before the guard: with one block holding 1000
// used bytes, cudaMemcpy from the caller's pointer after defragment() returned cudaErrorInvalidValue,
// and total_available() had risen from 3096 to 4096 because Block::size still claimed the old size.
TEST(MemoryPoolReuseTest, DefragmentLeavesLiveAllocationsAlone) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    void* a = pool.allocate(1000);
    ASSERT_NE(a, nullptr);
    CUDA_CHECK(cudaMemset(a, 0xAB, 1000));
    const size_t available_before = pool.total_available();

    pool.defragment();

    // The caller's pointer must still refer to its own data.
    unsigned char probe = 0;
    ASSERT_EQ(cudaMemcpy(&probe, a, 1, cudaMemcpyDeviceToHost), cudaSuccess) << "defragment() must not move memory a live allocation points into";
    EXPECT_EQ(probe, 0xAB) << "the data must be unchanged";
    EXPECT_EQ(pool.total_available(), available_before) << "nothing should have been reclaimed while in use";

    pool.deallocate(a, 1000);
}

// num_active_streams and get_allocations_by_stream() describe the same thing and used to disagree.
// The metric counted Block::stream_id, which only records the stream of the most recent allocate()
// into that block; the map walks every live allocation. Four 256-byte allocations on streams 0, 1, 2
// and 1 all fit one 4096-byte block, so the block was tagged stream 1 and the metric reported 1 while
// the map reported 3.
TEST(MemoryPoolReuseTest, ActiveStreamCountMatchesThePerStreamBreakdown) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 8;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    EXPECT_NE(pool.allocate(256, 0), nullptr);
    EXPECT_NE(pool.allocate(256, 1), nullptr);
    EXPECT_NE(pool.allocate(256, 2), nullptr);
    EXPECT_NE(pool.allocate(256, 1), nullptr);

    const auto by_stream = pool.get_allocations_by_stream();
    const auto metrics = pool.get_metrics();

    EXPECT_EQ(by_stream.size(), 3u) << "three distinct streams hold allocations";
    EXPECT_EQ(static_cast<size_t>(metrics.num_active_streams), by_stream.size()) << "the metric must count the same streams the breakdown reports";
    EXPECT_EQ(pool.num_blocks(), 1u) << "all four allocations fit one block, which is what hid the bug";
}

// get_metrics() used to lock metrics_mutex_ while walking blocks_, which allocate_block() grows
// under mutex_. Two locks over one vector means a concurrent get_metrics() could iterate it
// mid-reallocation, and metrics_mutex_ had exactly one user -- get_metrics() itself -- so it
// protected nothing. The accessors were unsynchronised reads of counters written under mutex_.
//
// This case cannot prove the absence of a race, but it does exercise the paths concurrently, which
// is what a future change would have to keep working.
TEST(MemoryPoolConcurrencyTest, MetricsAndAllocationCanRunAtTheSameTime) {
    cuda::memory::MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 256;
    config.preallocate = false;
    cuda::memory::MemoryPool pool(config);

    std::atomic<bool> stop {false};
    std::atomic<long> reads {0};
    std::thread reader([&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const auto metrics = pool.get_metrics();
            (void)metrics;
            (void)pool.total_allocated();
            (void)pool.total_available();
            (void)pool.num_blocks();
            ++reads;
        }
    });

    std::thread worker([&] {
        for (int i = 0; i < 2000; ++i) {
            void* p = pool.allocate(64);
            if (p != nullptr) {
                pool.deallocate(p, 64);
            }
        }
    });
    worker.join();
    stop = true;
    reader.join();

    EXPECT_GT(reads.load(), 0L) << "the reader must have observed at least one snapshot";
    EXPECT_EQ(pool.total_allocated(), 0u) << "everything was released";
}
