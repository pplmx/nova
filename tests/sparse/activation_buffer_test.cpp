/**
 * @file activation_buffer_test.cpp
 * @brief Contracts for the double-buffered pipeline activation storage
 *
 * include/cuda/pipeline/activation_buffer.h backs the ping-pong activation
 * buffers that let pipeline parallelism overlap compute with communication. It
 * had no tests, and the constructor had two defects:
 *
 * - It allocated two buffers with two separate cudaMalloc calls and no cleanup
 *   between them. When the second failed, the constructor threw and the
 *   destructor never ran, leaking the first allocation. Reproduced by asking
 *   for ~70% of free memory per buffer: the first succeeded, the second failed,
 *   and 59 GB were stranded.
 *
 * - It called cudaSetDevice() and never restored, so constructing a buffer for
 *   another device silently migrated every later kernel on the calling thread.
 */

#include <cuda/pipeline/activation_buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace cuda::pipeline::test {

namespace {

int current_device() {
    int device = -1;
    cudaGetDevice(&device);
    return device;
}

size_t free_device_memory() {
    size_t free_mem = 0, total = 0;
    cudaMemGetInfo(&free_mem, &total);
    return free_mem;
}

}  // namespace

TEST(ActivationBufferTest, AllocatesTwoDistinctBuffers) {
    ActivationBuffer buffer(0, 1024);

    EXPECT_EQ(buffer.device(), 0);
    EXPECT_EQ(buffer.capacity(), 1024u);
    EXPECT_NE(buffer.active_buffer(), nullptr);
    EXPECT_NE(buffer.inactive_buffer(), nullptr);
    EXPECT_NE(buffer.active_buffer(), buffer.inactive_buffer())
        << "ping and pong must be separate allocations";
}

TEST(ActivationBufferTest, SwapExchangesTheRoles) {
    ActivationBuffer buffer(0, 256);

    const void* active_before = buffer.active_buffer();
    const void* inactive_before = buffer.inactive_buffer();
    EXPECT_TRUE(buffer.is_ping_active());

    buffer.swap();
    EXPECT_FALSE(buffer.is_ping_active());
    EXPECT_EQ(buffer.active_buffer(), inactive_before);
    EXPECT_EQ(buffer.inactive_buffer(), active_before);

    buffer.swap();
    EXPECT_TRUE(buffer.is_ping_active());
    EXPECT_EQ(buffer.active_buffer(), active_before);
}

TEST(ActivationBufferTest, ZeroCapacityIsAllowed) {
    // cudaMalloc(0) is legal and yields null; the object must still be usable so
    // a degenerate pipeline configuration does not throw at construction.
    EXPECT_NO_THROW({
        ActivationBuffer buffer(0, 0);
        EXPECT_EQ(buffer.capacity(), 0u);
        buffer.swap();
        EXPECT_FALSE(buffer.is_ping_active());
    });
}

// Regression: the caller's selected device must survive construction and
// destruction. A buffer for device 0 used to leave device 0 selected even when
// the caller was somewhere else.
TEST(ActivationBufferTest, ConstructingDoesNotChangeTheCallersDevice) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
        GTEST_SKIP() << "needs at least two devices to observe a device switch";
    }

    const int original = current_device();
    const int other = (original == 0) ? 1 : 0;
    ASSERT_EQ(cudaSetDevice(original), cudaSuccess);

    {
        ActivationBuffer buffer(other, 256);
        EXPECT_EQ(buffer.device(), other);
        EXPECT_EQ(current_device(), original)
            << "constructing a buffer must not migrate the calling thread";
    }

    EXPECT_EQ(current_device(), original)
        << "destroying a buffer must not migrate the calling thread";
    EXPECT_EQ(cudaSetDevice(original), cudaSuccess);
}

// Regression: the second allocation failed and the first leaked. Asking for
// more than half of free memory makes that happen deterministically, because
// the two buffers together cannot fit.
TEST(ActivationBufferTest, FailedSecondAllocationDoesNotLeakTheFirst) {
    const size_t free_before = free_device_memory();
    ASSERT_GT(free_before, 0u);

    // 70% of free: one fits, two do not.
    const size_t capacity = static_cast<size_t>(free_before * 0.70);
    if (capacity == 0) {
        GTEST_SKIP() << "no free device memory to work with";
    }

    bool threw = false;
    try {
        ActivationBuffer buffer(0, capacity);
    } catch (const std::exception&) {
        threw = true;
    }

    if (!threw) {
        // The allocation happened to succeed; nothing to assert about the
        // failure path, but the object must still have cleaned up.
        GTEST_SKIP() << "both allocations fit; the failure path was not exercised";
    }

    const size_t free_after = free_device_memory();
    // Allow a little slack for allocator bookkeeping, but a leaked buffer is
    // ~70% of free memory — orders of magnitude above it.
    const size_t tolerance = free_before / 100;  // 1%
    EXPECT_LE(free_before - free_after, tolerance)
        << "a failed construction leaked " << (free_before - free_after) << " bytes";
}

TEST(ActivationBufferTest, InvalidDeviceThrowsAndLeavesTheDeviceAlone) {
    const int original = current_device();
    ASSERT_EQ(cudaSetDevice(original), cudaSuccess);

    EXPECT_THROW(ActivationBuffer(-1, 256), std::exception);

    // The guard drains the sticky error state the failed call left behind.
    (void)cudaGetLastError();

    EXPECT_EQ(current_device(), original)
        << "a failed construction must not leave the caller on another device";
}

}  // namespace cuda::pipeline::test
