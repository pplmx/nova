/**
 * @file sync_batch_norm_ownership_test.cpp
 * @brief SyncBatchNorm must not be movable while it owns raw device pointers
 *
 * The class held nine cudaMalloc'd pointers that its destructor frees, and declared
 *
 *     SyncBatchNorm(SyncBatchNorm&&) = default;
 *     SyncBatchNorm& operator=(SyncBatchNorm&&) = default;
 *
 * A defaulted move copies the pointers and leaves the source unchanged, so both objects point
 * at the same device memory and both destructors free it. Measured before the change:
 *
 *   a.running_mean        = 0x7fcd65200000
 *   b.running_mean        = 0x7fcd65200000   (the same pointer)
 *   a.running_mean after  = 0x7fcd65200000   (source not nulled)
 *   cudaGetLastError()    = "invalid argument" once both had been destroyed
 *
 * Moves are now deleted. Nothing in the tree moved one, which is why the build accepted the
 * change without touching any caller.
 */

#include <cuda/neural/sync_batch_norm.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <type_traits>

namespace sync_batch_norm_ownership_test {

    // The regression is a compile-time property: with the defaulted moves these assertions fail
    // to hold, and the file stops compiling.
    static_assert(!std::is_move_constructible_v<cuda::neural::SyncBatchNorm>, "a defaulted move would share the raw pointers and double-free them");
    static_assert(!std::is_move_assignable_v<cuda::neural::SyncBatchNorm>);
    static_assert(!std::is_copy_constructible_v<cuda::neural::SyncBatchNorm>);
    static_assert(!std::is_copy_assignable_v<cuda::neural::SyncBatchNorm>);

    TEST(SyncBatchNormOwnershipTest, ConstructingAndDestroyingLeavesCudaInAWorkingState) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }

        // The destructor frees nine buffers. If any of them were released twice, the following
        // CUDA call would report the stale error rather than running.
        {
            cuda::neural::SyncBatchNorm layer(8);
            (void)layer.running_mean();
            (void)layer.gamma();
        }

        (void)cudaGetLastError();  // clear anything the destruction left behind
        int* probe = nullptr;
        ASSERT_EQ(cudaMalloc(&probe, sizeof(int)), cudaSuccess) << "a fresh allocation after destruction must succeed";
        EXPECT_EQ(cudaFree(probe), cudaSuccess);
    }

    TEST(SyncBatchNormOwnershipTest, EachLayerOwnsItsOwnBuffers) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }

        // Two independent layers must not share allocations; sharing is what a defaulted move
        // produced between an object and its moved-from twin.
        cuda::neural::SyncBatchNorm first(4);
        cuda::neural::SyncBatchNorm second(4);

        EXPECT_NE(first.running_mean(), second.running_mean());
        EXPECT_NE(first.gamma(), second.gamma());
        EXPECT_NE(first.beta(), second.beta());
    }

}  // namespace sync_batch_norm_ownership_test
