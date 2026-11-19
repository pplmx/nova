/**
 * @file bfs_result_ctor_test.cpp
 * @brief cuda::graph::BFSResult's constructor must not leak when its device allocation fails
 *
 * The constructor allocated two host arrays and two device arrays in sequence:
 *
 *     distances = new int[num_vertices];
 *     visited   = new bool[num_vertices];
 *     CUDA_CHECK(cudaMalloc(&d_distances, ...));   // throws on failure
 *     CUDA_CHECK(cudaMalloc(&d_visited, ...));
 *
 * A throwing constructor does not run its own destructor, so a failure at either cudaMalloc
 * left everything before it unreferenced. The body is now wrapped and calls clear() before
 * rethrowing; the member initialiser list already nulls all four pointers, so clear() is safe
 * on a partially constructed object.
 *
 * These cases cover what can be observed without provoking an allocation failure: an ordinary
 * construction and destruction leaves CUDA usable, and the buffers are distinct per result.
 */

#include <cuda/graph/bfs.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace bfs_result_ctor_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

    }  // namespace

    TEST(BfsResultCtorTest, ConstructingAndDestroyingLeavesCudaUsable) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        {
            cuda::graph::BFSResult result(64);
            EXPECT_EQ(result.num_vertices, 64);
            EXPECT_NE(result.distances, nullptr) << "the host distance array is allocated";
            EXPECT_NE(result.d_visited, nullptr) << "the device visited array is allocated";
        }

        (void)cudaGetLastError();
        int* probe = nullptr;
        ASSERT_EQ(cudaMalloc(&probe, sizeof(int)), cudaSuccess) << "a fresh allocation after destruction must succeed";
        EXPECT_EQ(cudaFree(probe), cudaSuccess);
    }

    TEST(BfsResultCtorTest, EachResultOwnsItsOwnBuffers) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Sharing is what the move constructor explicitly avoids by nulling the source; two
        // independently constructed results must not share either.
        cuda::graph::BFSResult first(32);
        cuda::graph::BFSResult second(32);

        EXPECT_NE(first.d_distances, second.d_distances);
        EXPECT_NE(first.d_visited, second.d_visited);
        EXPECT_NE(first.distances, second.distances);
    }

    TEST(BfsResultCtorTest, MovingTransfersOwnershipWithoutLeavingTheSourceOwningToo) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        int* original_device = nullptr;
        {
            cuda::graph::BFSResult source(16);
            original_device = source.d_distances;
            ASSERT_NE(original_device, nullptr);

            cuda::graph::BFSResult destination(std::move(source));

            EXPECT_EQ(destination.d_distances, original_device);
            EXPECT_EQ(source.d_distances, nullptr) << "the moved-from result must not still hold the pointer";
            EXPECT_EQ(source.distances, nullptr);
        }
        // Both objects went out of scope; only the destination freed anything.
    }

}  // namespace bfs_result_ctor_test
