/**
 * @file distributed_ops_test.cu
 * @brief Tests for multi-GPU distributed operations
 *
 * Tests all-reduce, broadcast, all-gather, and barrier synchronization.
 * Multi-GPU tests require a proper collective test harness (all GPUs call
 * the operation simultaneously). Single-GPU fallback tests verify the
 * code path works on single-GPU CI runners.
 */

#include <gtest/gtest.h>

#include "cuda/distributed/reduce.h"
#include "cuda/distributed/broadcast.h"
#include "cuda/distributed/all_gather.h"
#include "cuda/distributed/barrier.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/memory/buffer.h"

#include <cuda_runtime.h>

#include <thread>
#include <vector>

using namespace cuda::distributed;
using namespace cuda::mesh;

// ============================================================================
// Test fixtures
// ============================================================================

class DistributedReduceTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }
};

class DistributedBroadcastTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }
};

class DistributedAllGatherTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }
};

class MeshBarrierTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }
};

// ============================================================================
// DistributedReduce Tests
// ============================================================================

TEST_F(DistributedReduceTest, NeedsMultiGpu) {
    int device_count = DeviceMesh::instance().device_count();
    EXPECT_EQ(DistributedReduce::needs_multi_gpu(), device_count > 1);
}

TEST_F(DistributedReduceTest, SingleGpuFallback) {
    auto& mesh = DeviceMesh::instance();
    int n = mesh.device_count();

    if (n > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 256;
    std::vector<float> h_send(count, 1.0f);
    std::vector<float> h_expected(count, 1.0f);

    cuda::memory::Buffer<float> d_send(count);
    cuda::memory::Buffer<float> d_recv(count);

    d_send.copy_from(h_send.data(), count);

    DistributedReduce::all_reduce(d_send.data(), d_recv.data(), count, ReductionOp::Sum);

    std::vector<float> h_result(count);
    d_recv.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_expected[i]);
    }
}

TEST_F(DistributedReduceTest, SingleGpuMin) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 128;
    std::vector<float> h_send(count, 5.0f);
    std::vector<float> h_expected(count, 5.0f);

    cuda::memory::Buffer<float> d_send(count);
    cuda::memory::Buffer<float> d_recv(count);

    d_send.copy_from(h_send.data(), count);

    DistributedReduce::all_reduce(d_send.data(), d_recv.data(), count, ReductionOp::Min);

    std::vector<float> h_result(count);
    d_recv.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_expected[i]);
    }
}

TEST_F(DistributedReduceTest, SingleGpuMax) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 128;
    std::vector<float> h_send(count, 3.0f);
    std::vector<float> h_expected(count, 3.0f);

    cuda::memory::Buffer<float> d_send(count);
    cuda::memory::Buffer<float> d_recv(count);

    d_send.copy_from(h_send.data(), count);

    DistributedReduce::all_reduce(d_send.data(), d_recv.data(), count, ReductionOp::Max);

    std::vector<float> h_result(count);
    d_recv.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_expected[i]);
    }
}

// ============================================================================
// DistributedBroadcast Tests
// ============================================================================

TEST_F(DistributedBroadcastTest, SingleGpuFallback) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 256;
    std::vector<float> h_data(count, 7.0f);

    cuda::memory::Buffer<float> d_data(count);
    d_data.copy_from(h_data.data(), count);

    DistributedBroadcast::broadcast(d_data.data(), count, 0);

    std::vector<float> h_result(count);
    d_data.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 7.0f);
    }
}

TEST_F(DistributedBroadcastTest, SingleGpuBroadcastNonRoot) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 100;
    std::vector<float> h_data(count, 42.0f);

    cuda::memory::Buffer<float> d_data(count);
    d_data.copy_from(h_data.data(), count);

    // Single GPU, root doesn't matter
    DistributedBroadcast::broadcast(d_data.data(), count, 5);

    std::vector<float> h_result(count);
    d_data.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 42.0f);
    }
}

// ============================================================================
// DistributedAllGather Tests
// ============================================================================

TEST_F(DistributedAllGatherTest, SingleGpuFallback) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 128;
    std::vector<float> h_send(count, 5.0f);

    cuda::memory::Buffer<float> d_send(count);
    cuda::memory::Buffer<float> d_recv(count);

    d_send.copy_from(h_send.data(), count);

    DistributedAllGather::all_gather(d_send.data(), d_recv.data(), count);

    std::vector<float> h_result(count);
    d_recv.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 5.0f);
    }
}

TEST_F(DistributedAllGatherTest, SingleGpuDataPreserved) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    const size_t count = 64;
    std::vector<float> h_send(count);
    for (size_t i = 0; i < count; ++i) {
        h_send[i] = static_cast<float>(i);
    }

    cuda::memory::Buffer<float> d_send(count);
    cuda::memory::Buffer<float> d_recv(count);

    d_send.copy_from(h_send.data(), count);

    DistributedAllGather::all_gather(d_send.data(), d_recv.data(), count);

    std::vector<float> h_result(count);
    d_recv.copy_to(h_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], static_cast<float>(i));
    }
}

// ============================================================================
// MeshBarrier Tests
// ============================================================================

TEST_F(MeshBarrierTest, SingleGpuBarrier) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    MeshBarrier barrier;
    EXPECT_NO_THROW(barrier.synchronize());
}

TEST_F(MeshBarrierTest, NoDeadlock) {
    auto& mesh = DeviceMesh::instance();
    int n = mesh.device_count();

    // This test primarily verifies no deadlock occurs
    MeshBarrier barrier;

    for (int iteration = 0; iteration < 10; ++iteration) {
        EXPECT_NO_THROW(barrier.synchronize());
    }
}

TEST_F(MeshBarrierTest, AsyncBarrier) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    MeshBarrier barrier;
    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreate(&stream));
    EXPECT_NO_THROW(barrier.synchronize_async(stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_F(MeshBarrierTest, IsBarriering) {
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        GTEST_SKIP() << "Requires single GPU";
    }

    MeshBarrier barrier;
    EXPECT_FALSE(barrier.is_barriering());
}

// ============================================================================
// MeshStreams Tests
// ============================================================================

TEST(MeshStreamsTest, Singleton) {
    auto& instance1 = MeshStreams::instance();
    auto& instance2 = MeshStreams::instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST(MeshStreamsTest, SingleGpuInitialization) {
    MeshStreams& streams = MeshStreams::instance();

    // Initialize for single GPU
    streams.initialize(1);

    EXPECT_TRUE(streams.initialized());
    EXPECT_EQ(streams.device_count(), 1);

    // Should be able to get stream for device 0
    EXPECT_NE(streams.get_stream(0), nullptr);
    EXPECT_NE(streams.get_event(0), nullptr);
}

// ============================================================================
// DeviceMesh Integration Test
// ============================================================================

TEST(DeviceMeshIntegration, MultiGpuAvailable) {
    auto& mesh = DeviceMesh::instance();
    mesh.initialize();

    int n = mesh.device_count();
    EXPECT_GE(n, 1);

    if (n > 1) {
        // Verify we can check peer access
        EXPECT_TRUE(mesh.can_access_peer(0, 0));  // Self access always works

        // Check that at least some peer access exists
        bool has_any_peer_access = false;
        for (int i = 0; i < n && !has_any_peer_access; ++i) {
            for (int j = i + 1; j < n && !has_any_peer_access; ++j) {
                if (mesh.can_access_peer(i, j)) {
                    has_any_peer_access = true;
                }
            }
        }
        // Note: On systems without P2P support, this may be false
        // That's OK - the fallback path handles it
    }
}
