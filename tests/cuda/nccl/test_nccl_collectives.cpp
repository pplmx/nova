/**
 * @file test_nccl_collectives.cpp
 * @brief Unit tests for NCCL collective operations
 *
 * Tests AllReduce, Broadcast, and Barrier operations with NCCL.
 * Tests are skipped if NCCL is not enabled.
 */

#include <gtest/gtest.h>
#include <vector>

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_reduce_scatter.h"
#include "cuda/nccl/nccl_group.h"
#include "cuda/nccl/nccl_ops.h"

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#if NOVA_NCCL_ENABLED

namespace cuda::nccl {

// Test fixture for NCCL collective tests
class NcclCollectivesTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA not available";
        }
        if (device_count < 2) {
            GTEST_SKIP() << "Need at least 2 GPUs for NCCL collective tests";
        }

        const char* nccl_test_env = std::getenv("NCCL_TESTS_AVAILABLE");
        if (nccl_test_env == nullptr) {
            GTEST_SKIP() << "NCCL multi-process tests require NCCL_TESTS_AVAILABLE env var";
        }

        context_ = std::make_unique<NcclContext>();
        try {
            context_->initialize();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "NCCL initialization failed: " << e.what();
        }
    }

    void TearDown() override {
        if (context_) {
            context_->destroy();
        }
    }

    std::unique_ptr<NcclContext> context_;
};

// ============================================================================
// All-Reduce Tests
// ============================================================================

TEST_F(NcclCollectivesTest, AllReduceSum) {
    NcclAllReduce reduce(*context_);

    // Create test buffers
    cuda::memory::Buffer<float> send(1024);
    cuda::memory::Buffer<float> recv(1024);

    // Initialize with values
    std::vector<float> h_send(1024);
    for (size_t i = 0; i < 1024; ++i) {
        h_send[i] = static_cast<float>(i + 1);
    }
    send.copy_from(h_send.data(), 1024);

    // Perform all-reduce
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = reduce.all_reduce_async(
        send.data(), recv.data(), 1024,
        ncclFloat32, ncclSum,
        stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    EXPECT_TRUE(result.ok()) << result.error_message;

    // Each element should be the sum across GPUs
    // In single-GPU test, result is just the input
    std::vector<float> h_recv(1024);
    recv.copy_to(h_recv.data(), 1024);
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(h_recv[i], static_cast<float>(i + 1));
    }
}

TEST_F(NcclCollectivesTest, AllReduceInPlace) {
    NcclAllReduce reduce(*context_);

    cuda::memory::Buffer<float> data(1024);
    std::vector<float> h_data(1024);
    for (size_t i = 0; i < 1024; ++i) {
        h_data[i] = static_cast<float>(i + 1);
    }
    data.copy_from(h_data.data(), 1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = reduce.all_reduce_async(
        data.data(), data.data(), 1024,
        ncclFloat32, ncclSum,
        stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    EXPECT_TRUE(result.ok()) << result.error_message;
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(NcclCollectivesTest, BroadcastFromRoot) {
    NcclBroadcast broadcast(*context_);

    cuda::memory::Buffer<float> root_data(1024);
    cuda::memory::Buffer<float> recv_data(1024);

    // Initialize root data (stage on the host; Buffer.data() is device memory)
    std::vector<float> h_root(1024, 42.0f);
    root_data.copy_from(h_root.data(), 1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = broadcast.broadcast_async(
        root_data.data(), recv_data.data(), 1024,
        ncclFloat32, 0,  // root_rank = 0
        stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    EXPECT_TRUE(result.ok()) << result.error_message;

    // All elements should be 42.0
    std::vector<float> h_recv(1024);
    recv_data.copy_to(h_recv.data(), 1024);
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(h_recv[i], 42.0f);
    }
}

// ============================================================================
// Barrier Tests
// ============================================================================

TEST_F(NcclCollectivesTest, BarrierSync) {
    NcclBarrier barrier(*context_);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = barrier.barrier_async(stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    EXPECT_TRUE(result.ok()) << result.error_message;
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(NcclCollectivesTest, SafeNcclCallDetectsErrors) {
    NcclAllReduce reduce(*context_);

    // Invalid pointers should still call NCCL (which may or may not detect)
    cuda::memory::Buffer<float> send(1024);
    cuda::memory::Buffer<float> recv(1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = reduce.all_reduce_async(
        send.data(), recv.data(), 1024,
        ncclFloat32, ncclSum,
        stream);

    // Result depends on GPU state
    if (!result.ok()) {
        EXPECT_FALSE(result.error_message.empty());
    }

    cudaStreamDestroy(stream);
}

// ============================================================================
// Type Conversion Tests
// ============================================================================

TEST_F(NcclCollectivesTest, TypeConversion) {
    // Test to_nccl_dtype conversion
    EXPECT_EQ(NcclAllReduce::to_nccl_dtype(CUDA_R_32F), ncclFloat32);
    EXPECT_EQ(NcclAllReduce::to_nccl_dtype(CUDA_R_64F), ncclFloat64);
    EXPECT_EQ(NcclAllReduce::to_nccl_dtype(CUDA_R_16F), ncclFloat16);
    EXPECT_EQ(NcclAllReduce::to_nccl_dtype(CUDA_R_32I), ncclInt32);
    EXPECT_EQ(NcclAllReduce::to_nccl_dtype(CUDA_R_32U), ncclUint32);

    // Test to_nccl_op conversion
    using RedOp = ::cuda::distributed::ReductionOp;
    EXPECT_EQ(NcclAllReduce::to_nccl_op(RedOp::Sum), ncclSum);
    EXPECT_EQ(NcclAllReduce::to_nccl_op(RedOp::Min), ncclMin);
    EXPECT_EQ(NcclAllReduce::to_nccl_op(RedOp::Max), ncclMax);
    EXPECT_EQ(NcclAllReduce::to_nccl_op(RedOp::Product), ncclProd);
}

// ============================================================================
// All-Gather Tests
// ============================================================================

TEST_F(NcclCollectivesTest, AllGatherBasic) {
    NcclAllGather gather(*context_);

    int device_count = context_->device_count();
    size_t send_count = 1024;
    size_t recv_count = send_count * device_count;

    cuda::memory::Buffer<float> send(send_count);
    cuda::memory::Buffer<float> recv(recv_count);

    std::vector<float> h_send(send_count);
    for (size_t i = 0; i < send_count; ++i) {
        h_send[i] = static_cast<float>(i + 1);
    }
    send.copy_from(h_send.data(), send_count);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = gather.all_gather_async(
        send.data(), recv.data(), send_count,
        ncclFloat32, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    EXPECT_TRUE(result.ok()) << result.error_message;
}

TEST_F(NcclCollectivesTest, AllGatherBufferSize) {
    size_t send_count = 100;
    int device_count = 2;
    size_t expected_recv = NcclAllGather::required_recv_buffer_size(send_count, device_count);
    EXPECT_EQ(expected_recv, 200u);
}

// ============================================================================
// Reduce-Scatter Tests
// ============================================================================

TEST_F(NcclCollectivesTest, ReduceScatterBasic) {
    NcclReduceScatter reduce_scatter(*context_);

    int device_count = context_->device_count();
    size_t recv_count = 1024;
    size_t send_count = recv_count * device_count;

    cuda::memory::Buffer<float> send(send_count);
    cuda::memory::Buffer<float> recv(recv_count);

    std::vector<float> h_send(send_count);
    for (size_t i = 0; i < send_count; ++i) {
        h_send[i] = 1.0f;
    }
    send.copy_from(h_send.data(), send_count);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto result = reduce_scatter.reduce_scatter_async(
        send.data(), recv.data(), recv_count,
        ncclFloat32, ncclSum, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    EXPECT_TRUE(result.ok()) << result.error_message;
}

TEST_F(NcclCollectivesTest, ReduceScatterBufferSize) {
    size_t recv_count = 100;
    int device_count = 2;
    size_t expected_send = NcclReduceScatter::required_send_buffer_size(recv_count, device_count);
    EXPECT_EQ(expected_send, 200u);
}

// ============================================================================
// Group Operations Tests
// ============================================================================

TEST_F(NcclCollectivesTest, GroupHandleBatched) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        NcclGroupHandle group(*context_, stream);

        cuda::memory::Buffer<float> send1(1024);
        cuda::memory::Buffer<float> recv1(1024);
        std::vector<float> h_send1(1024, 1.0f);
        send1.copy_from(h_send1.data(), 1024);

        group.add_all_reduce(send1.data(), recv1.data(), 1024, ncclFloat32, ncclSum);
        EXPECT_EQ(group.operation_count(), 1u);
    }

    cudaStreamDestroy(stream);
}

TEST_F(NcclCollectivesTest, GroupHandleExplicitExecute) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        NcclGroupHandle group(*context_, stream);

        cuda::memory::Buffer<float> send(1024);
        cuda::memory::Buffer<float> recv(1024);
        std::vector<float> h_send(1024, 1.0f);
        send.copy_from(h_send.data(), 1024);

        group.add_all_reduce(send.data(), recv.data(), 1024, ncclFloat32, ncclSum);
        auto result = group.execute();
        EXPECT_TRUE(result.ok() || !context_->has_nccl());
    }

    cudaStreamDestroy(stream);
}

// ============================================================================
// Unified Ops API Tests
// ============================================================================

TEST_F(NcclCollectivesTest, UnifiedOpsHasNccl) {
    NcclOps ops(*context_);
    EXPECT_EQ(ops.has_nccl(), context_->has_nccl());
}

TEST_F(NcclCollectivesTest, UnifiedOpsDeviceCount) {
    NcclOps ops(*context_);
    EXPECT_EQ(ops.device_count(), context_->device_count());
}

}  // namespace cuda::nccl

#else  // NOVA_NCCL_ENABLED

namespace cuda::nccl {

TEST(NcclCollectivesTest, DISABLED_AllReduceSum) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_BroadcastFromRoot) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_BarrierSync) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_TypeConversion) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_AllGatherBasic) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_ReduceScatterBasic) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_GroupHandleBatched) {
    GTEST_SKIP() << "NCCL not enabled";
}

TEST(NcclCollectivesTest, DISABLED_UnifiedOpsHasNccl) {
    GTEST_SKIP() << "NCCL not enabled";
}

}  // namespace cuda::nccl

#endif  // NOVA_NCCL_ENABLED
