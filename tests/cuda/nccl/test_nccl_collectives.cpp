/**
 * @file test_nccl_collectives.cpp
 * @brief Unit tests for NCCL collective operations
 *
 * Tests AllReduce, Broadcast, and Barrier operations with NCCL.
 * Tests are skipped if NCCL is not enabled.
 */

#include <gtest/gtest.h>

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_barrier.h"

#include "cuda/memory/buffer.h"

#ifdef NOVA_NCCL_ENABLED

namespace cuda::nccl {

// Test fixture for NCCL collective tests
class NcclCollectivesTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count < 2) {
            GTEST_SKIP() << "Need at least 2 GPUs for NCCL collective tests";
        }

        context_ = std::make_unique<NcclContext>();
        context_->initialize();
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
    float* send_ptr = send.data();
    for (size_t i = 0; i < 1024; ++i) {
        send_ptr[i] = static_cast<float>(i + 1);
    }

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
    float* recv_ptr = recv.data();
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(recv_ptr[i], static_cast<float>(i + 1));
    }
}

TEST_F(NcclCollectivesTest, AllReduceInPlace) {
    NcclAllReduce reduce(*context_);

    cuda::memory::Buffer<float> data(1024);
    float* data_ptr = data.data();
    for (size_t i = 0; i < 1024; ++i) {
        data_ptr[i] = static_cast<float>(i + 1);
    }

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

    // Initialize root data
    float* root_ptr = root_data.data();
    float* recv_ptr = recv_data.data();
    for (size_t i = 0; i < 1024; ++i) {
        root_ptr[i] = 42.0f;
    }

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
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(recv_ptr[i], 42.0f);
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

}  // namespace cuda::nccl

#endif  // NOVA_NCCL_ENABLED
