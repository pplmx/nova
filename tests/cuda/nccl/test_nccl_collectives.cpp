/**
 * @file test_nccl_collectives.cpp
 * @brief Unit tests for NCCL collective operations
 *
 * Tests AllReduce, Broadcast, and Barrier operations with NCCL.
 * Tests are skipped if NCCL is not enabled.
 *
 * These tests drive a real multi-rank NCCL communicator group from a single
 * process: one thread per rank/device, each posting the collective on that
 * rank's communicator. NCCL collectives are per-rank — every rank must post
 * its op before any of them completes — which the old single-call style (one
 * buffer on one device, "device 0's communicator") could never express, and
 * was exactly why these tests had been gated behind NCCL_TESTS_AVAILABLE and
 * never run. Set NCCL_TESTS_AVAILABLE=1 and expose >=2 GPUs to run them.
 */

#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "cuda/nccl/nccl_context.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/nccl/nccl_broadcast.h"
#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_reduce_scatter.h"
#include "cuda/nccl/nccl_group.h"
#include "cuda/nccl/nccl_ops.h"
#include "cuda/nccl/nccl_error.h"

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#if NOVA_NCCL_ENABLED

namespace cuda::nccl {

namespace {

// Barrier so every rank thread enters the collective together: a rank that
// posted its op waits for the others to post before any can complete, so the
// first thread would otherwise burn part of safe_stream_wait's timeout while
// the last thread is still starting up.
class RankBarrier {
public:
    explicit RankBarrier(int count) : total_(count) {}

    void wait() {
        std::unique_lock<std::mutex> lk(mutex_);
        ++arrived_;
        cv_.wait(lk, [this] { return arrived_ >= total_; });
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int total_;
    int arrived_ = 0;
};

// Runs `per_rank(d)` on one thread per visible device, each pinned to its own
// device. cudaSetDevice is per-thread state, so this is the only way to issue
// each rank's part of a multi-device NCCL collective from one process.
//
// Thread exceptions (e.g. a CUDA_CHECK surfacing a sticky device error) are
// collected and rethrown once after all threads join — rethrowing from inside
// join() on the first thread would leave the others running and hit
// std::terminate instead of producing a normal test failure.
template <typename F>
void run_per_rank(int device_count, F&& per_rank) {
    RankBarrier barrier(device_count);
    std::vector<std::thread> threads;
    threads.reserve(device_count);
    std::mutex error_mutex;
    std::exception_ptr first_error;
    for (int d = 0; d < device_count; ++d) {
        threads.emplace_back([d, &barrier, &per_rank, &error_mutex, &first_error]() {
            try {
                CUDA_CHECK(cudaSetDevice(d));
                barrier.wait();
                per_rank(d);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace

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

    constexpr static size_t kCount = 1024;
};

// ============================================================================
// All-Reduce Tests
// ============================================================================

TEST_F(NcclCollectivesTest, AllReduceSum) {
    NcclAllReduce reduce(*context_);
    const int ndev = context_->device_count();

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cuda::memory::Buffer<float> send(kCount);
        cuda::memory::Buffer<float> recv(kCount);
        std::vector<float> h_send(kCount);
        for (size_t i = 0; i < kCount; ++i) {
            h_send[i] = static_cast<float>(i + 1);
        }
        send.copy_from(h_send.data(), kCount);

        auto result = reduce.all_reduce_async(
            send.data(), recv.data(), kCount,
            ncclFloat32, ncclSum,
            stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ASSERT_TRUE(result.ok()) << result.error_message;

        // Each rank provided i+1, so after the all-reduce every rank sees
        // ndev * (i+1).
        std::vector<float> h_recv(kCount);
        recv.copy_to(h_recv.data(), kCount);
        for (size_t i = 0; i < kCount; ++i) {
            EXPECT_FLOAT_EQ(h_recv[i], static_cast<float>(ndev * (i + 1)));
        }
    });
}

TEST_F(NcclCollectivesTest, AllReduceInPlace) {
    NcclAllReduce reduce(*context_);
    const int ndev = context_->device_count();

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cuda::memory::Buffer<float> data(kCount);
        std::vector<float> h_data(kCount);
        for (size_t i = 0; i < kCount; ++i) {
            h_data[i] = static_cast<float>((d + 1) * (i + 1));
        }
        data.copy_from(h_data.data(), kCount);

        auto result = reduce.all_reduce_async(
            data.data(), data.data(), kCount,
            ncclFloat32, ncclSum,
            stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ASSERT_TRUE(result.ok()) << result.error_message;

        // Rank d contributed (d+1)*(i+1); the sum is (i+1)*sum_{r=0}^{ndev-1}(r+1).
        data.copy_to(h_data.data(), kCount);
        const float factor = static_cast<float>(ndev * (ndev + 1) / 2);
        for (size_t i = 0; i < kCount; ++i) {
            EXPECT_FLOAT_EQ(h_data[i], factor * static_cast<float>(i + 1));
        }
    });
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(NcclCollectivesTest, BroadcastFromRoot) {
    NcclBroadcast broadcast(*context_);
    const int ndev = context_->device_count();
    constexpr int kRoot = 0;

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Root seeds the value; non-root buffers are ignored by NCCL.
        cuda::memory::Buffer<float> data(kCount);
        std::vector<float> h_data(kCount, d == kRoot ? 42.0f : -1.0f);
        data.copy_from(h_data.data(), kCount);

        auto result = broadcast.broadcast_async(
            data.data(), data.data(), kCount,
            ncclFloat32, kRoot, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ASSERT_TRUE(result.ok()) << result.error_message;

        // Every rank ends up with the root's value.
        data.copy_to(h_data.data(), kCount);
        for (size_t i = 0; i < kCount; ++i) {
            EXPECT_FLOAT_EQ(h_data[i], 42.0f);
        }
    });
}

// ============================================================================
// Barrier Tests
// ============================================================================

TEST_F(NcclCollectivesTest, BarrierSync) {
    NcclBarrier barrier(*context_);
    const int ndev = context_->device_count();

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        auto result = barrier.barrier_async(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        EXPECT_TRUE(result.ok()) << result.error_message;
    });
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(NcclCollectivesTest, SafeNcclCallDetectsErrors) {
    // safe_nccl_call must surface an immediate NCCL error as an NcclResult
    // (never throw) with a human-readable message.
    auto result = safe_nccl_call(
        []() -> ncclResult_t { return ncclInvalidArgument; },
        context_->current_comm());

    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(result.error_message.empty());
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
    const int ndev = context_->device_count();

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cuda::memory::Buffer<float> send(kCount);
        cuda::memory::Buffer<float> recv(kCount * ndev);
        std::vector<float> h_send(kCount);
        for (size_t i = 0; i < kCount; ++i) {
            h_send[i] = static_cast<float>(d * 1000 + i + 1);  // rank-distinct
        }
        send.copy_from(h_send.data(), kCount);

        auto result = gather.all_gather_async(
            send.data(), recv.data(), kCount,
            ncclFloat32, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ASSERT_TRUE(result.ok()) << result.error_message;

        // recv is [rank0's send | rank1's send | ...] in rank order.
        std::vector<float> h_recv(kCount * ndev);
        recv.copy_to(h_recv.data(), kCount * ndev);
        for (int r = 0; r < ndev; ++r) {
            for (size_t i = 0; i < kCount; ++i) {
                EXPECT_FLOAT_EQ(h_recv[r * kCount + i],
                                static_cast<float>(r * 1000 + i + 1));
            }
        }
    });
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
    const int ndev = context_->device_count();
    constexpr size_t kRecvCount = 1024;
    const size_t send_count = kRecvCount * ndev;

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cuda::memory::Buffer<float> send(send_count);
        cuda::memory::Buffer<float> recv(kRecvCount);
        std::vector<float> h_send(send_count);
        for (size_t j = 0; j < send_count; ++j) {
            h_send[j] = static_cast<float>(d * 1000 + j + 1);
        }
        send.copy_from(h_send.data(), send_count);

        auto result = reduce_scatter.reduce_scatter_async(
            send.data(), recv.data(), kRecvCount,
            ncclFloat32, ncclSum, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        ASSERT_TRUE(result.ok()) << result.error_message;

        // Rank d receives the sum over all ranks of its quotient chunk.
        std::vector<float> h_recv(kRecvCount);
        recv.copy_to(h_recv.data(), kRecvCount);
        for (size_t i = 0; i < kRecvCount; ++i) {
            const size_t idx = d * kRecvCount + i;
            float expected = 0.0f;
            for (int r = 0; r < ndev; ++r) {
                expected += static_cast<float>(r * 1000 + idx + 1);
            }
            EXPECT_FLOAT_EQ(h_recv[i], expected);
        }
    });
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
    const int ndev = context_->device_count();

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Buffers must outlive the group handle: the group's destructor executes
        // the queued ops, so declaring them after the handle would free them
        // first (reverse destruction order) and pass dangling pointers to NCCL.
        cuda::memory::Buffer<float> send(kCount);
        cuda::memory::Buffer<float> recv(kCount);
        std::vector<float> h_send(kCount, 1.0f);
        send.copy_from(h_send.data(), kCount);
        {
            NcclGroupHandle group(*context_, stream);

            group.add_all_reduce(send.data(), recv.data(), kCount, ncclFloat32, ncclSum);
            EXPECT_EQ(group.operation_count(), 1u);
        }  // destructor executes the queued ops

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    });
}

TEST_F(NcclCollectivesTest, GroupHandleExplicitExecute) {
    const int ndev = context_->device_count();

    run_per_rank(ndev, [&](int d) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        cuda::memory::Buffer<float> send(kCount);
        cuda::memory::Buffer<float> recv(kCount);
        std::vector<float> h_send(kCount, 2.0f);
        send.copy_from(h_send.data(), kCount);
        {
            NcclGroupHandle group(*context_, stream);

            group.add_all_reduce(send.data(), recv.data(), kCount, ncclFloat32, ncclSum);
            auto result = group.execute();
            EXPECT_TRUE(result.ok()) << result.error_message;
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
    });
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
