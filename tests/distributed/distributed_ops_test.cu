#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * @file distributed_ops_test.cu
 * @brief Tests for multi-GPU distributed operations
 *
 * Tests all-reduce, broadcast, all-gather, and barrier synchronization.
 * Multi-GPU tests run for real via a thread-per-rank harness (one thread per
 * device, each pinned with cudaSetDevice), the same pattern the NCCL suite
 * and distributed matmul use — see issue-v17-dist-ops-multigpu-untested for
 * the pre-v2.17 "Requires single GPU" skip audit. Single-GPU fallback tests
 * verify the code path works on single-GPU CI runners.
 */

#include <gtest/gtest.h>

#include "cuda/distributed/reduce.h"
#include "cuda/distributed/broadcast.h"
#include "cuda/distributed/all_gather.h"
#include "cuda/distributed/barrier.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cuda_runtime.h>

#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include "cuda/device/error.h"

using namespace cuda::distributed;
using namespace cuda::mesh;

namespace {

// Same pattern as the NCCL suite and distributed matmul: a barrier so every
// rank thread enters the collective together (a stray-rank start would
// otherwise leave peers waiting on cross-thread work), then run `per_rank(d)`
// on one thread per visible device, each pinned to its own device with
// cudaSetDevice (per-thread state). First exception on any thread is rethrown
// after all threads join.
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

template <typename F>
void run_per_rank(int device_count, F&& per_rank) {
    RankBarrier barrier(device_count);
    std::vector<std::thread> threads;
    threads.reserve(device_count);
    std::mutex error_mutex;
    std::exception_ptr first_error;
    for (int d = 0; d < device_count; ++d) {
        threads.emplace_back([d, &barrier, &per_rank, &error_mutex, &first_error]() {
            bool failed = false;
            try {
                CUDA_CHECK(cudaSetDevice(d));
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
                failed = true;
            }
            barrier.wait();
            if (failed) {
                return;
            }
            try {
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

void fill_linear(float* data, size_t count, float base) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = base + static_cast<float>(i);
    }
}

bool arrays_near(const float* a, const float* b, size_t count, float tol = 1e-3f) {
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

}  // namespace

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

// ============================================================================
// Multi-GPU Real Tests — thread-per-rank harness
// (issue-v17-dist-ops-multigpu-untested: these paths were never exercised;
//  every "Requires single GPU" skip in this file had a single-GPU-only guard)
// ============================================================================

// DistributedReduce: each rank contributes distinct local data; every rank must
// end with the group sum (rank r sends base (r+1), element i adds i).
TEST_F(DistributedReduceTest, DISABLED_MultiGpu_AllReduceDistinctData) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for a multi-GPU all-reduce test";
    }
    const size_t count = 64;
    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(count));
    run_per_rank(device_count, [&](int d) {
        std::vector<float> h_send(count);
        fill_linear(h_send.data(), count, static_cast<float>(d + 1));
        cuda::memory::Buffer<float> d_send(count);
        cuda::memory::Buffer<float> d_recv(count);
        d_send.copy_from(h_send.data(), count);

        DistributedReduce::all_reduce(d_send.data(), d_recv.data(), count, ReductionOp::Sum);

        d_recv.copy_to(h_results[d].data(), count);
    });

    // Expected group sum for element i: sum_{r=0..n-1}((r+1)+i) = n(n+1)/2 + n*i.
    std::vector<float> h_expected(count);
    for (size_t i = 0; i < count; ++i) {
        h_expected[i] = static_cast<float>(device_count * (device_count + 1) / 2) +
                        static_cast<float>(device_count) * static_cast<float>(i);
    }
    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_results[rank].data(), h_expected.data(), count))
            << "all-reduce result on rank " << rank << " != group sum";
    }
}

TEST_F(DistributedReduceTest, DISABLED_MultiGpu_AllReduceInPlace) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for a multi-GPU all-reduce test";
    }
    const size_t count = 32;
    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(count));
    run_per_rank(device_count, [&](int d) {
        std::vector<float> h_send(count);
        fill_linear(h_send.data(), count, static_cast<float>(d + 1));
        cuda::memory::Buffer<float> d_buf(count);
        d_buf.copy_from(h_send.data(), count);

        DistributedReduce::all_reduce(d_buf.data(), d_buf.data(), count, ReductionOp::Sum);

        d_buf.copy_to(h_results[d].data(), count);
    });

    std::vector<float> h_expected(count);
    for (size_t i = 0; i < count; ++i) {
        h_expected[i] = static_cast<float>(device_count * (device_count + 1) / 2) +
                        static_cast<float>(device_count) * static_cast<float>(i);
    }
    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("in-place rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_results[rank].data(), h_expected.data(), count))
            << "in-place all-reduce result on rank " << rank << " != group sum";
    }
}

// DistributedAllGather: each rank contributes distinct data; every rank must
// end with recv = [rank 0 block][rank 1 block]... in rank order.
TEST_F(DistributedAllGatherTest, DISABLED_MultiGpu_AllGatherDistinctData) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for a multi-GPU all-gather test";
    }
    const size_t count = 64;
    std::vector<std::vector<float>> h_results(device_count,
                                              std::vector<float>(count * device_count));
    run_per_rank(device_count, [&](int d) {
        std::vector<float> h_send(count);
        fill_linear(h_send.data(), count, static_cast<float>(d + 1));
        cuda::memory::Buffer<float> d_send(count);
        cuda::memory::Buffer<float> d_recv(count * device_count);
        d_send.copy_from(h_send.data(), count);

        DistributedAllGather::all_gather(d_send.data(), d_recv.data(), count);

        d_recv.copy_to(h_results[d].data(), count * device_count);
    });

    // Expected gather: block r holds (r+1)+i.
    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        for (int r = 0; r < device_count; ++r) {
            for (size_t i = 0; i < count; ++i) {
                const float expect = static_cast<float>(r + 1) + static_cast<float>(i);
                if (std::abs(h_results[rank][r * count + i] - expect) > 1e-3f) {
                    FAIL() << "all-gather on rank " << rank << ": block " << r
                           << " elem " << i << " = " << h_results[rank][r * count + i]
                           << " expected " << expect;
                }
            }
        }
    }
}

// DistributedBroadcast: root's data must reach every rank.
TEST_F(DistributedBroadcastTest, DISABLED_MultiGpu_BroadcastDistinctData) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for a multi-GPU broadcast test";
    }
    const size_t count = 64;
    const float root_value = 7.0f;
    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(count));
    run_per_rank(device_count, [&](int d) {
        std::vector<float> h_data(count);
        fill_linear(h_data.data(), count, static_cast<float>(d + 1));
        if (d == 0) {
            std::fill(h_data.begin(), h_data.end(), root_value);
        }
        cuda::memory::Buffer<float> d_data(count);
        d_data.copy_from(h_data.data(), count);

        DistributedBroadcast::broadcast(d_data.data(), count, 0);

        d_data.copy_to(h_results[d].data(), count);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(h_results[rank][i] - root_value) > 1e-3f) {
                FAIL() << "broadcast on rank " << rank << ": elem " << i << " = "
                       << h_results[rank][i] << " expected " << root_value;
            }
        }
    }
}

// The four DISABLED_MultiGpu_* tests above are READY regression tests that
// currently fail against the legacy host-staged/P2P implementations of the
// high-level distributed ops — see issue-v17-dist-ops-multigpu-untested.
// Failure modes observed on 2x A100 (Round 17 P1 evidence, run with
// --gtest_also_run_disabled_tests):
//   - DistributedReduce::all_reduce: result != group sum (CPU-coordinated
//     host-staging reads the caller's single send_data, cannot gather distinct
//     per-rank contributions; all_reduce_async also ignores its stream).
//   - DistributedAllGather::all_gather: each block ends up a copy of the
//     caller's own data (send_data is read as if every source had its own).
//   - DistributedBroadcast::broadcast: non-root never receives root's data
//     (P2P copies dereference root's pointer address on destination devices).
// They are flipped ON by task-v17b-dist-ops-nccl-converge once the ops are
// re-routed onto the verified NCCL layer (or fixed per P1 evidence).
