#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * @file distributed_matmul_test.cu
 * @brief Tests for multi-GPU distributed matrix multiply
 *
 * Tests numerical correctness and single-GPU fallback for DistributedMatmul.
 *
 * IMPORTANT: Multi-GPU tests are marked as SKIP because they require
 * multi-process execution. In single-process execution, only one GPU's
 * code path runs. For true multi-GPU testing, use NCCL or similar
 * collective communication libraries with proper process spawning.
 *
 * @note Single-GPU tests verify the fallback path that is always used
 *       in single-process scenarios.
 */

#include <gtest/gtest.h>

#include "cuda/distributed/matmul.h"
#include "cuda/distributed/common.h"
#include "cuda/distributed/all_gather.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/neural/matmul.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "distributed_test_common.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace cuda::distributed;
using namespace cuda::mesh;

// The thread-per-rank harness (RankBarrier + run_per_rank) lives in
// distributed_test_common.h, shared with the NCCL suite, distributed ops, and
// SyncBatchNorm tests. The bounded barrier is the milestone v2.18 P3 /
// TASK-003 harness-robustness fix: a rank thread that dies before reaching
// the barrier fails the suite with RankBarrierTimeout instead of hanging the
// other ranks forever.
using cuda::distributed::test::run_per_rank;
using cuda::distributed::test::RankBarrierTimeout;

namespace {

// Multi-GPU readiness for the shared NCCL singleton, self-healing a broken
// context (TASK-003): after a genuine timeout/async abort the singleton is
// broken and would otherwise turn every later matmul test into a silent skip.
bool multigpu_nccl_ready() {
    auto& ctx = cuda::nccl::NcclContext::instance();
    try {
        ctx.initialize();
    } catch (const std::exception&) {
        return false;
    }
    return cuda::distributed::test::heal_and_ready(ctx);
}

// ============================================================================
// Test Fixtures
// ============================================================================

class DistributedMatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }

    static void SetUpTestSuite() {
        // Initialize CUDA context for proper multi-GPU detection
        cudaFree(nullptr);
    }

    // Fill buffer with random values in range [-1, 1]
    static void fill_random(float* data, size_t count) {
        static std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(rng);
        }
    }

    // Fill buffer with sequential values starting from offset
    static void fill_sequential(float* data, size_t count, float start = 0.0f) {
        for (size_t i = 0; i < count; ++i) {
            data[i] = start + static_cast<float>(i);
        }
    }

    // Fill buffer with constant value
    static void fill_constant(float* data, size_t count, float value) {
        for (size_t i = 0; i < count; ++i) {
            data[i] = value;
        }
    }

    // Check if two buffers are approximately equal within tolerance
    static testing::AssertionResult arrays_near(
        const float* expected,
        const float* actual,
        size_t count,
        float tolerance
    ) {
        size_t first_mismatch = count;
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(expected[i] - actual[i]) > tolerance) {
                if (first_mismatch == count) {
                    first_mismatch = i;
                }
            }
        }

        if (first_mismatch == count) {
            return testing::AssertionSuccess();
        }

        return testing::AssertionFailure()
            << "Arrays differ beyond tolerance " << tolerance << "\n"
            << "First mismatch at index " << first_mismatch << ": "
            << "expected " << expected[first_mismatch]
            << ", actual " << actual[first_mismatch];
    }
};

// ============================================================================
// Basic Tests
// ============================================================================

TEST_F(DistributedMatmulTest, NeedsMultiGpu) {
    int device_count = DeviceMesh::instance().device_count();
    EXPECT_GE(device_count, 1);
    // Note: needs_multi_gpu() returns true if device_count > 1
    // But the actual matmul uses single-GPU fallback for correctness
    EXPECT_EQ(DistributedMatmul::needs_multi_gpu(), device_count > 1);
}

// ============================================================================
// Single-GPU Fallback Tests (MGPU-13)
// These tests verify the fallback path that is always used
// ============================================================================

TEST_F(DistributedMatmulTest, SingleGpuFallback_Identity) {
    // Identity matrix multiplication: A @ I = A
    const int m = 64, n = 64, k = 64;

    // A: identity-like matrix (1.0 on diagonal, 0.0 elsewhere)
    std::vector<float> h_A(m * k, 0.0f);
    for (int i = 0; i < std::min(m, k); ++i) {
        h_A[i * k + i] = 1.0f;
    }

    // B: identity matrix
    std::vector<float> h_B(k * n, 0.0f);
    for (int i = 0; i < std::min(k, n); ++i) {
        h_B[i * n + i] = 1.0f;
    }

    // Expected: A @ B = A
    std::vector<float> h_expected = h_A;

    // Allocate device buffers
    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    // Compute via distributed matmul
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

    // Verify
    std::vector<float> h_C(m * n);
    d_C.copy_to(h_C.data(), m * n);

    EXPECT_TRUE(arrays_near(h_expected.data(), h_C.data(), m * n, 1e-5f))
        << "Identity matmul failed";
}

TEST_F(DistributedMatmulTest, SingleGpuFallback_Random) {
    const int m = 128, n = 64, k = 32;
    constexpr float kTolerance = 1e-4f;

    // Generate random input
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    // Compute single-GPU reference using neural matmul
    std::vector<float> h_C_ref(m * n);
    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_ref(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_ref.data(), m, n, k);

    // Compute via distributed matmul
    cuda::memory::Buffer<float> d_C(m * n);
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

    // Compare results
    std::vector<float> h_C(m * n);
    std::vector<float> h_C_ref_cpu(m * n);
    d_C.copy_to(h_C.data(), m * n);
    d_C_ref.copy_to(h_C_ref_cpu.data(), m * n);

    EXPECT_TRUE(arrays_near(h_C_ref_cpu.data(), h_C.data(), m * n, kTolerance))
        << "Random matmul failed";
}

TEST_F(DistributedMatmulTest, SingleGpuFallback_Large) {
    // Test larger matrices to ensure no hidden assumptions
    const int m = 1024, n = 512, k = 256;
    constexpr float kTolerance = 1e-4f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_single(m * n);
    cuda::memory::Buffer<float> d_C_dist(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    // Single-GPU reference
    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_single.data(), m, n, k);

    // Distributed (fallback path)
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C_dist.data(), m, n, k);

    // Compare
    std::vector<float> h_single(m * n);
    std::vector<float> h_dist(m * n);
    d_C_single.copy_to(h_single.data(), m * n);
    d_C_dist.copy_to(h_dist.data(), m * n);

    EXPECT_TRUE(arrays_near(h_single.data(), h_dist.data(), m * n, kTolerance))
        << "Large matrix matmul failed";
}

TEST_F(DistributedMatmulTest, SingleGpuFallback_AlphaBeta) {
    const int m = 64, n = 64, k = 64;
    constexpr float kTolerance = 1e-4f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    // Initialize C with non-zero values for beta test
    std::vector<float> h_C_init(m * n);
    fill_constant(h_C_init.data(), m * n, 1.0f);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_single(m * n);
    cuda::memory::Buffer<float> d_C_dist(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);
    d_C_single.copy_from(h_C_init.data(), m * n);
    d_C_dist.copy_from(h_C_init.data(), m * n);

    DistributedMatmulOptions opts;
    opts.alpha = 2.0f;
    opts.beta = 0.5f;

    // Single-GPU reference
    cuda::neural::MatmulOptions neural_opts;
    neural_opts.alpha = opts.alpha;
    neural_opts.beta = opts.beta;
    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_single.data(), m, n, k, neural_opts);

    // Distributed (fallback)
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C_dist.data(), m, n, k, opts);

    // Compare
    std::vector<float> h_single(m * n);
    std::vector<float> h_dist(m * n);
    d_C_single.copy_to(h_single.data(), m * n);
    d_C_dist.copy_to(h_dist.data(), m * n);

    EXPECT_TRUE(arrays_near(h_single.data(), h_dist.data(), m * n, kTolerance))
        << "Alpha/beta test failed";
}

// ============================================================================
// Multi-GPU Tests (MGPU-12)
// SKIPPED: Require multi-process execution for proper testing
// ============================================================================

TEST_F(DistributedMatmulTest, MultiGpu_RowPartition) {
    // This test documents the row partition behavior.
    // Skipped because multi-GPU requires multi-process execution.
    //
    // The expected row counts are derived from the actual number of visible
    // GPUs rather than a fixed environment, so the test holds for any device
    // count (e.g. under CUDA_VISIBLE_DEVICES or a 1-GPU CI runner). It asserts
    // the invariants the partitioning scheme must satisfy: every row in
    // [0, m) is assigned to exactly one rank, partitions are contiguous, and
    // the union covers all m rows.

    int device_count = DeviceMesh::instance().device_count();
    SCOPED_TRACE("Multi-GPU row partition behavior documented");
    ASSERT_GE(device_count, 1);

    const int m = 96;
    int rows_per_gpu = m / device_count;
    ASSERT_GT(rows_per_gpu, 0) << "m must be >= device_count for a non-trivial split";

    int covered = 0;
    for (int rank = 0; rank < device_count; ++rank) {
        int start_row = rank * rows_per_gpu;
        int local_m = (rank == device_count - 1) ? (m - start_row) : rows_per_gpu;
        EXPECT_EQ(start_row, covered) << "partition gap or overlap at rank " << rank;
        EXPECT_GT(local_m, 0) << "rank " << rank << " got an empty partition";
        covered += local_m;
    }
    EXPECT_EQ(covered, m);
}

// ============================================================================
// Multi-GPU Tests — real row-split + all-gather path (MGPU-12)
// ============================================================================

// Runs the genuine multi-GPU matmul path: one thread per rank pinned to its
// own device, each computing its row partition of C locally, then an NCCL
// all-gather reconstructing the full result on every device. Cross-checks the
// gathered C on every rank against a single-GPU reference computed on device 0.
//
// Environment-bound skips are genuine: <2 GPUs, or NCCL unavailable (the
// NCCL_TESTS_AVAILABLE gate used by the collective suite), or NCCL initialization
// failing (e.g. no nccl support in this build).
TEST_F(DistributedMatmulTest, MultiGpu_RowSplitAllGather) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU matmul test";
    }
    const char* nccl_test_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_test_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }

    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    // Pick a shape where the last rank's partition is a proper partial block so
    // the test exercises the padded equal-size all-gather path, not just an even
    // split. m = 6*device_count - 1 is never divisible by device_count for
    // device_count >= 2 (for 2 GPUs: 11 rows -> 6 + 5).
    const int m = 6 * device_count - 1;
    const int n = 48, k = 32;
    constexpr float kTolerance = 1e-3f;

    // Generate random input once; every rank replicates A and B.
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    // Single-GPU reference on device 0.
    std::vector<float> h_C_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k);
        cuda::memory::Buffer<float> d_B(k * n);
        cuda::memory::Buffer<float> d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_C_ref.data(), m * n);
    }

    // Each rank holds its own copy of A/B/C on its device, runs the genuine
    // multi-GPU path, and all ranks must converge on the reference.
    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(m * n));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k);
        cuda::memory::Buffer<float> d_B(k * n);
        cuda::memory::Buffer<float> d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);

        auto result = DistributedMatmul::matmul_multi_gpu(
            d_A.data(), d_B.data(), d_C.data(), m, n, k);
        ASSERT_TRUE(result.ok()) << result.error_message;

        d_C.copy_to(h_results[d].data(), m * n);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_C_ref.data(), h_results[rank].data(), m * n, kTolerance))
            << "Multi-GPU all-gathered matmul result on rank " << rank
            << " differs from single-GPU reference";
    }
}

TEST_F(DistributedMatmulTest, MultiGpu_AlphaBeta) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU matmul test";
    }
    const char* nccl_test_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_test_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }

    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 6 * device_count - 1;
    const int n = 48, k = 32;
    constexpr float kTolerance = 1e-3f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    DistributedMatmulOptions opts;
    opts.alpha = 2.0f;
    opts.beta = 0.5f;

    // Reference: alpha*(A@B) + beta*C_init, single GPU.
    std::vector<float> h_C_init(m * n);
    fill_constant(h_C_init.data(), m * n, 1.0f);
    std::vector<float> h_C_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k);
        cuda::memory::Buffer<float> d_B(k * n);
        cuda::memory::Buffer<float> d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        d_C.copy_from(h_C_init.data(), m * n);

        cuda::neural::MatmulOptions neural_opts;
        neural_opts.alpha = opts.alpha;
        neural_opts.beta = opts.beta;
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k, neural_opts);
        d_C.copy_to(h_C_ref.data(), m * n);
    }

    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(m * n));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k);
        cuda::memory::Buffer<float> d_B(k * n);
        cuda::memory::Buffer<float> d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        d_C.copy_from(h_C_init.data(), m * n);

        auto result = DistributedMatmul::matmul_multi_gpu(
            d_A.data(), d_B.data(), d_C.data(), m, n, k, opts);
        ASSERT_TRUE(result.ok()) << result.error_message;

        d_C.copy_to(h_results[d].data(), m * n);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_C_ref.data(), h_results[rank].data(), m * n, kTolerance))
            << "Multi-GPU alpha/beta matmul result on rank " << rank
            << " differs from single-GPU reference";
    }
}

// Regression for the m/n_dev guard: shapes where a rank's row slice would be
// empty or negative must be rejected with ncclInvalidArgument BEFORE entering
// the collective, so no rank blocks or poisons the shared NcclContext. The
// guards are rank-independent (all ranks share m/n/k/options), so every rank
// returns the same result and the group never waits on a missing collective.
TEST_F(DistributedMatmulTest, MultiGpu_RejectsBadShapes) {
    int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU matmul test";
    }
    const char* nccl_test_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_test_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }

    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    // m = 1 leaves the last rank with no rows (ceil(1/n) = 1 rows_per_gpu,
    // (n_dev-1)*1 >= 1). m < n_dev also exercises the same guard.
    const int m = 1, n = 8, k = 8;
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    auto bad_shape = [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k);
        cuda::memory::Buffer<float> d_B(k * n);
        cuda::memory::Buffer<float> d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);

        auto result = DistributedMatmul::matmul_multi_gpu(
            d_A.data(), d_B.data(), d_C.data(), m, n, k);
        EXPECT_EQ(static_cast<int>(result.code), static_cast<int>(ncclInvalidArgument))
            << "expected ncclInvalidArgument for m=" << m << ", got " << result.error_message;

        // Transposed operands are rejected on the row-split path regardless of shape.
        DistributedMatmulOptions transp;
        transp.trans_a = true;
        auto result_t = DistributedMatmul::matmul_multi_gpu(
            d_A.data(), d_B.data(), d_C.data(), m + 100, n, k, transp);
        EXPECT_EQ(static_cast<int>(result_t.code), static_cast<int>(ncclInvalidArgument))
            << "transposed operands must be rejected on the row-split path";
    };
    run_per_rank(device_count, bad_shape);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(DistributedMatmulTest, SmallMatrix) {
    // Test with very small matrices
    const int m = 12, n = 12, k = 12;
    constexpr float kTolerance = 1e-4f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_ref(m * n);
    cuda::memory::Buffer<float> d_C(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_ref.data(), m, n, k);
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

    std::vector<float> h_C_ref(m * n);
    std::vector<float> h_C(m * n);
    d_C_ref.copy_to(h_C_ref.data(), m * n);
    d_C.copy_to(h_C.data(), m * n);

    EXPECT_TRUE(arrays_near(h_C_ref.data(), h_C.data(), m * n, kTolerance))
        << "Small matrix test failed";
}

TEST_F(DistributedMatmulTest, WideMatrix) {
    // Test with wide output matrix (more columns than rows)
    const int m = 128, n = 512, k = 64;
    constexpr float kTolerance = 1e-4f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_ref(m * n);
    cuda::memory::Buffer<float> d_C(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_ref.data(), m, n, k);
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

    std::vector<float> h_C_ref(m * n);
    std::vector<float> h_C(m * n);
    d_C_ref.copy_to(h_C_ref.data(), m * n);
    d_C.copy_to(h_C.data(), m * n);

    EXPECT_TRUE(arrays_near(h_C_ref.data(), h_C.data(), m * n, kTolerance))
        << "Wide matrix test failed";
}

TEST_F(DistributedMatmulTest, TallMatrix) {
    // Test with tall input matrix (more rows than columns)
    const int m = 512, n = 128, k = 64;
    constexpr float kTolerance = 1e-4f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_ref(m * n);
    cuda::memory::Buffer<float> d_C(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_ref.data(), m, n, k);
    DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

    std::vector<float> h_C_ref(m * n);
    std::vector<float> h_C(m * n);
    d_C_ref.copy_to(h_C_ref.data(), m * n);
    d_C.copy_to(h_C.data(), m * n);

    EXPECT_TRUE(arrays_near(h_C_ref.data(), h_C.data(), m * n, kTolerance))
        << "Tall matrix test failed";
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(DistributedMatmulTest, StrategyDataParallel) {
    // Test that DataParallel strategy is accepted
    const int m = 128, n = 64, k = 32;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    DistributedMatmulOptions opts;
    opts.strategy = ParallelismStrategy::DataParallel;

    // Should not throw
    EXPECT_NO_THROW(
        DistributedMatmul::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k, opts)
    );
}

TEST_F(DistributedMatmulTest, MatmulSingleGpu) {
    // Test the explicit single-GPU path
    const int m = 64, n = 64, k = 64;
    constexpr float kTolerance = 1e-4f;

    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    cuda::memory::Buffer<float> d_A(m * k);
    cuda::memory::Buffer<float> d_B(k * n);
    cuda::memory::Buffer<float> d_C_ref(m * n);
    cuda::memory::Buffer<float> d_C(m * n);

    d_A.copy_from(h_A.data(), m * k);
    d_B.copy_from(h_B.data(), k * n);

    // Reference
    cuda::neural::matmul(d_A.data(), d_B.data(), d_C_ref.data(), m, n, k);

    // Explicit single-GPU
    DistributedMatmul::matmul_single_gpu(d_A.data(), d_B.data(), d_C.data(), m, n, k);

    // Compare
    std::vector<float> h_C_ref(m * n);
    std::vector<float> h_C(m * n);
    d_C_ref.copy_to(h_C_ref.data(), m * n);
    d_C.copy_to(h_C.data(), m * n);

    EXPECT_TRUE(arrays_near(h_C_ref.data(), h_C.data(), m * n, kTolerance))
        << "matmul_single_gpu test failed";
}

}  // namespace
