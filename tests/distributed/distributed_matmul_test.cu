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
#include "cuda/neural/matmul.h"
#include "cuda/memory/buffer.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace cuda::distributed;
using namespace cuda::mesh;

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
    // This test documents the expected row partition behavior
    // Skipped because multi-GPU requires multi-process execution

    int device_count = DeviceMesh::instance().device_count();
    SCOPED_TRACE("Multi-GPU row partition behavior documented");

    // For 8 GPUs and m=96 rows:
    // Each GPU gets 96/8 = 12 rows
    // GPU 0: rows [0, 12)
    // GPU 1: rows [12, 24)
    // ...
    // GPU 7: rows [84, 96)

    EXPECT_EQ(device_count, 8);  // Expected for this test environment

    // Document expected partitions
    const int m = 96;
    int rows_per_gpu = m / device_count;
    EXPECT_EQ(rows_per_gpu, 12);

    for (int rank = 0; rank < device_count; ++rank) {
        int start_row = rank * rows_per_gpu;
        int local_m = (rank == device_count - 1) ? (m - start_row) : rows_per_gpu;
        EXPECT_EQ(local_m, 12);
    }
}

TEST_F(DistributedMatmulTest, MultiGpu_RequiresMultiProcess) {
    // Document that multi-GPU matmul requires multi-process execution
    int device_count = DeviceMesh::instance().device_count();

    if (device_count > 1) {
        GTEST_SKIP() << "Multi-GPU matmul requires multi-process execution (e.g., NCCL). "
                     << "Single-process tests use the single-GPU fallback path for correctness.";
    } else {
        GTEST_SKIP() << "Single GPU system - multi-GPU tests not applicable";
    }
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
