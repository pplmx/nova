/**
 * @file tensor_parallel_multigpu_test.cpp
 * @brief Real thread-per-rank multi-GPU execution tests for TensorParallelMatmul
 *
 * Milestone v2.19 P1: TensorParallelMatmul's multi-GPU execution paths
 * (matmul / matmul_async, ColumnParallel and RowParallel) have never been
 * driven from real per-rank threads. tests/neural/tensor_parallel_test.cpp
 * checks only buffer-size formulas and profiler math; the sequence-parallel
 * tests pass comm = nullptr and never touch a real NCCL collective.
 *
 * The R15-R18 pattern is that every never-run multi-GPU surface hides real
 * bugs (NCCL collectives, distributed ops, MeshBarrier, distributed matmul).
 * These tests follow the same thread-per-rank harness the verified suites use
 * (distributed_test_common.h): one thread per visible device pinned with
 * cudaSetDevice, all entering the operation together, and the result asserted
 * against a single-GPU reference.
 *
 * P1 (round 19) shipped bug-finding variants of these as DISABLED_MultiGpu_*
 * and proved them RED (HYP-002 / EV-004): the ColumnParallel partition math
 * (contiguous-slice B/C offsets) produced wrong output against a row-major
 * matmul, n % tp != 0 was silently dropped instead of rejected, and RowParallel
 * ignored the rank index. P2 converged the implementation onto the corrected
 * partition math (zero-padded strided block compute + a single AllReduce;
 * per-rank row blocks; explicit shape rejection) — these renamed MultiGpu_*
 * tests are the green acceptance gate.
 */

#include <gtest/gtest.h>

#include "cuda/neural/tensor_parallel_matmul.h"
#include "cuda/neural/matmul.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include "distributed_test_common.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

using namespace cuda::neural;
using namespace cuda::mesh;

using cuda::distributed::test::run_per_rank;
using cuda::distributed::test::RankBarrierTimeout;

namespace {

// Multi-GPU readiness for the shared NCCL singleton, self-healing a broken
// context (milestone v2.18 P3): after a genuine timeout/async abort the
// singleton is broken and would otherwise turn every later test into a silent
// skip. Same shape as the distributed suites' gate.
bool multigpu_nccl_ready() {
    auto& ctx = cuda::nccl::NcclContext::instance();
    try {
        ctx.initialize();
    } catch (const std::exception&) {
        return false;
    }
    return cuda::distributed::test::heal_and_ready(ctx);
}

void fill_random(float* data, size_t count) {
    static std::mt19937 rng(7);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

// Single matmul on host-visible B; small matrices so 1e-2 is a loose bound for
// fp32 accumulate order differences across the slice-then-AllReduce path.
constexpr float kTolerance = 1e-2f;

}  // namespace

class TensorParallelMultiGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }

    static void SetUpTestSuite() {
        cudaFree(nullptr);  // Ensure CUDA context for device discovery
    }

    static testing::AssertionResult arrays_near(
        const float* expected, const float* actual,
        size_t count, float tolerance) {
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(expected[i] - actual[i]) > tolerance) {
                return testing::AssertionFailure()
                       << "mismatch at " << i << ": expected " << expected[i]
                       << " actual " << actual[i] << " (|diff| "
                       << std::abs(expected[i] - actual[i]) << " > " << tolerance << ")";
            }
        }
        return testing::AssertionSuccess();
    }
};

// ============================================================================
// P2 acceptance tests — TensorParallelMatmul on real multi-GPU
// ============================================================================

// Column-parallel forward: every rank drives the genuine multi-GPU path with
// the shared NcclContext singleton; the full result must match the single-GPU
// reference on every rank (a 1e-2 bound absorbs fp32 accumulate-order
// differences across the slice-then-AllReduce path).
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelColumnParallel) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP matmul test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 32, n = 8 * device_count, k = 16;  // n divisible by tp_degree
    std::vector<float> h_A(m * k), h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    // Single-GPU reference on device 0.
    std::vector<float> h_C_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_C_ref.data(), m * n);
    }

    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(m * n));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);

        TensorParallelMatmul tpm(ctx, TensorParallelStrategy::ColumnParallel);
        tpm.matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

        d_C.copy_to(h_results[d].data(), m * n);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_C_ref.data(), h_results[rank].data(), m * n, kTolerance))
            << "ColumnParallel TP matmul result on rank " << rank
            << " differs from the single-GPU reference";
    }
}

// Same as above but through matmul_async on a per-rank stream (the path
// collectives are ordered on).
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelColumnParallelAsync) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP matmul test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 32, n = 8 * device_count, k = 16;
    std::vector<float> h_A(m * k), h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    std::vector<float> h_C_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_C_ref.data(), m * n);
    }

    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(m * n));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        {
            TensorParallelMatmul tpm(ctx, TensorParallelStrategy::ColumnParallel);
            tpm.matmul_async(d_A.data(), d_B.data(), d_C.data(), m, n, k, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        d_C.copy_to(h_results[d].data(), m * n);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_C_ref.data(), h_results[rank].data(), m * n, kTolerance))
            << "ColumnParallel async TP matmul result on rank " << rank
            << " differs from the single-GPU reference";
    }
}

// Column-parallel with n NOT divisible by tp_degree. A correct contraction
// contract must reject the shape explicitly (the library's fail-fast
// convention — cf. MeshBarrier's subset-throws contract) rather than silently
// dropping the remainder columns. This test pins that contract.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelColumnParallelNonDivisible) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP matmul test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 32, n = 4 * device_count + 1, k = 16;  // n % tp_degree == 1
    ASSERT_NE(n / device_count * device_count, n) << "n must not be divisible";

    std::vector<float> h_A(m * k), h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    std::vector<float> h_C_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_C_ref.data(), m * n);
    }

    // Every rank must reject the non-divisible shape (fail fast), not produce a
    // partial result. run_per_rank rethrows the first per-rank exception, so an
    // EXPECT_THROW around it is satisfied when any rank raises.
    EXPECT_THROW(
        run_per_rank(device_count, [&](int d) {
            cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
            d_A.copy_from(h_A.data(), m * k);
            d_B.copy_from(h_B.data(), k * n);

            TensorParallelMatmul tpm(ctx, TensorParallelStrategy::ColumnParallel);
            tpm.matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        }),
        std::exception)
        << "ColumnParallel TP matmul with n % tp_degree != 0 must reject the "
           "shape explicitly instead of silently dropping columns";
}

// RowParallel: each rank computes only its contiguous row block of C (no
// communication). The union of the per-rank blocks must equal the reference.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelRowParallel) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP matmul test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 8 * device_count, n = 48, k = 16;  // m divisible by tp_degree
    std::vector<float> h_A(m * k), h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    std::vector<float> h_C_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_C_ref.data(), m * n);
    }

    const int local_m = m / device_count;
    std::vector<std::vector<float>> h_results(device_count, std::vector<float>(local_m * n));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(local_m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_B.data(), k * n);

        TensorParallelMatmul tpm(ctx, TensorParallelStrategy::RowParallel);
        tpm.matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);

        // Contract: rank r returns ITS contiguous row block in C[0, local_m*n).
        // (The current implementation ignores the rank index entirely and writes
        // the same second block on every rank — ——.)
        d_C.copy_to(h_results[d].data(), local_m * n);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_C_ref.data() + rank * local_m * n,
                                h_results[rank].data(), local_m * n, kTolerance))
            << "RowParallel TP matmul row block on rank " << rank
            << " differs from the single-GPU reference";
    }
}
