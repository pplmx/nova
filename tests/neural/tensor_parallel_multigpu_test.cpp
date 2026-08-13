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
#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/matmul.h"
#include "cuda/neural/activations.h"
#include "cuda/neural/loss/loss_functions.h"
#include "cuda/neural/optimizers/optimizers.h"
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
using namespace cuda::neural::optimizers;
using namespace cuda::neural::loss;
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

// Host double-precision matmul: C[m x n] = A[m x k] @ B[k x n] (row-major).
void host_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) {
                acc += static_cast<double>(A[i * k + t]) *
                       static_cast<double>(B[t * n + j]);
            }
            C[i * n + j] = static_cast<float>(acc);
        }
    }
}

// Analytic backward of Y = X W in double precision: dW = X^T dY, dX = dY W^T.
void ref_linear_backward(const float* X, const float* W, const float* dY,
                         float* dX, float* dW, int m, int k, int n) {
    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < n; ++b) {
            double acc = 0.0;
            for (int i = 0; i < m; ++i) {
                acc += static_cast<double>(X[i * k + a]) * dY[i * n + b];
            }
            dW[a * n + b] = static_cast<float>(acc);
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int a = 0; a < k; ++a) {
            double acc = 0.0;
            for (int b = 0; b < n; ++b) {
                acc += static_cast<double>(dY[i * n + b]) * W[a * n + b];
            }
            dX[i * k + a] = static_cast<float>(acc);
        }
    }
}

// Single-GPU SiLU-gated MLP reference through the public matmul/activations
// primitives: out = (silu(A @ W_gate) .* (A @ W_up)) @ W_down. The multi-GPU
// TensorParallelMLP is compared against this (full-weight) computation.
void ref_gated_mlp(const float* A, const float* W_gate, const float* W_up,
                   const float* W_down, float* out, int m, int h, int inter) {
    cuda::memory::Buffer<float> d_A(m * h);
    cuda::memory::Buffer<float> d_gate(m * inter), d_up(m * inter),
        d_sub(m * inter), d_out(m * h);
    cuda::memory::Buffer<float> d_W_gate(h * inter), d_W_up(h * inter),
        d_W_down(h * inter);
    d_A.copy_from(A, static_cast<size_t>(m) * h);
    d_W_gate.copy_from(W_gate, static_cast<size_t>(h) * inter);
    d_W_up.copy_from(W_up, static_cast<size_t>(h) * inter);
    d_W_down.copy_from(W_down, static_cast<size_t>(h) * inter);

    cuda::neural::matmul(d_A.data(), d_W_gate.data(), d_gate.data(),
                         m, inter, h);
    cuda::neural::matmul(d_A.data(), d_W_up.data(), d_up.data(), m, inter, h);
    cuda::neural::silu_and_mul(d_gate.data(), d_up.data(), d_sub.data(),
                               m * inter);
    cuda::neural::matmul(d_sub.data(), d_W_down.data(), d_out.data(),
                         m, h, inter);
    d_out.copy_to(out, static_cast<size_t>(m) * h);
}

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

        // Contract: rank r returns ITS contiguous row block in C[0, local_m*n)
        // (the union of the per-rank buffers reconstructs the full result).
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

// ============================================================================
// v2.21 weight-managed layer acceptance — TensorParallelLayers on real multi-GPU
// (TASK-010 / issue-v20-tp-layers-stubs, replacing the DEC-006 fail-fast stubs)
// ============================================================================

// ColumnParallelLayer shards its weight along the output dimension: each rank
// owns columns [r*n_local, +n_local) and computes A @ W_rank locally (no comm).
// The column-wise concatenation of the per-rank sharded outputs must equal the
// single-GPU full-weight reference. This pins the shard slice math that v2.19's
// RED tests proved the pre-v2.20 code got wrong.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_ColumnParallelLayer_ShardConcatMatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 16, k = 16;
    const int n = 8 * device_count;  // divisible by tp_degree
    const int n_local = n / device_count;
    std::vector<float> h_A(m * k), h_W(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_W.data(), k * n);

    std::vector<float> h_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_W.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_ref.data(), m * n);
    }

    std::vector<std::vector<float>> shards(
        device_count, std::vector<float>(m * n_local));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * k), d_out(m * n_local);
        d_A.copy_from(h_A.data(), m * k);

        ColumnParallelLayer layer(ctx, k, n);
        layer.set_weight(h_W.data());  // slices this rank's column shard
        layer.forward(d_A.data(), d_out.data(), m, 1);
        d_out.copy_to(shards[d].data(), m * n_local);
    });

    // Assemble the sharded outputs back into the full width: out_r occupies
    // columns [r*n_local, +n_local) of the full result.
    std::vector<float> assembled(m * n);
    for (int i = 0; i < m; ++i) {
        for (int r = 0; r < device_count; ++r) {
            for (int j = 0; j < n_local; ++j) {
                assembled[i * n + r * n_local + j] =
                    shards[r][i * n_local + j];
            }
        }
    }
    EXPECT_TRUE(arrays_near(h_ref.data(), assembled.data(), m * n, kTolerance))
        << "concatenated ColumnParallelLayer shards differ from the "
           "single-GPU full-weight reference";
}

// ColumnParallelLayer with out_features NOT divisible by tp_degree: the layer
// cannot own whole column shards, so construction must fail fast (library
// fail-fast convention), not slice a ragged remainder.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_ColumnParallelLayer_RejectsNonDivisibleOut) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int n = 8 * device_count + 1;  // out_features % tp_degree == 1
    EXPECT_THROW(
        run_per_rank(device_count, [&](int) {
            ColumnParallelLayer layer(ctx, /*in_features=*/16, n);
        }),
        std::exception)
        << "ColumnParallelLayer with out_features % tp_degree != 0 must "
           "reject the shape at construction";
}

// RowParallelLayer shards its weight along the input dimension and AllReduces
// the per-rank partial to a replicated full-width output. Every rank's output
// must equal the single-GPU full-weight reference — the real comm path.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_RowParallelLayer_ReplicatedMatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 8, n = 48;
    const int k = 16 * device_count;  // in_features divisible by tp_degree
    const int k_local = k / device_count;
    std::vector<float> h_A(m * k), h_W(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_W.data(), k * n);

    std::vector<float> h_ref(m * n);
    {
        cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
        d_A.copy_from(h_A.data(), m * k);
        d_B.copy_from(h_W.data(), k * n);
        cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        d_C.copy_to(h_ref.data(), m * n);
    }

    std::vector<std::vector<float>> results(
        device_count, std::vector<float>(m * n));
    run_per_rank(device_count, [&](int d) {
        // Each rank consumes its segment of the input (columns of X split by
        // the weight's input dimension) and its row shard of W.
        std::vector<float> x_r(static_cast<size_t>(m) * k_local);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k_local; ++j) {
                x_r[i * k_local + j] = h_A[i * k + d * k_local + j];
            }
        }
        cuda::memory::Buffer<float> d_x(m * k_local), d_out(m * n);
        d_x.copy_from(x_r.data(), m * k_local);

        RowParallelLayer layer(ctx, k, n);
        layer.set_weight(h_W.data());  // slices this rank's row shard
        layer.forward(d_x.data(), d_out.data(), m, 1);
        d_out.copy_to(results[d].data(), m * n);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_ref.data(), results[rank].data(), m * n,
                                kTolerance))
            << "replicated RowParallelLayer output on rank " << rank
            << " differs from the single-GPU full-weight reference";
    }
}

// RowParallelLayer with in_features NOT divisible by tp_degree: reject at
// construction.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_RowParallelLayer_RejectsNonDivisibleIn) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int k = 16 * device_count + 1;  // in_features % tp_degree == 1
    EXPECT_THROW(
        run_per_rank(device_count, [&](int) {
            RowParallelLayer layer(ctx, k, /*out_features=*/48);
        }),
        std::exception)
        << "RowParallelLayer with in_features % tp_degree != 0 must reject "
           "the shape at construction";
}

// The full SiLU-gated TensorParallelMLP on real multi-GPU: gate/up are
// column-parallel (sharded intermediate), silu_and_mul, then the row-parallel
// down projection AllReduces to the replicated hidden-width output. Every
// rank's output must equal the single-GPU full-weight gated MLP reference.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelMLP_GatedMatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 8, h = 16;
    const int inter = 8 * device_count;  // intermediate divisible by tp_degree
    std::vector<float> h_A(m * h);
    std::vector<float> h_W_gate(h * inter), h_W_up(h * inter);
    std::vector<float> h_W_down(inter * h);
    fill_random(h_A.data(), m * h);
    fill_random(h_W_gate.data(), h_W_gate.size());
    fill_random(h_W_up.data(), h_W_up.size());
    fill_random(h_W_down.data(), h_W_down.size());

    std::vector<float> h_ref(m * h);
    ref_gated_mlp(h_A.data(), h_W_gate.data(), h_W_up.data(), h_W_down.data(),
                  h_ref.data(), m, h, inter);

    std::vector<std::vector<float>> results(
        device_count, std::vector<float>(m * h));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_A(m * h), d_out(m * h);
        d_A.copy_from(h_A.data(), m * h);

        TensorParallelMLP mlp(ctx, h, inter);
        mlp.set_weight(h_W_gate.data(), h_W_up.data(), h_W_down.data());
        mlp.forward(d_A.data(), d_out.data(), m, 1);
        d_out.copy_to(results[d].data(), m * h);
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_ref.data(), results[rank].data(), m * h,
                                kTolerance))
            << "replicated TensorParallelMLP output on rank " << rank
            << " differs from the single-GPU full-weight gated MLP reference";
    }
}

// RowParallel with m NOT divisible by tp_degree: must reject the shape
// explicitly (same fail-fast contract as the column path), not silently drop
// rows. Complement to the ColumnParallel non-divisible test (MEDIUM-6).
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelRowParallelNonDivisible) {
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

    const int m = 8 * device_count + 1, n = 48, k = 16;  // m % tp_degree == 1
    ASSERT_NE(m / device_count * device_count, m) << "m must not be divisible";

    std::vector<float> h_A(m * k), h_B(k * n);
    fill_random(h_A.data(), m * k);
    fill_random(h_B.data(), k * n);

    EXPECT_THROW(
        run_per_rank(device_count, [&](int d) {
            cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m);
            d_A.copy_from(h_A.data(), m * k);
            d_B.copy_from(h_B.data(), k * n);

            TensorParallelMatmul tpm(ctx, TensorParallelStrategy::RowParallel);
            tpm.matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
        }),
        std::exception)
        << "RowParallel TP matmul with m % tp_degree != 0 must reject the "
           "shape explicitly instead of silently dropping rows";
}

// ============================================================================
// v2.23 backward acceptance — milestone v2.23 P1/TASK-019 (RED against the
// provisional throw-stub; GREEN with the P2 implementation). The row layer's
// grad-input is local (matches the sharded-input layout); the column layer's
// grad-input is the AllReduce of the per-rank dY W^T.
// ============================================================================

// ColumnParallelLayer backward on real multi-GPU: each rank writes its grad
// weight column slice and contributes dY_r W_r^T to the grad-input; the
// AllReduce must make every rank's grad_input equal the full reference
// dX = dY_full W^T, and the assembled grad-weight slices must equal X^T dY_full.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_ColumnParallelLayer_Backward_MatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 16, k = 16;
    const int n = 8 * device_count;  // divisible by tp_degree
    const int n_local = n / device_count;
    std::vector<float> h_X(m * k), h_W(k * n), h_dY(m * n);
    fill_random(h_X.data(), m * k);
    fill_random(h_W.data(), k * n);
    fill_random(h_dY.data(), m * n);

    std::vector<float> ref_dX(m * k), ref_dW(k * n);
    ref_linear_backward(h_X.data(), h_W.data(), h_dY.data(),
                        ref_dX.data(), ref_dW.data(), m, k, n);

    std::vector<std::vector<float>> dX_rank(device_count,
                                            std::vector<float>(m * k));
    std::vector<std::vector<float>> dW_slices(device_count,
                                              std::vector<float>(k * n_local));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_X(m * k), d_dY(m * n_local),
            d_dX(m * k), d_dW(k * n);
        d_X.copy_from(h_X.data(), m * k);

        ColumnParallelLayer layer(ctx, k, n);
        layer.set_weight(h_W.data());
        // Build the sharded grad_output on host then copy (columns of dY).
        std::vector<float> dY_r(static_cast<size_t>(m) * n_local);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n_local; ++j) {
                dY_r[static_cast<size_t>(i) * n_local + j] =
                    h_dY[i * n + d * n_local + j];
            }
        }
        d_dY.copy_from(dY_r.data(), dY_r.size());

        layer.backward(d_X.data(), d_dY.data(), d_dX.data(), d_dW.data(), m, 1);
        d_dX.copy_to(dX_rank[d].data(), m * k);
        // Copy the device grad-weight buffer out and slice this rank's block.
        std::vector<float> full_dW(k * n);
        d_dW.copy_to(full_dW.data(), k * n);
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < n_local; ++b) {
                dW_slices[d][static_cast<size_t>(a) * n_local + b] =
                    full_dW[static_cast<size_t>(a) * n + d * n_local + b];
            }
        }
    });

    // Every rank's grad-input must be the full AllReduced reference.
    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(ref_dX.data(), dX_rank[rank].data(), m * k,
                                kTolerance))
            << "column-layer AllReduced grad-input on rank " << rank
            << " differs from the full reference dX (the grad-input AllReduce "
               "collective is broken or not implemented)";
    }
    // Assemble the per-rank grad-weight column slices.
    std::vector<float> assembled_dW(k * n, 0.0f);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < n_local; ++b) {
                assembled_dW[static_cast<size_t>(a) * n + r * n_local + b] =
                    dW_slices[r][static_cast<size_t>(a) * n_local + b];
            }
        }
    }
    EXPECT_TRUE(arrays_near(ref_dW.data(), assembled_dW.data(), k * n, kTolerance))
        << "assembled column-layer grad-weight slices differ from X^T dY_full";
}

// RowParallelLayer backward on real multi-GPU: grad-input is local — each
// rank's dX_r must equal its block of the full reference (dY W^T), and the
// assembled grad-weight row slices must equal X_full^T dY.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_RowParallelLayer_Backward_MatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 8, n = 48;
    const int k = 16 * device_count;  // in_features divisible by tp_degree
    const int k_local = k / device_count;
    std::vector<float> h_X(m * k), h_W(k * n), h_dY(m * n);
    fill_random(h_X.data(), m * k);
    fill_random(h_W.data(), k * n);
    fill_random(h_dY.data(), m * n);

    std::vector<float> ref_dX(m * k), ref_dW(k * n);
    ref_linear_backward(h_X.data(), h_W.data(), h_dY.data(),
                        ref_dX.data(), ref_dW.data(), m, k, n);

    std::vector<std::vector<float>> dX_rank(device_count,
                                            std::vector<float>(m * k_local));
    std::vector<std::vector<float>> dW_slices(device_count,
                                              std::vector<float>(k_local * n));
    run_per_rank(device_count, [&](int d) {
        std::vector<float> x_r(static_cast<size_t>(m) * k_local);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k_local; ++j) {
                x_r[static_cast<size_t>(i) * k_local + j] =
                    h_X[i * k + d * k_local + j];
            }
        }
        cuda::memory::Buffer<float> d_X(m * k_local), d_dY(m * n),
            d_dX(m * k_local), d_dW(k * n);
        d_X.copy_from(x_r.data(), x_r.size());
        d_dY.copy_from(h_dY.data(), m * n);

        RowParallelLayer layer(ctx, k, n);
        layer.set_weight(h_W.data());
        layer.backward(d_X.data(), d_dY.data(), d_dX.data(), d_dW.data(), m, 1);

        d_dX.copy_to(dX_rank[d].data(), m * k_local);
        std::vector<float> full_dW(k * n);
        d_dW.copy_to(full_dW.data(), k * n);
        for (int a = 0; a < k_local; ++a) {
            for (int b = 0; b < n; ++b) {
                dW_slices[d][static_cast<size_t>(a) * n + b] =
                    full_dW[static_cast<size_t>(d * k_local + a) * n + b];
            }
        }
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        // rank r's grad-input is the r-th FEATURE block of the full reference
        // grad-input (a [m x k] row-major matrix): columns [r*k_local, +k_local)
        // stride by the full row length k, so extract them contiguously rather
        // than viewing a bogus contiguous pointer.
        std::vector<float> expect_block(static_cast<size_t>(m) * k_local);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k_local; ++j) {
                expect_block[static_cast<size_t>(i) * k_local + j] =
                    ref_dX[i * k + rank * k_local + j];
            }
        }
        EXPECT_TRUE(arrays_near(expect_block.data(), dX_rank[rank].data(),
                                m * k_local, kTolerance))
            << "row-layer local grad-input block on rank " << rank
            << " differs from the reference";
    }
    std::vector<float> assembled_dW(k * n, 0.0f);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < k_local; ++a) {
            for (int b = 0; b < n; ++b) {
                assembled_dW[static_cast<size_t>(r * k_local + a) * n + b] =
                    dW_slices[r][static_cast<size_t>(a) * n + b];
            }
        }
    }
    EXPECT_TRUE(arrays_near(ref_dW.data(), assembled_dW.data(), k * n, kTolerance))
        << "assembled row-layer grad-weight slices differ from X^T dY";
}

// The full SiLU-gated TensorParallelMLP backward on real multi-GPU: grad-input
// is the AllReduce of the gate+up per-rank contributions; grad weights assemble
// to the full analytic gradients. Money test for the whole backward chain.
TEST_F(TensorParallelMultiGpuTest, MultiGpu_TensorParallelMLP_Backward_MatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU TP layer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU TP layers require NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 8, h = 16;
    const int inter = 8 * device_count;  // intermediate divisible by tp
    const int inter_local = inter / device_count;
    std::vector<float> h_X(m * h);
    std::vector<float> h_Wg(h * inter), h_Wu(h * inter), h_Wd(inter * h);
    std::vector<float> h_dY(m * h);
    fill_random(h_X.data(), m * h);
    fill_random(h_Wg.data(), h_Wg.size());
    fill_random(h_Wu.data(), h_Wu.size());
    fill_random(h_Wd.data(), h_Wd.size());
    fill_random(h_dY.data(), m * h);

    // Host reference: forward activations + analytic backward (full weights).
    auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    auto silu_prime = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 + x * (1.0 - s));
    };
    std::vector<float> G(static_cast<size_t>(m) * inter);
    std::vector<float> U(static_cast<size_t>(m) * inter);
    std::vector<float> sub(static_cast<size_t>(m) * inter);
    host_matmul(h_X.data(), h_Wg.data(), G.data(), m, inter, h);
    host_matmul(h_X.data(), h_Wu.data(), U.data(), m, inter, h);
    for (size_t i = 0; i < sub.size(); ++i) {
        sub[i] = static_cast<float>(silu(G[i]) * U[i]);
    }

    std::vector<float> dsub(static_cast<size_t>(m) * inter);
    std::vector<float> ref_dWd(static_cast<size_t>(inter) * h);
    ref_linear_backward(sub.data(), h_Wd.data(), h_dY.data(),
                        dsub.data(), ref_dWd.data(), m, inter, h);
    std::vector<float> dgate(static_cast<size_t>(m) * inter);
    std::vector<float> dup(static_cast<size_t>(m) * inter);
    for (int i = 0; i < m * inter; ++i) {
        dgate[i] = static_cast<float>(dsub[i] * U[i] * silu_prime(G[i]));
        dup[i] = static_cast<float>(dsub[i] * silu(G[i]));
    }
    std::vector<float> dXg(static_cast<size_t>(m) * h);
    std::vector<float> dXu(static_cast<size_t>(m) * h);
    std::vector<float> ref_dX(static_cast<size_t>(m) * h);
    std::vector<float> ref_dWg(static_cast<size_t>(h) * inter);
    std::vector<float> ref_dWu(static_cast<size_t>(h) * inter);
    ref_linear_backward(h_X.data(), h_Wg.data(), dgate.data(),
                        dXg.data(), ref_dWg.data(), m, h, inter);
    ref_linear_backward(h_X.data(), h_Wu.data(), dup.data(),
                        dXu.data(), ref_dWu.data(), m, h, inter);
    for (int i = 0; i < m * h; ++i) {
        ref_dX[i] = dXg[i] + dXu[i];
    }

    std::vector<std::vector<float>> dX_rank(device_count,
                                            std::vector<float>(m * h));
    std::vector<std::vector<float>> dWg_slices(device_count,
                                               std::vector<float>(h * inter_local));
    std::vector<std::vector<float>> dWu_slices(device_count,
                                               std::vector<float>(h * inter_local));
    std::vector<std::vector<float>> dWd_slices(device_count,
                                               std::vector<float>(inter_local * h));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_X(m * h), d_dY(m * h), d_out(m * h),
            d_dX(m * h);
        cuda::memory::Buffer<float> d_dWg(h * inter), d_dWu(h * inter),
            d_dWd(inter * h);
        d_X.copy_from(h_X.data(), m * h);
        d_dY.copy_from(h_dY.data(), m * h);

        TensorParallelMLP mlp(ctx, h, inter);
        mlp.set_weight(h_Wg.data(), h_Wu.data(), h_Wd.data());
        mlp.forward(d_X.data(), d_out.data(), m, 1);  // save activations
        mlp.backward(d_X.data(), d_dY.data(), d_dX.data(),
                     d_dWg.data(), d_dWu.data(), d_dWd.data(), m, 1);

        d_dX.copy_to(dX_rank[d].data(), m * h);
        std::vector<float> full_dWg(h * inter), full_dWu(h * inter),
            full_dWd(inter * h);
        d_dWg.copy_to(full_dWg.data(), h * inter);
        d_dWu.copy_to(full_dWu.data(), h * inter);
        d_dWd.copy_to(full_dWd.data(), inter * h);
        // Column-shard slices for gate/up, row-shard slices for down.
        for (int a = 0; a < h; ++a) {
            for (int b = 0; b < inter_local; ++b) {
                dWg_slices[d][static_cast<size_t>(a) * inter_local + b] =
                    full_dWg[static_cast<size_t>(a) * inter + d * inter_local + b];
                dWu_slices[d][static_cast<size_t>(a) * inter_local + b] =
                    full_dWu[static_cast<size_t>(a) * inter + d * inter_local + b];
            }
        }
        for (int a = 0; a < inter_local; ++a) {
            for (int b = 0; b < h; ++b) {
                dWd_slices[d][static_cast<size_t>(a) * h + b] =
                    full_dWd[static_cast<size_t>(d * inter_local + a) * h + b];
            }
        }
    });

    for (int rank = 0; rank < device_count; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(ref_dX.data(), dX_rank[rank].data(), m * h,
                                kTolerance))
            << "MLP AllReduced grad-input on rank " << rank
            << " differs from the full reference (the grad-input AllReduce is "
               "broken in the chained backward)";
    }
    std::vector<float> assembled_dWg(h * inter, 0.0f);
    std::vector<float> assembled_dWu(h * inter, 0.0f);
    std::vector<float> assembled_dWd(inter * h, 0.0f);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < h; ++a) {
            for (int b = 0; b < inter_local; ++b) {
                assembled_dWg[static_cast<size_t>(a) * inter + r * inter_local + b] =
                    dWg_slices[r][static_cast<size_t>(a) * inter_local + b];
                assembled_dWu[static_cast<size_t>(a) * inter + r * inter_local + b] =
                    dWu_slices[r][static_cast<size_t>(a) * inter_local + b];
            }
        }
        for (int a = 0; a < inter_local; ++a) {
            for (int b = 0; b < h; ++b) {
                assembled_dWd[static_cast<size_t>(r * inter_local + a) * h + b] =
                    dWd_slices[r][static_cast<size_t>(a) * h + b];
            }
        }
    }
    EXPECT_TRUE(arrays_near(ref_dWg.data(), assembled_dWg.data(), h * inter,
                            kTolerance))
        << "assembled MLP gate grad-weight slices differ from the reference";
    EXPECT_TRUE(arrays_near(ref_dWu.data(), assembled_dWu.data(), h * inter,
                            kTolerance))
        << "assembled MLP up grad-weight slices differ from the reference";
    EXPECT_TRUE(arrays_near(ref_dWd.data(), assembled_dWd.data(), inter * h,
                            kTolerance))
        << "assembled MLP down grad-weight slices differ from the reference";
}

// ============================================================================
// v2.24 end-to-end training-step acceptance (TASK-024, milestone P1). The
// tensor-parallel stack has forward+backward but no way to train it: the loss
// layer is host-side with no grad (its device kernels are dead code) and no
// layer method steps the private weight shards. This test runs K identical
// AdamW training steps on the sharded TensorParallelMLP on real multi-GPU —
// forward -> cross_entropy_logits_backward (seed) -> MLP backward (full grad
// buffers) -> step() (AdamW on each rank shard) — then assembles the per-rank
// weight shards and compares against a host double-precision full-weight
// reference trained over the same steps. RED against the throw-stubs.
// ============================================================================

namespace {

// Host double-precision AdamW reference (identical formula to the library).
struct HostAdamW {
    std::vector<double> m, v;
    void step(const float* w, const float* dw, float* w_out, size_t n,
              int step, float lr, float beta1, float beta2, float eps,
              float wd) {
        if (m.size() != n) {
            m.assign(n, 0.0);
            v.assign(n, 0.0);
        }
        const double b1p = std::pow(static_cast<double>(beta1), step);
        const double b2p = std::pow(static_cast<double>(beta2), step);
        const double lr_t =
            static_cast<double>(lr) * std::sqrt(1.0 - b2p) / (1.0 - b1p);
        for (size_t i = 0; i < n; ++i) {
            const double g = dw[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            const double m_hat = m[i] / (1.0 - b1p);
            const double v_hat = v[i] / (1.0 - b2p);
            double update = m_hat / (std::sqrt(v_hat) + eps) + wd * w[i];
            w_out[i] = static_cast<float>(w[i] - lr_t * update);
        }
    }
};

// One host fp64 training step of the gated MLP on full weights:
// logits = down(silu(gate) .* up); dlogits = (softmax - onehot)/B; analytic
// backward for dWg/dWu/dWd; then AdamW steps each full weight.
void host_training_step(const float* X, const int* targets,
                        float* Wg, float* Wu, float* Wd,
                        HostAdamW& ag, HostAdamW& au, HostAdamW& ad,
                        int m, int h, int inter, int step_no,
                        float lr, float beta1, float beta2, float eps,
                        float wd) {
    auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    auto silu_prime = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 + x * (1.0 - s));
    };
    std::vector<float> G(m * inter), U(m * inter), sub(m * inter);
    std::vector<float> logits(m * h);
    host_matmul(X, Wg, G.data(), m, inter, h);
    host_matmul(X, Wu, U.data(), m, inter, h);
    for (size_t i = 0; i < sub.size(); ++i) {
        sub[i] = static_cast<float>(silu(G[i]) * U[i]);
    }
    host_matmul(sub.data(), Wd, logits.data(), m, h, inter);

    // dlogits = (softmax - onehot) / B (reduction_mean).
    std::vector<float> dlogits(m * h);
    for (int b = 0; b < m; ++b) {
        double mm = logits[b * h];
        for (int c = 1; c < h; ++c) {
            mm = std::max(mm, static_cast<double>(logits[b * h + c]));
        }
        double sum = 0.0;
        for (int c = 0; c < h; ++c) {
            sum += std::exp(static_cast<double>(logits[b * h + c]) - mm);
        }
        for (int c = 0; c < h; ++c) {
            double sm = std::exp(static_cast<double>(logits[b * h + c]) - mm) / sum;
            dlogits[b * h + c] = static_cast<float>((sm - (c == targets[b])) / m);
        }
    }

    // Analytic MLP backward to the full-weight grads.
    std::vector<float> dsub(m * inter), dg(m * inter), du(m * inter);
    std::vector<float> dWg(h * inter), dWu(h * inter), dWd(inter * h);
    std::vector<float> dXg(m * h), dXu(m * h);
    ref_linear_backward(sub.data(), Wd, dlogits.data(), dsub.data(),
                        dWd.data(), m, inter, h);
    for (int i = 0; i < m * inter; ++i) {
        dg[i] = static_cast<float>(dsub[i] * U[i] * silu_prime(G[i]));
        du[i] = static_cast<float>(dsub[i] * silu(G[i]));
    }
    std::vector<float> tmp_dx(m * h);  // unused grad-input
    ref_linear_backward(X, Wg, dg.data(), tmp_dx.data(), dWg.data(), m, h, inter);
    ref_linear_backward(X, Wu, du.data(), tmp_dx.data(), dWu.data(), m, h, inter);

    ag.step(Wg, dWg.data(), Wg, static_cast<size_t>(h) * inter, step_no,
            lr, beta1, beta2, eps, wd);
    au.step(Wu, dWu.data(), Wu, static_cast<size_t>(h) * inter, step_no,
            lr, beta1, beta2, eps, wd);
    ad.step(Wd, dWd.data(), Wd, static_cast<size_t>(inter) * h, step_no,
            lr, beta1, beta2, eps, wd);
}

}  // namespace

TEST_F(TensorParallelMultiGpuTest, MultiGpu_TrainingStep_MatchesHostFullWeight) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU training-step test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU training-step requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 8, h = 16, inter = 32;
    const int K = 3;  // a few identical AdamW steps
    std::vector<float> X(static_cast<size_t>(m) * h);
    std::vector<int> targets(m);
    std::vector<float> Wg0(static_cast<size_t>(h) * inter);
    std::vector<float> Wu0(static_cast<size_t>(h) * inter);
    std::vector<float> Wd0(static_cast<size_t>(inter) * h);
    fill_random(X.data(), X.size());
    fill_random(Wg0.data(), Wg0.size());
    fill_random(Wu0.data(), Wu0.size());
    fill_random(Wd0.data(), Wd0.size());
    std::mt19937 rng(20260814);
    for (int b = 0; b < m; ++b) {
        targets[b] = static_cast<int>(rng() % h);
    }

    // Host fp64 full-weight reference: K identical training steps.
    OptimizerConfig opt_cfg;
    opt_cfg.learning_rate = 0.01f;
    std::vector<float> rWg = Wg0, rWu = Wu0, rWd = Wd0;
    HostAdamW ri_rg, ri_ru, ri_rd;
    for (int s = 1; s <= K; ++s) {
        host_training_step(X.data(), targets.data(),
                           rWg.data(), rWu.data(), rWd.data(),
                           ri_rg, ri_ru, ri_rd, m, h, inter, s,
                           opt_cfg.learning_rate, opt_cfg.beta1,
                           opt_cfg.beta2, opt_cfg.epsilon,
                           opt_cfg.weight_decay);
    }

    // Device: each rank steps its own shard over the same K steps.
    const int inter_local = inter / device_count;
    std::vector<std::vector<float>> sg(device_count,
                                      std::vector<float>(h * inter_local));
    std::vector<std::vector<float>> su(device_count,
                                      std::vector<float>(h * inter_local));
    std::vector<std::vector<float>> sd(device_count,
                                      std::vector<float>(inter_local * h));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_X(m * h), d_out(m * h),
            d_dlogits(m * h);
        cuda::memory::Buffer<int> d_targets(m);
        cuda::memory::Buffer<float> d_dX(m * h);
        cuda::memory::Buffer<float> d_dWg(h * inter), d_dWu(h * inter),
            d_dWd(inter * h);
        d_X.copy_from(X.data(), X.size());
        d_targets.copy_from(targets.data(), m);

        TensorParallelMLP mlp(ctx, h, inter);
        mlp.set_weight(Wg0.data(), Wu0.data(), Wd0.data());
        AdamWOptimizer opt(opt_cfg);
        CrossEntropyConfig ce_cfg;
        ce_cfg.num_classes = h;
        ce_cfg.reduction_mean = true;

        for (int s = 1; s <= K; ++s) {
            mlp.forward(d_X.data(), d_out.data(), m, 1);
            // Seed the chain. The MLP output is replicated (the down row layer
            // AllReduces), so every rank computes the identical dlogits =
            // (softmax - onehot)/B from the same full-width logits — no comm.
            cross_entropy_logits_backward(
                d_out.data(), d_targets.data(), d_dlogits.data(),
                m, h, ce_cfg, nullptr);
            mlp.backward(d_X.data(), d_dlogits.data(), d_dX.data(),
                         d_dWg.data(), d_dWu.data(), d_dWd.data(), m, 1);
            EXPECT_NO_THROW(
                mlp.step(opt, d_dWg.data(), d_dWu.data(), d_dWd.data(), s));
        }

        mlp.copy_weights(sg[d].data(), su[d].data(), sd[d].data());
    });

    // Assemble the per-rank shards into the full weights and compare to the
    // host fp64 reference.
    std::vector<float> aWg(h * inter), aWu(h * inter), aWd(inter * h);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < h; ++a) {
            for (int b = 0; b < inter_local; ++b) {
                aWg[a * inter + r * inter_local + b] =
                    sg[r][a * inter_local + b];
                aWu[a * inter + r * inter_local + b] =
                    su[r][a * inter_local + b];
            }
        }
        for (int a = 0; a < inter_local; ++a) {
            for (int b = 0; b < h; ++b) {
                aWd[(r * inter_local + a) * h + b] =
                    sd[r][a * h + b];
            }
        }
    }
    // 2e-2 absorbs fp32 accumulate-order + shard-step noise across K=3 steps.
    EXPECT_TRUE(arrays_near(rWg.data(), aWg.data(), h * inter, 2e-2f))
        << "multi-GPU trained gate weights differ from the host full-weight "
           "reference";
    EXPECT_TRUE(arrays_near(rWu.data(), aWu.data(), h * inter, 2e-2f))
        << "multi-GPU trained up weights differ from the host full-weight "
           "reference";
    EXPECT_TRUE(arrays_near(rWd.data(), aWd.data(), inter * h, 2e-2f))
        << "multi-GPU trained down weights differ from the host full-weight "
           "reference";
}
