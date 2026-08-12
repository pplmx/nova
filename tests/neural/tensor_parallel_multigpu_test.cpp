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
