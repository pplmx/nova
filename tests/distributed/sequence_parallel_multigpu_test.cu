/**
 * @file sequence_parallel_multigpu_test.cu
 * @brief Real thread-per-rank multi-GPU tests for SequenceParallelAttention
 *
 * Milestone v2.19 P3: SequenceParallelAttention's multi-GPU paths
 * (gather_kv AllGather, scatter_output ReduceScatter, all_reduce_sequence
 * AllReduce) have never been driven with a real NCCL communicator — every test
 * in tests/distributed/sequence_parallel_test.cpp passes comm = nullptr, so
 * the #if NOVA_NCCL_ENABLED + real-collective branches are never-run.
 *
 * Per the R15-R19 pattern (never-run multi-GPU surfaces hide real bugs), these
 * tests construct a real NcclContext communicator group (one thread per rank
 * via distributed_test_common.h), hand each rank its own communicator to
 * SequenceParallelAttention, and assert the collective contracts against host
 * references:
 *   - all_reduce_sequence: every rank ends with the sum of all ranks' data.
 *   - gather_kv: gathered[t*L + i] == rank t's local[i] for all t.
 *   - scatter_output (ReduceScatter): identical full_output on every rank gives
 *     local = sp * full_output[rank's block] (reduce-scatter sums the
 *     replicated input across ranks).
 */

#include <gtest/gtest.h>

#include "cuda/distributed/sequence_parallel.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/stream/stream.h"

#include "distributed_test_common.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

using namespace cuda::distributed;
using namespace cuda::mesh;

using cuda::distributed::test::run_per_rank;

namespace {

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
    static std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

testing::AssertionResult arrays_near(
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

}  // namespace

class SequenceParallelMultiGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }

    static void SetUpTestSuite() {
        cudaFree(nullptr);
    }
};

// all_reduce_sequence: per-rank distinct data must become the global sum on
// every rank.
TEST_F(SequenceParallelMultiGpuTest, MultiGpu_AllReduceSequence) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU sequence-parallel test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU sequence-parallel requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int L = 64;
    const int sp = device_count;

    // Reference: element i of the global sum over all ranks.
    std::vector<float> ref(L, 0.0f);
    std::vector<std::vector<float>> rank_input(sp, std::vector<float>(L));
    for (int r = 0; r < sp; ++r) {
        for (int i = 0; i < L; ++i) {
            rank_input[r][i] = static_cast<float>(r * 100 + i + 1);
            ref[i] += rank_input[r][i];
        }
    }

    std::vector<std::vector<float>> h_results(sp, std::vector<float>(L));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> data(L);
        data.copy_from(rank_input[d].data(), L);

        SequenceParallelConfig cfg{
            .num_model_parallel_gpus = sp,
            .sequence_parallel_size = sp,
            .reduce_scatter_output = true,
            .rank = d,
            .world_size = sp,
            .comm = static_cast<void*>(ctx.current_comm()),
        };
        SequenceParallelAttention sp_attn(cfg);
        cuda::stream::Stream stream;
        sp_attn.all_reduce_sequence(data, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        data.copy_to(h_results[d].data(), L);
    });

    for (int rank = 0; rank < sp; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(ref.data(), h_results[rank].data(), L, 1e-3f))
            << "all_reduce_sequence result on rank " << rank
            << " is not the global sum";
    }
}

// gather_kv: each rank's local_k/local_v must land in its own rank slot of the
// gathered buffers on every rank.
TEST_F(SequenceParallelMultiGpuTest, MultiGpu_GatherKV) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU sequence-parallel test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU sequence-parallel requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int L = 32;
    const int sp = device_count;

    // Expected gathered buffer after all-gather: slot r holds rank r's local data.
    std::vector<float> ref_k(sp * L, 0.0f), ref_v(sp * L, 0.0f);
    std::vector<std::vector<float>> rank_k(sp, std::vector<float>(L));
    std::vector<std::vector<float>> rank_v(sp, std::vector<float>(L));
    for (int r = 0; r < sp; ++r) {
        for (int i = 0; i < L; ++i) {
            rank_k[r][i] = static_cast<float>(r * 64 + i);
            rank_v[r][i] = static_cast<float>(r * 64 - i);
            ref_k[r * L + i] = rank_k[r][i];
            ref_v[r * L + i] = rank_v[r][i];
        }
    }

    std::vector<std::vector<float>> h_gk(sp, std::vector<float>(sp * L));
    std::vector<std::vector<float>> h_gv(sp, std::vector<float>(sp * L));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> local_k(L), local_v(L);
        local_k.copy_from(rank_k[d].data(), L);
        local_v.copy_from(rank_v[d].data(), L);
        cuda::memory::Buffer<float> gathered_k(0), gathered_v(0);

        SequenceParallelConfig cfg{
            .num_model_parallel_gpus = sp,
            .sequence_parallel_size = sp,
            .reduce_scatter_output = true,
            .rank = d,
            .world_size = sp,
            .comm = static_cast<void*>(ctx.current_comm()),
        };
        SequenceParallelAttention sp_attn(cfg);
        cuda::stream::Stream stream;
        sp_attn.gather_kv(gathered_k, gathered_v, local_k, local_v, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        gathered_k.copy_to(h_gk[d].data(), sp * L);
        gathered_v.copy_to(h_gv[d].data(), sp * L);
    });

    for (int rank = 0; rank < sp; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(ref_k.data(), h_gk[rank].data(), sp * L, 1e-3f))
            << "gathered K on rank " << rank << " is not the all-gathered reference";
        EXPECT_TRUE(arrays_near(ref_v.data(), h_gv[rank].data(), sp * L, 1e-3f))
            << "gathered V on rank " << rank << " is not the all-gathered reference";
    }
}

// scatter_output with ReduceScatter: on identical full_output across ranks,
// each rank's local slice is sp * full_output[rank's block].
TEST_F(SequenceParallelMultiGpuTest, MultiGpu_ScatterReduceScatter) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU sequence-parallel test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU sequence-parallel requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int L = 32;
    const int sp = device_count;

    std::vector<float> full(sp * L);
    fill_random(full.data(), sp * L);

    // Reference: reduce-scatter of an identical replicated input sums sp copies.
    std::vector<std::vector<float>> h_results(sp, std::vector<float>(L));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> full_out(sp * L), local_out(L);
        full_out.copy_from(full.data(), sp * L);

        SequenceParallelConfig cfg{
            .num_model_parallel_gpus = sp,
            .sequence_parallel_size = sp,
            .reduce_scatter_output = true,
            .rank = d,
            .world_size = sp,
            .comm = static_cast<void*>(ctx.current_comm()),
        };
        SequenceParallelAttention sp_attn(cfg);
        cuda::stream::Stream stream;
        sp_attn.scatter_output(local_out, full_out, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        local_out.copy_to(h_results[d].data(), L);
    });

    for (int rank = 0; rank < sp; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        std::vector<float> ref(L);
        for (int i = 0; i < L; ++i) {
            ref[i] = static_cast<float>(sp) * full[rank * L + i];
        }
        EXPECT_TRUE(arrays_near(ref.data(), h_results[rank].data(), L, 1e-3f))
            << "reduce-scatter output on rank " << rank
            << " is not sp * replicated full_output block";
    }
}

// RingSequenceParallelism's multi-GPU ring attention requires the feature dim
// (config.hidden_dim) to interpret the Q/K/V buffers; without it the call must
// fail fast (the pre-v2.22 path was a silent garbage no-op — issue-v19-ring-
// parallel-noop). This pins the input-validation contract of the new path.
TEST_F(SequenceParallelMultiGpuTest, MultiGpu_RingAttention_RejectsMissingHiddenDim) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU sequence-parallel test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU sequence-parallel requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int sp = device_count;
    EXPECT_THROW(
        run_per_rank(device_count, [&](int d) {
            cuda::memory::Buffer<float> q(32), k(32), v(32), out(32);
            SequenceParallelConfig cfg{
                .num_model_parallel_gpus = sp,
                .sequence_parallel_size = sp,
                .reduce_scatter_output = true,
                .rank = d,
                .world_size = sp,
                .comm = static_cast<void*>(ctx.current_comm()),
                // hidden_dim omitted: must be rejected, not silently computed.
            };
            RingSequenceParallelism ring(cfg);
            cuda::stream::Stream stream;
            ring.ring_attention(q, k, v, out, stream);
        }),
        std::exception)
        << "multi-GPU ring attention with unset hidden_dim must fail fast, not "
           "misinterpret the buffers";
}

// ============================================================================
// v2.22 ring-attention acceptance — RingSequenceParallelism on real multi-GPU
// (TASK-014 / issue-v19-ring-parallel-noop). P1 (TASK-015) shipped this as RED
// against the DEC-004 fail-fast throw; P2 (TASK-016) implemented the ring and
// it is now the GREEN acceptance gate.
// ============================================================================

// Contract for the real ring attention: each rank owns a local sequence shard
// of Q/K/V; ring_attention must return, for the rank's local query tokens, the
// attention over the FULL KV sequence across every rank (the ring walks P-1
// remote blocks via send/recv, accumulating with online softmax). The
// reference is standard scaled dot-product attention over the concat of all
// ranks' K/V, computed in double precision on host.
TEST_F(SequenceParallelMultiGpuTest, MultiGpu_RingAttention_MatchesFullSequenceReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU sequence-parallel test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU sequence-parallel requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int sp = device_count;
    const int hidden = 8;
    const int local_seq = 4;                 // per-rank query/KV tokens
    const int full_seq = sp * local_seq;

    // Random per-rank local Q/K/V [local_seq x hidden].
    std::vector<std::vector<float>> h_q(sp), h_k(sp), h_v(sp);
    for (int r = 0; r < sp; ++r) {
        h_q[r].resize(static_cast<size_t>(local_seq) * hidden);
        h_k[r].resize(static_cast<size_t>(local_seq) * hidden);
        h_v[r].resize(static_cast<size_t>(local_seq) * hidden);
        fill_random(h_q[r].data(), h_q[r].size());
        fill_random(h_k[r].data(), h_k[r].size());
        fill_random(h_v[r].data(), h_v[r].size());
    }

    // Assemble the full sequence K/V (concat along seq) and compute, per rank,
    // the double-precision reference out_r = softmax(Q_r K^T / sqrt(h)) V.
    std::vector<float> k_full(static_cast<size_t>(full_seq) * hidden);
    std::vector<float> v_full(static_cast<size_t>(full_seq) * hidden);
    for (int r = 0; r < sp; ++r) {
        for (int t = 0; t < local_seq; ++t) {
            for (int c = 0; c < hidden; ++c) {
                k_full[static_cast<size_t>(r * local_seq + t) * hidden + c] =
                    h_k[r][static_cast<size_t>(t) * hidden + c];
                v_full[static_cast<size_t>(r * local_seq + t) * hidden + c] =
                    h_v[r][static_cast<size_t>(t) * hidden + c];
            }
        }
    }
    const double inv_sqrt_h = 1.0 / std::sqrt(static_cast<double>(hidden));
    std::vector<std::vector<float>> h_ref(sp, std::vector<float>(
        static_cast<size_t>(local_seq) * hidden));
    for (int r = 0; r < sp; ++r) {
        for (int i = 0; i < local_seq; ++i) {
            // scores over the full sequence (double precision).
            std::vector<double> scores(full_seq);
            double max_s = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < full_seq; ++j) {
                double dot = 0.0;
                for (int c = 0; c < hidden; ++c) {
                    dot += static_cast<double>(h_q[r][static_cast<size_t>(i) * hidden + c]) *
                           static_cast<double>(k_full[static_cast<size_t>(j) * hidden + c]);
                }
                scores[j] = dot * inv_sqrt_h;
                max_s = std::max(max_s, scores[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < full_seq; ++j) {
                scores[j] = std::exp(scores[j] - max_s);
                sum += scores[j];
            }
            for (int c = 0; c < hidden; ++c) {
                double acc = 0.0;
                for (int j = 0; j < full_seq; ++j) {
                    acc += (scores[j] / sum) *
                           static_cast<double>(v_full[static_cast<size_t>(j) * hidden + c]);
                }
                h_ref[r][static_cast<size_t>(i) * hidden + c] = static_cast<float>(acc);
            }
        }
    }

    std::vector<std::vector<float>> h_results(sp, std::vector<float>(
        static_cast<size_t>(local_seq) * hidden));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> q(local_seq * hidden), k(local_seq * hidden),
            v(local_seq * hidden), out(local_seq * hidden);
        q.copy_from(h_q[d].data(), h_q[d].size());
        k.copy_from(h_k[d].data(), h_k[d].size());
        v.copy_from(h_v[d].data(), h_v[d].size());

        SequenceParallelConfig cfg{
            .num_model_parallel_gpus = sp,
            .sequence_parallel_size = sp,
            .reduce_scatter_output = true,
            .rank = d,
            .world_size = sp,
            .comm = static_cast<void*>(ctx.current_comm()),
            .hidden_dim = hidden,
        };
        RingSequenceParallelism ring(cfg);
        cuda::stream::Stream stream;
        ring.ring_attention(q, k, v, out, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        out.copy_to(h_results[d].data(), h_results[d].size());
    });

    for (int rank = 0; rank < sp; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(h_ref[rank].data(), h_results[rank].data(),
                                h_ref[rank].size(), 1e-2f))
            << "ring attention output on rank " << rank
            << " differs from the full-sequence attention reference (the "
               "multi-GPU ring path is not computing full-sequence attention)";
    }
}

// scatter_output with reduce_scatter_output = false takes a D2D slice copy.
TEST_F(SequenceParallelMultiGpuTest, MultiGpu_ScatterSliceCopy) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU sequence-parallel test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU sequence-parallel requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int L = 32;
    const int sp = device_count;

    std::vector<float> full(sp * L);
    fill_random(full.data(), sp * L);

    std::vector<std::vector<float>> h_results(sp, std::vector<float>(L));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> full_out(sp * L), local_out(L);
        full_out.copy_from(full.data(), sp * L);

        SequenceParallelConfig cfg{
            .num_model_parallel_gpus = sp,
            .sequence_parallel_size = sp,
            .reduce_scatter_output = false,
            .rank = d,
            .world_size = sp,
            .comm = static_cast<void*>(ctx.current_comm()),
        };
        SequenceParallelAttention sp_attn(cfg);
        cuda::stream::Stream stream;
        sp_attn.scatter_output(local_out, full_out, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        local_out.copy_to(h_results[d].data(), L);
    });

    for (int rank = 0; rank < sp; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        EXPECT_TRUE(arrays_near(full.data() + rank * L, h_results[rank].data(), L, 1e-3f))
            << "slice-copy scatter output on rank " << rank
            << " is not full_output[rank's block]";
    }
}
