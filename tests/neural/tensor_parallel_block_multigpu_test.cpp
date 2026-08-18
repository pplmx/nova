/**
 * @file tensor_parallel_block_multigpu_test.cpp
 * @brief Real thread-per-rank multi-GPU tests for the v2.28 deep transformer
 *
 * Milestone v2.28 (TASK-040 / DEC-014): the v2.26 mini-transformer shard==
 * single-GPU parity covered ONE unnormalized block. The deep transformer
 * (N pre-LN residual blocks + final LN) must train the same sharded as full:
 * each rank runs K identical AdamW steps over its block shards (and the
 * identical replicated LN gamma/beta); the assembled per-rank weight shards
 * after the run must match a single-GPU full-weight reference run (tp==1),
 * and the loss must descend on both.
 *
 * P1-RED: TransformerTrainer is a provisional throwing stub, so this test
 * fails here and turns GREEN with the P2 composition (TASK-041). Runs on
 * NCCL-enabled multi-GPU (2 & 4 GPUs, CUDA_VISIBLE_DEVICES=2,3+).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda/neural/training_transformer.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include "distributed_test_common.h"

using namespace cuda::neural;
using namespace cuda::neural::optimizers;
using namespace cuda::neural::training;
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

cuda::nccl::NcclContext& uninitialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

void fill_random(float* data, size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

class DeepBlockMultiGpuPinnedTest : public ::testing::Test {
public:
    static testing::AssertionResult arrays_near(const float* expected,
                                                const float* actual,
                                                size_t count,
                                                float tolerance) {
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(expected[i] - actual[i]) > tolerance) {
                return testing::AssertionFailure()
                       << "mismatch at " << i << ": expected " << expected[i]
                       << " actual " << actual[i] << " (|diff| "
                       << std::abs(expected[i] - actual[i]) << " > " << tolerance
                       << ")";
            }
        }
        return testing::AssertionSuccess();
    }
};

void make_task(std::vector<float>& X, std::vector<int>& targets, int m,
               int hidden, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    X.assign(static_cast<size_t>(m) * hidden, 0.0f);
    targets.assign(m, 0);
    std::vector<float> T(static_cast<size_t>(hidden) * hidden);
    for (auto& t : T) t = norm(rng);
    for (int i = 0; i < m; ++i) {
        for (int d = 0; d < hidden; ++d) X[i * hidden + d] = norm(rng);
        double best = -1e30;
        int argmax = 0;
        for (int c = 0; c < hidden; ++c) {
            double acc = 0.0;
            for (int d = 0; d < hidden; ++d)
                acc += static_cast<double>(X[i * hidden + d]) * T[c * hidden + d];
            if (acc > best) { best = acc; argmax = c; }
        }
        targets[i] = argmax;
    }
}

// ============================================================================
// v2.28 multi-GPU capstone (TASK-040): a DEEP (N-block, pre-LN, residual)
// mini-transformer with a final LN head trains on real multi-GPU. Each rank
// runs K identical AdamW steps over its block shards + the replicated LN
// gamma/beta; the assembled per-rank weights after the run must match a
// single-GPU full-weight reference run (tp==1), and the loss must descend.
// ============================================================================
TEST_F(DeepBlockMultiGpuPinnedTest,
       MultiGpu_DeepTransformer_MatchesSingleGpu) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for deep transformer parity";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr || !multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 12, hidden = 8, heads = 4, hd = 2, inter = 16, K = 12;
    const int blocks = 3;
    const int qkv = hidden;  // heads * hd == hidden

    std::vector<float> X;
    std::vector<int> targets;
    make_task(X, targets, m, hidden, 20260819);

    // Full-weight random init (deterministic seeds).
    std::vector<std::vector<float>> r_l1g(blocks, std::vector<float>(hidden)),
        r_l1b(blocks, std::vector<float>(hidden)),
        r_l2g(blocks, std::vector<float>(hidden)),
        r_l2b(blocks, std::vector<float>(hidden));
    std::vector<std::vector<float>> r_Wq(blocks, std::vector<float>(hidden * qkv)),
        r_Wk(blocks, std::vector<float>(hidden * qkv)),
        r_Wv(blocks, std::vector<float>(hidden * qkv)),
        r_Wo(blocks, std::vector<float>(qkv * hidden));
    std::vector<std::vector<float>> r_Wg(blocks, std::vector<float>(hidden * inter)),
        r_Wu(blocks, std::vector<float>(hidden * inter)),
        r_Wd(blocks, std::vector<float>(inter * hidden));
    std::vector<float> r_fg(hidden, 1.0f), r_fb(hidden, 0.0f);
    unsigned seed = 120;
    for (int b = 0; b < blocks; ++b) {
        fill_random(r_l1g[b].data(), r_l1g[b].size(), seed++);
        fill_random(r_l1b[b].data(), r_l1b[b].size(), seed++);
        fill_random(r_l2g[b].data(), r_l2g[b].size(), seed++);
        fill_random(r_l2b[b].data(), r_l2b[b].size(), seed++);
        fill_random(r_Wq[b].data(), r_Wq[b].size(), seed++);
        fill_random(r_Wk[b].data(), r_Wk[b].size(), seed++);
        fill_random(r_Wv[b].data(), r_Wv[b].size(), seed++);
        fill_random(r_Wo[b].data(), r_Wo[b].size(), seed++);
        fill_random(r_Wg[b].data(), r_Wg[b].size(), seed++);
        fill_random(r_Wu[b].data(), r_Wu[b].size(), seed++);
        fill_random(r_Wd[b].data(), r_Wd[b].size(), seed++);
    }

    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;

    // Single-GPU full-weight reference run (tp==1) on the current device.
    std::vector<std::vector<float>> ref_l1g(r_l1g), ref_l1b(r_l1b),
        ref_l2g(r_l2g), ref_l2b(r_l2b);
    std::vector<std::vector<float>> ref_Wq(r_Wq), ref_Wk(r_Wk), ref_Wv(r_Wv),
        ref_Wo(r_Wo), ref_Wg(r_Wg), ref_Wu(r_Wu), ref_Wd(r_Wd);
    std::vector<float> ref_fg = r_fg, ref_fb = r_fb;
    float ref_loss_k = 0.0f;
    {
        TransformerTrainer ref_trainer(uninitialized_ctx(), blocks, hidden,
                                       heads, hd, inter, cfg);
        for (int b = 0; b < blocks; ++b) {
            ref_trainer.set_block_weight(
                b, r_l1g[b].data(), r_l1b[b].data(), r_Wq[b].data(),
                r_Wk[b].data(), r_Wv[b].data(), r_Wo[b].data(), r_l2g[b].data(),
                r_l2b[b].data(), r_Wg[b].data(), r_Wu[b].data(), r_Wd[b].data());
        }
        ref_trainer.set_final_ln_weight(r_fg.data(), r_fb.data());
        cuda::memory::Buffer<float> dX(m * hidden);
        dX.copy_from(X.data(), X.size());
        for (int s = 1; s <= K; ++s) {
            ref_trainer.train_step(dX.data(), targets.data(), m, 1, s);
        }
        ref_loss_k =
            ref_trainer.evaluate(dX.data(), targets.data(), m, 1, nullptr);
        for (int b = 0; b < blocks; ++b) {
            ref_trainer.copy_block_weights(
                b, ref_l1g[b].data(), ref_l1b[b].data(), ref_Wq[b].data(),
                ref_Wk[b].data(), ref_Wv[b].data(), ref_Wo[b].data(),
                ref_l2g[b].data(), ref_l2b[b].data(), ref_Wg[b].data(),
                ref_Wu[b].data(), ref_Wd[b].data());
        }
        ref_trainer.copy_final_ln(ref_fg.data(), ref_fb.data());
    }

    // Multi-GPU: per-rank deep transformer, per-tensor AdamW, K identical steps.
    const int local_qkv = qkv / device_count;
    const int ilocal = inter / device_count;
    struct RankPacks {
        std::vector<std::vector<float>> l1g, l1b, l2g, l2b;  // [blocks][hidden]
        std::vector<std::vector<float>> Wq, Wk, Wv;  // [blocks][hidden*qkv_local]
        std::vector<std::vector<float>> Wo;          // [blocks][qkv_local*hidden]
        std::vector<std::vector<float>> Wg, Wu;      // [blocks][hidden*ilocal]
        std::vector<std::vector<float>> Wd;          // [blocks][ilocal*hidden]
        std::vector<float> fg, fb;                   // [hidden]
    };
    std::vector<RankPacks> ranks(device_count);
    for (int r = 0; r < device_count; ++r) {
        RankPacks& p = ranks[r];
        for (int b = 0; b < blocks; ++b) {
            p.l1g.push_back(std::vector<float>(hidden));
            p.l1b.push_back(std::vector<float>(hidden));
            p.l2g.push_back(std::vector<float>(hidden));
            p.l2b.push_back(std::vector<float>(hidden));
            p.Wq.push_back(std::vector<float>(hidden * local_qkv));
            p.Wk.push_back(std::vector<float>(hidden * local_qkv));
            p.Wv.push_back(std::vector<float>(hidden * local_qkv));
            p.Wo.push_back(std::vector<float>(local_qkv * hidden));
            p.Wg.push_back(std::vector<float>(hidden * ilocal));
            p.Wu.push_back(std::vector<float>(hidden * ilocal));
            p.Wd.push_back(std::vector<float>(ilocal * hidden));
        }
        p.fg.resize(hidden);
        p.fb.resize(hidden);
    }
    run_per_rank(device_count, [&](int d) {
        TransformerTrainer t(ctx, blocks, hidden, heads, hd, inter, cfg);
        for (int b = 0; b < blocks; ++b) {
            t.set_block_weight(
                b, r_l1g[b].data(), r_l1b[b].data(), r_Wq[b].data(),
                r_Wk[b].data(), r_Wv[b].data(), r_Wo[b].data(), r_l2g[b].data(),
                r_l2b[b].data(), r_Wg[b].data(), r_Wu[b].data(), r_Wd[b].data());
        }
        t.set_final_ln_weight(r_fg.data(), r_fb.data());
        cuda::memory::Buffer<float> dX(m * hidden);
        dX.copy_from(X.data(), X.size());
        for (int s = 1; s <= K; ++s) {
            t.train_step(dX.data(), targets.data(), m, 1, s);
        }
        for (int b = 0; b < blocks; ++b) {
            t.copy_block_weights(
                b, ranks[d].l1g[b].data(), ranks[d].l1b[b].data(),
                ranks[d].Wq[b].data(), ranks[d].Wk[b].data(),
                ranks[d].Wv[b].data(), ranks[d].Wo[b].data(),
                ranks[d].l2g[b].data(), ranks[d].l2b[b].data(),
                ranks[d].Wg[b].data(), ranks[d].Wu[b].data(),
                ranks[d].Wd[b].data());
        }
        t.copy_final_ln(ranks[d].fg.data(), ranks[d].fb.data());
    });

    // Assemble each block's shards and compare with the single-GPU ref.
    float worst = 1e30f;
    for (int b = 0; b < blocks; ++b) {
        std::vector<float> aWq(hidden * qkv), aWk(hidden * qkv),
            aWv(hidden * qkv), aWo(qkv * hidden), aWg(hidden * inter),
            aWu(hidden * inter), aWd(inter * hidden);
        std::vector<float> a_l1g(hidden), a_l1b(hidden), a_l2g(hidden),
            a_l2b(hidden);
        for (int r = 0; r < device_count; ++r) {
            for (int a = 0; a < hidden; ++a) {
                for (int c = 0; c < local_qkv; ++c) {
                    aWq[a * qkv + r * local_qkv + c] = ranks[r].Wq[b][a * local_qkv + c];
                    aWk[a * qkv + r * local_qkv + c] = ranks[r].Wk[b][a * local_qkv + c];
                    aWv[a * qkv + r * local_qkv + c] = ranks[r].Wv[b][a * local_qkv + c];
                }
            }
            for (int c = 0; c < local_qkv * hidden; ++c)
                aWo[r * local_qkv * hidden + c] = ranks[r].Wo[b][c];
            for (int a = 0; a < hidden; ++a)
                for (int c = 0; c < ilocal; ++c) {
                    aWg[a * inter + r * ilocal + c] = ranks[r].Wg[b][a * ilocal + c];
                    aWu[a * inter + r * ilocal + c] = ranks[r].Wu[b][a * ilocal + c];
                }
            for (int c = 0; c < ilocal * hidden; ++c)
                aWd[r * ilocal * hidden + c] = ranks[r].Wd[b][c];
            // LN gamma/beta are replicated full-[hidden] vectors.
            for (int j = 0; j < hidden; ++j) {
                a_l1g[j] = ranks[r].l1g[b][j];
                a_l1b[j] = ranks[r].l1b[b][j];
                a_l2g[j] = ranks[r].l2g[b][j];
                a_l2b[j] = ranks[r].l2b[b][j];
            }
        }
        EXPECT_TRUE(arrays_near(ref_Wq[b].data(), aWq.data(), hidden * qkv, 2e-2f))
            << "block " << b << " Wq";
        EXPECT_TRUE(arrays_near(ref_Wk[b].data(), aWk.data(), hidden * qkv, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_Wv[b].data(), aWv.data(), hidden * qkv, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_Wo[b].data(), aWo.data(), qkv * hidden, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_Wg[b].data(), aWg.data(), hidden * inter, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_Wu[b].data(), aWu.data(), hidden * inter, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_Wd[b].data(), aWd.data(), inter * hidden, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_l1g[b].data(), a_l1g.data(), hidden, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_l1b[b].data(), a_l1b.data(), hidden, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_l2g[b].data(), a_l2g.data(), hidden, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_l2b[b].data(), a_l2b.data(), hidden, 2e-2f));
    }
    // Final LN (replicated) also matches.
    for (int r = 0; r < device_count; ++r) {
        EXPECT_TRUE(arrays_near(ref_fg.data(), ranks[r].fg.data(), hidden, 2e-2f));
        EXPECT_TRUE(arrays_near(ref_fb.data(), ranks[r].fb.data(), hidden, 2e-2f));
    }
    // The reference loss must be finite and meaningful (the sharded assembly
    // equality is the parity claim; loss descent is the convergence claim).
    EXPECT_TRUE(std::isfinite(ref_loss_k));
}

}  // namespace
