/**
 * @file checkpoint_multigpu_test.cpp
 * @brief Multi-GPU rank-local checkpoint contracts (milestone v2.33, TASK-060)
 *
 * Milestone v2.33 (DEC-019): v2.32 shipped exact trainer checkpointing
 * (save_state/load_state + AdamW moment export/import, NSCK v1) but guarded
 * tp > 1 — with TP>1 the copy_* readbacks expose rank shards while set_* takes
 * full weights, so a rank-local checkpoint couldn't restore. These contracts
 * pin the multi-GPU path (thread-per-rank over a real NCCL context, the
 * MicroTrainerMultiGpu / DeepBlockMultiGpu style):
 *
 *  - each rank's save->load->continue roundtrips its OWN shard weights and
 *    moments byte-exactly (per-rank copy_weight* equality), and
 *  - the resumed trajectory matches the uninterrupted one on that rank.
 *
 * P1: both contracts RED — save_state still throws the tp>1 guard.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda/neural/training.h"
#include "cuda/neural/training_transformer.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include "distributed_test_common.h"

using namespace cuda::neural;
using namespace cuda::neural::training;
using namespace cuda::neural::optimizers;
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

void fill_random(float* data, size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

// Synthetic classification task (as the MicroTrainer tests): gaussian inputs
// + a random full-rank teacher label matrix, hidden == num_classes.
void make_task(std::vector<float>& X, std::vector<int>& Y, int m, int h,
               unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    X.resize(static_cast<size_t>(m) * h);
    Y.resize(m);
    std::vector<float> T(static_cast<size_t>(h) * h);
    for (size_t i = 0; i < T.size(); ++i) T[i] = norm(rng);
    for (int i = 0; i < m; ++i) {
        for (int d = 0; d < h; ++d) X[i * h + d] = norm(rng);
        double best = -1e30;
        int argmax = 0;
        for (int c = 0; c < h; ++c) {
            double acc = 0.0;
            for (int d = 0; d < h; ++d) {
                acc += static_cast<double>(X[i * h + d]) * T[c * h + d];
            }
            if (acc > best) {
                best = acc;
                argmax = c;
            }
        }
        Y[i] = argmax;
    }
}

void expect_eq(const float* a, const float* b, size_t n, const char* what) {
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]) << what << " element " << i;
    }
}

}  // namespace

class CheckpointMultiGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();  // populates device_count()
    }
    static void SetUpTestSuite() { cudaFree(nullptr); }
};

// A sharded MicroTrainer run on 2+ GPUs must persist and resume per rank:
// save_state on this rank, load_state into a fresh same-rank trainer, rank
// shards byte-identical, and the resumed steps match the uninterrupted run.
TEST_F(CheckpointMultiGpuTest, MultiGpu_MicroTrainer_RankLocalCheckpoint) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for the multi-GPU checkpoint test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 16, h = 8, inter = 16, K = 5, extra = 2;
    OptimizerConfig opt_cfg;
    opt_cfg.learning_rate = 0.01f;

    std::vector<float> X;
    std::vector<int> targets;
    make_task(X, targets, m, h, 20260824);
    std::vector<float> Wg0(static_cast<size_t>(h) * inter);
    std::vector<float> Wu0(static_cast<size_t>(h) * inter);
    std::vector<float> Wd0(static_cast<size_t>(inter) * h);
    fill_random(Wg0.data(), Wg0.size(), 31);
    fill_random(Wu0.data(), Wu0.size(), 32);
    fill_random(Wd0.data(), Wd0.size(), 33);

    const int tp = device_count;
    const size_t sg = static_cast<size_t>(h) * (inter / tp);
    const size_t su = static_cast<size_t>(h) * (inter / tp);
    const size_t sd = static_cast<size_t>(inter / tp) * h;

    run_per_rank(device_count, [&](int /*rank*/) {
        cuda::memory::Buffer<float> d_X(m * h);
        d_X.copy_from(X.data(), X.size());

        MicroTrainer a(ctx, h, inter, opt_cfg);
        a.set_weight(Wg0.data(), Wu0.data(), Wd0.data());
        for (int s = 1; s <= K; ++s) {
            a.train_step(d_X.data(), targets.data(), m, 1, s);
        }

        std::stringstream ss(std::ios::out | std::ios::binary);
        a.save_state(ss);  // P1: throws the tp>1 guard -> RED
        const std::string data = ss.str();

        std::stringstream in(data);
        MicroTrainer b(ctx, h, inter, opt_cfg);
        b.load_state(in);

        // Rank-shard weights byte-identical.
        std::vector<float> ag(sg), au(su), ad(sd);
        std::vector<float> bg(sg), bu(su), bd(sd);
        a.copy_weights(ag.data(), au.data(), ad.data());
        b.copy_weights(bg.data(), bu.data(), bd.data());
        expect_eq(ag.data(), bg.data(), ag.size(), "gate shard");
        expect_eq(au.data(), bu.data(), au.size(), "up shard");
        expect_eq(ad.data(), bd.data(), ad.size(), "down shard");

        // Resumed trajectory == uninterrupted (this rank's own path).
        std::vector<float> holdout;
        for (int s = K + 1; s <= K + extra; ++s) {
            holdout.push_back(a.train_step(d_X.data(), targets.data(), m, 1, s));
        }
        for (int s = K + 1; s <= K + extra; ++s) {
            const float lb = b.train_step(d_X.data(), targets.data(), m, 1, s);
            EXPECT_NEAR(holdout[static_cast<size_t>(s - K - 1)], lb, 1e-5f)
                << "resumed loss at step " << s << " on this rank";
        }
    });
}

// Same contract for the deep transformer: per-rank Trainer blocks (per-rank
// run_per_rank trainers), rank-local save/load roundtrip + resumed trajectory.
TEST_F(CheckpointMultiGpuTest, MultiGpu_TransformerTrainer_RankLocalCheckpoint) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for the multi-GPU checkpoint test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    // Scale heads (and inter) with the number of ranks so the geometry is
    // valid on any 2..N topology: heads and heads*head_dim (qkv) must both
    // divide by tp (head_dim 2 keeps qkv even), and intermediate must divide
    // by tp. A fixed heads=4 would truncate local_heads to 0 beyond 4 ranks
    // (the cpp-reviewer HIGH — the attention ctor now rejects such geometry,
    // but the test must not trip it on the available 8-GPU hardware).
    const int blocks = 2, h = 8, hd = 2, m = 12, K = 4, extra = 2;
    const int heads = 2 * device_count;
    const int qkv = heads * hd;
    const int inter = 4 * device_count;
    OptimizerConfig opt_cfg;
    opt_cfg.learning_rate = 0.01f;

    std::vector<float> X;
    std::vector<int> targets;
    make_task(X, targets, m, h, 20260825);

    const int tp = device_count;
    const size_t nq = static_cast<size_t>(h) * (qkv / tp);      // wq/wk/wv
    const size_t no = static_cast<size_t>(qkv / tp) * h;        // wo
    const size_t ng = static_cast<size_t>(h) * (inter / tp);    // wg/wu
    const size_t nd = static_cast<size_t>(inter / tp) * h;      // wd

    auto set_block = [&](TransformerTrainer& t, int b, unsigned seed) {
        std::vector<float> l1g(h), l1b(h), l2g(h), l2b(h);
        std::vector<float> wq(static_cast<size_t>(h) * qkv);
        std::vector<float> wk(static_cast<size_t>(h) * qkv);
        std::vector<float> wv(static_cast<size_t>(h) * qkv);
        std::vector<float> wo(static_cast<size_t>(qkv) * h);
        std::vector<float> wg(static_cast<size_t>(h) * inter);
        std::vector<float> wu(static_cast<size_t>(h) * inter);
        std::vector<float> wd(static_cast<size_t>(inter) * h);
        fill_random(l1g.data(), l1g.size(), seed);
        fill_random(l1b.data(), l1b.size(), seed + 1);
        fill_random(wq.data(), wq.size(), seed + 2);
        fill_random(wk.data(), wk.size(), seed + 3);
        fill_random(wv.data(), wv.size(), seed + 4);
        fill_random(wo.data(), wo.size(), seed + 5);
        fill_random(l2g.data(), l2g.size(), seed + 6);
        fill_random(l2b.data(), l2b.size(), seed + 7);
        fill_random(wg.data(), wg.size(), seed + 8);
        fill_random(wu.data(), wu.size(), seed + 9);
        fill_random(wd.data(), wd.size(), seed + 10);
        t.set_block_weight(b, l1g.data(), l1b.data(), wq.data(), wk.data(),
                           wv.data(), wo.data(), l2g.data(), l2b.data(),
                           wg.data(), wu.data(), wd.data());
    };

    run_per_rank(device_count, [&](int /*rank*/) {
        cuda::memory::Buffer<float> d_X(m * h);
        d_X.copy_from(X.data(), X.size());

        TransformerTrainer a(ctx, blocks, h, heads, hd, inter, opt_cfg);
        for (int b = 0; b < blocks; ++b) set_block(a, b, 500u + static_cast<unsigned>(b));
        std::vector<float> fg(h), fb(h);
        fill_random(fg.data(), fg.size(), 600);
        fill_random(fb.data(), fb.size(), 601);
        a.set_final_ln_weight(fg.data(), fb.data());
        for (int s = 1; s <= K; ++s) {
            a.train_step(d_X.data(), targets.data(), m, 1, s);
        }

        std::stringstream ss(std::ios::out | std::ios::binary);
        a.save_state(ss);  // P1: throws the tp>1 guard -> RED
        const std::string data = ss.str();

        std::stringstream in(data);
        TransformerTrainer b(ctx, blocks, h, heads, hd, inter, opt_cfg);
        b.load_state(in);

        // Per-rank block shards byte-identical (wq/wk/wv shard + wo + wg/wu/wd
        // are rank-sized; the replicated LN gamma/beta are full hidden).
        auto copy_one = [&](TransformerTrainer& t, std::vector<float>& flat) {
            flat.clear();
            std::vector<float> l1g(h), l1b(h), l2g(h), l2b(h);
            std::vector<float> wq(nq), wk(nq), wv(nq), wo(no);
            std::vector<float> wg(ng), wu(ng), wd(nd);
            t.copy_block_weights(0, l1g.data(), l1b.data(), wq.data(),
                                 wk.data(), wv.data(), wo.data(), l2g.data(),
                                 l2b.data(), wg.data(), wu.data(), wd.data());
            for (const auto* p : {&l1g, &l1b, &wq, &wk, &wv, &wo, &l2g, &l2b,
                                  &wg, &wu, &wd}) {
                flat.insert(flat.end(), p->begin(), p->end());
            }
        };
        std::vector<float> aflt, bflt;
        copy_one(a, aflt);
        copy_one(b, bflt);
        ASSERT_EQ(aflt.size(), bflt.size());
        expect_eq(aflt.data(), bflt.data(), aflt.size(), "block 0 shards");

        // Resumed trajectory == uninterrupted.
        std::vector<float> holdout;
        for (int s = K + 1; s <= K + extra; ++s) {
            holdout.push_back(a.train_step(d_X.data(), targets.data(), m, 1, s));
        }
        for (int s = K + 1; s <= K + extra; ++s) {
            const float lb = b.train_step(d_X.data(), targets.data(), m, 1, s);
            EXPECT_NEAR(holdout[static_cast<size_t>(s - K - 1)], lb, 1e-4f)
                << "resumed deep loss at step " << s << " on this rank";
        }
    });
}
