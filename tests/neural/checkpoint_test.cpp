/**
 * @file checkpoint_test.cpp
 * @brief Model checkpointing contracts (milestone v2.32, TASK-056)
 *
 * Milestone v2.32 (DEC-018): the host-round-trip family is complete and the
 * train stack is real, but nothing persists — a trained MicroTrainer /
 * TransformerTrainer dies with the process, interrupted runs can't resume, and
 * models can't be exported. This file pins the new persistence surface on the
 * tp == 1 path (uninitialized NcclContext -> effective tp degree 1):
 *
 *  - AdamWOptimizer moment export/import (the optimizer-state path a resume
 *    needs — AdamW reads m/v, so weights alone diverge at the resumed step),
 *  - MicroTrainer::save_state / load_state,
 *  - TransformerTrainer::save_state / load_state,
 *
 * against these contracts: save->load roundtrip is byte-exact on weights and
 * moments; a restored trainer resumes to the same trajectory as the
 * uninterrupted one; a fresh (never-stepped) trainer roundtrips through the
 * zero-moment path; and a geometry/corrupt/truncated stream is rejected rather
 * than mis-restored. P1: all contracts RED against throw-stubs.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "cuda/neural/training.h"
#include "cuda/neural/training_transformer.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::neural::training;
using namespace cuda::neural::optimizers;

namespace {

cuda::nccl::NcclContext& uninitialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

void fill_random(float* data, size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) data[i] = dist(rng);
}

// Synthetic classification task (same generative model as the MicroTrainer
// tests): gaussian inputs + a random full-rank teacher label matrix, with
// hidden == num_classes.
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

void expect_near(const float* a, const float* b, size_t n, float tol,
                 const char* what) {
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << what << " element " << i;
    }
}

}  // namespace

class CheckpointTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        cudaFree(nullptr);  // Ensure a CUDA context for device-dependent calls
    }
};

// The optimizer-level oracle: exporting a stepped AdamW's m/v and importing
// them into a fresh optimizer must reproduce the exact next update. params /
// grads are device buffers (step() takes device pointers — v2.27).
TEST_F(CheckpointTest, AdamWMomentExportImportRoundtrip) {
    const size_t n = 256;
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    cfg.weight_decay = 0.0f;
    std::vector<float> p0(n), p_a(n), p_b(n), grads(n);
    fill_random(p0.data(), n, 101);
    fill_random(grads.data(), n, 102);
    cuda::memory::Buffer<float> d_a(n), d_b(n), d_g(n);
    d_a.copy_from(p0.data(), n);
    d_g.copy_from(grads.data(), n);

    AdamWOptimizer a(cfg);
    for (int s = 1; s <= 3; ++s) {
        a.step(d_a.data(), d_g.data(), n, s);
    }
    ASSERT_EQ(a.momentum_capacity(), n);

    std::vector<float> m(n), v(n);
    a.copy_moments_to(m.data(), v.data(), n, nullptr);

    AdamWOptimizer b(cfg);
    ASSERT_EQ(b.momentum_capacity(), 0u);
    b.copy_moments_from(m.data(), v.data(), n, nullptr);
    ASSERT_EQ(b.momentum_capacity(), n);

    // Both step once from the identical pre-step-4 state: pull A's post-3
    // params to host and stage B's params from them, so the only difference
    // is where the moments came from (A accumulated them, B imported them).
    d_a.copy_to(p_a.data(), n);
    d_b.copy_from(p_a.data(), n);
    a.step(d_a.data(), d_g.data(), n, 4);
    b.step(d_b.data(), d_g.data(), n, 4);
    d_a.copy_to(p_a.data(), n);
    d_b.copy_to(p_b.data(), n);
    expect_eq(p_a.data(), p_b.data(), n, "param");
}

// After warm training, a MicroTrainer save->load into a fresh trainer must be
// byte-exact on all three weight tensors.
TEST_F(CheckpointTest, MicroTrainerSaveLoadRoundTripIsByteExact) {
    const int h = 16, inter = 32, m = 8, K = 4;
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    MicroTrainer a(uninitialized_ctx(), h, inter, cfg);

    std::vector<float> Wg(static_cast<size_t>(h) * inter);
    std::vector<float> Wu(static_cast<size_t>(h) * inter);
    std::vector<float> Wd(static_cast<size_t>(inter) * h);
    fill_random(Wg.data(), Wg.size(), 201);
    fill_random(Wu.data(), Wu.size(), 202);
    fill_random(Wd.data(), Wd.size(), 203);
    a.set_weight(Wg.data(), Wu.data(), Wd.data());

    std::vector<float> X;
    std::vector<int> Y;
    make_task(X, Y, m, h, 204);
    cuda::memory::Buffer<float> d_X(m * h);
    d_X.copy_from(X.data(), X.size());
    for (int s = 1; s <= K; ++s) {
        a.train_step(d_X.data(), Y.data(), m, 1, s);  // warm weights + moments
    }

    std::stringstream ss(std::ios::out | std::ios::binary);
    a.save_state(ss);
    std::string data = ss.str();
    std::stringstream in(data);
    MicroTrainer b(uninitialized_ctx(), h, inter, cfg);
    b.load_state(in);

    std::vector<float> ag(Wg.size()), au(Wu.size()), ad(Wd.size());
    std::vector<float> bg(Wg.size()), bu(Wu.size()), bd(Wd.size());
    a.copy_weights(ag.data(), au.data(), ad.data());
    b.copy_weights(bg.data(), bu.data(), bd.data());
    expect_eq(ag.data(), bg.data(), ag.size(), "gate");
    expect_eq(au.data(), bu.data(), au.size(), "up");
    expect_eq(ad.data(), bd.data(), ad.size(), "down");
}

// Resuming a restored trainer must reproduce the uninterrupted trajectory: the
// restored weights AND moments make every resumed forward/backward/update
// bit-identical, not just "close".
TEST_F(CheckpointTest, MicroTrainerResumeAfterLoadMatchesUninterrupted) {
    const int h = 8, inter = 16, m = 16, K = 5, extra = 3;
    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
    MicroTrainer a(uninitialized_ctx(), h, inter, cfg);

    std::vector<float> Wg(static_cast<size_t>(h) * inter);
    std::vector<float> Wu(static_cast<size_t>(h) * inter);
    std::vector<float> Wd(static_cast<size_t>(inter) * h);
    fill_random(Wg.data(), Wg.size(), 301);
    fill_random(Wu.data(), Wu.size(), 302);
    fill_random(Wd.data(), Wd.size(), 303);
    a.set_weight(Wg.data(), Wu.data(), Wd.data());

    std::vector<float> X;
    std::vector<int> Y;
    make_task(X, Y, m, h, 304);
    cuda::memory::Buffer<float> d_X(m * h);
    d_X.copy_from(X.data(), X.size());
    for (int s = 1; s <= K; ++s) {
        a.train_step(d_X.data(), Y.data(), m, 1, s);
    }

    std::stringstream ss(std::ios::out | std::ios::binary);
    a.save_state(ss);
    std::string data = ss.str();

    // A keeps training (uninterrupted holdout) ...
    std::vector<float> holdout;
    for (int s = K + 1; s <= K + extra; ++s) {
        holdout.push_back(a.train_step(d_X.data(), Y.data(), m, 1, s));
    }
    // ... while B resumes from the checkpoint and must land on the same points.
    MicroTrainer b(uninitialized_ctx(), h, inter, cfg);
    std::stringstream in(data);
    b.load_state(in);
    for (int s = K + 1; s <= K + extra; ++s) {
        const float lb = b.train_step(d_X.data(), Y.data(), m, 1, s);
        EXPECT_NEAR(holdout[static_cast<size_t>(s - K - 1)], lb, 1e-5f)
            << "resumed loss at step " << s << " must match the uninterrupted run";
    }

    std::vector<float> ag(Wg.size()), au(Wu.size()), ad(Wd.size());
    std::vector<float> bg(Wg.size()), bu(Wu.size()), bd(Wd.size());
    a.copy_weights(ag.data(), au.data(), ad.data());
    b.copy_weights(bg.data(), bu.data(), bd.data());
    expect_near(ag.data(), bg.data(), ag.size(), 1e-6f, "gate");
    expect_near(au.data(), bu.data(), au.size(), 1e-6f, "up");
    expect_near(ad.data(), bd.data(), ad.size(), 1e-6f, "down");
}

// A never-stepped trainer has zero moments; its save/load must roundtrip the
// zero-moment path and leave both trainers stepping identically from scratch.
TEST_F(CheckpointTest, MicroTrainerFreshStateRoundTrip) {
    const int h = 8, inter = 16, m = 8;
    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
    MicroTrainer a(uninitialized_ctx(), h, inter, cfg);

    std::vector<float> Wg(static_cast<size_t>(h) * inter);
    std::vector<float> Wu(static_cast<size_t>(h) * inter);
    std::vector<float> Wd(static_cast<size_t>(inter) * h);
    fill_random(Wg.data(), Wg.size(), 401);
    fill_random(Wu.data(), Wu.size(), 402);
    fill_random(Wd.data(), Wd.size(), 403);
    a.set_weight(Wg.data(), Wu.data(), Wd.data());
    // No train_step: optimizers are fresh (momentum_capacity == 0).

    std::stringstream ss(std::ios::out | std::ios::binary);
    a.save_state(ss);
    std::string data = ss.str();
    std::stringstream in(data);
    MicroTrainer b(uninitialized_ctx(), h, inter, cfg);
    b.load_state(in);

    std::vector<float> bg(Wg.size()), bu(Wu.size()), bd(Wd.size());
    b.copy_weights(bg.data(), bu.data(), bd.data());
    expect_eq(Wg.data(), bg.data(), Wg.size(), "gate");
    expect_eq(Wu.data(), bu.data(), Wu.size(), "up");
    expect_eq(Wd.data(), bd.data(), Wd.size(), "down");

    std::vector<float> X;
    std::vector<int> Y;
    make_task(X, Y, m, h, 404);
    cuda::memory::Buffer<float> d_X(m * h);
    d_X.copy_from(X.data(), X.size());
    const float la = a.train_step(d_X.data(), Y.data(), m, 1, 1);
    const float lb = b.train_step(d_X.data(), Y.data(), m, 1, 1);
    EXPECT_NEAR(la, lb, 1e-6f)
        << "fresh restored trainer must take the same first step";
}

// The deep (TransformerTrainer) surface: all 11N+2 weight tensors + moments
// roundtrip byte-exactly and resume the deep trajectory.
TEST_F(CheckpointTest, TransformerTrainerSaveLoadRoundTripAndResume) {
    const int blocks = 2, h = 8, heads = 1, head_dim = 4, inter = 16;
    const int qkv = heads * head_dim;
    const int m = 8, K = 3, extra = 2;
    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
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
    auto copy_block = [&](TransformerTrainer& t, int b,
                          std::vector<float>& l1g, std::vector<float>& l1b,
                          std::vector<float>& wq, std::vector<float>& wk,
                          std::vector<float>& wv, std::vector<float>& wo,
                          std::vector<float>& l2g, std::vector<float>& l2b,
                          std::vector<float>& wg, std::vector<float>& wu,
                          std::vector<float>& wd) {
        l1g.resize(h);
        l1b.resize(h);
        l2g.resize(h);
        l2b.resize(h);
        wq.resize(static_cast<size_t>(h) * qkv);
        wk.resize(static_cast<size_t>(h) * qkv);
        wv.resize(static_cast<size_t>(h) * qkv);
        wo.resize(static_cast<size_t>(qkv) * h);
        wg.resize(static_cast<size_t>(h) * inter);
        wu.resize(static_cast<size_t>(h) * inter);
        wd.resize(static_cast<size_t>(inter) * h);
        t.copy_block_weights(b, l1g.data(), l1b.data(), wq.data(), wk.data(),
                             wv.data(), wo.data(), l2g.data(), l2b.data(),
                             wg.data(), wu.data(), wd.data());
    };

    TransformerTrainer a(uninitialized_ctx(), blocks, h, heads, head_dim, inter,
                         cfg);
    for (int b = 0; b < blocks; ++b) set_block(a, b, 500u + b);
    std::vector<float> fg(h), fb(h);
    fill_random(fg.data(), fg.size(), 600);
    fill_random(fb.data(), fb.size(), 601);
    a.set_final_ln_weight(fg.data(), fb.data());

    std::vector<float> X;
    std::vector<int> Y;
    make_task(X, Y, m, h, 602);
    cuda::memory::Buffer<float> d_X(m * h);
    d_X.copy_from(X.data(), X.size());
    for (int s = 1; s <= K; ++s) {
        a.train_step(d_X.data(), Y.data(), m, 1, s);
    }

    std::stringstream ss(std::ios::out | std::ios::binary);
    a.save_state(ss);
    std::string data = ss.str();

    // A's post-K weights (captured from the same logical state the checkpoint
    // was written from) must equal B's just-loaded weights byte-for-byte.
    std::vector<std::vector<float>> aw;
    for (int b = 0; b < blocks; ++b) {
        std::vector<float> l1g, l1b, wq, wk, wv, wo, l2g, l2b, wg, wu, wd;
        copy_block(a, b, l1g, l1b, wq, wk, wv, wo, l2g, l2b, wg, wu, wd);
        std::vector<float> flat;
        for (const auto* p : {&l1g, &l1b, &wq, &wk, &wv, &wo, &l2g, &l2b, &wg,
                              &wu, &wd}) {
            flat.insert(flat.end(), p->begin(), p->end());
        }
        aw.push_back(std::move(flat));
    }
    std::vector<float> afg(h), afb(h);
    a.copy_final_ln(afg.data(), afb.data());

    TransformerTrainer b(uninitialized_ctx(), blocks, h, heads, head_dim, inter,
                         cfg);
    std::stringstream in(data);
    b.load_state(in);
    for (int blk = 0; blk < blocks; ++blk) {
        std::vector<float> l1g, l1b, wq, wk, wv, wo, l2g, l2b, wg, wu, wd;
        copy_block(b, blk, l1g, l1b, wq, wk, wv, wo, l2g, l2b, wg, wu, wd);
        std::vector<float> flat;
        for (const auto* p : {&l1g, &l1b, &wq, &wk, &wv, &wo, &l2g, &l2b, &wg,
                              &wu, &wd}) {
            flat.insert(flat.end(), p->begin(), p->end());
        }
        ASSERT_EQ(aw[static_cast<size_t>(blk)].size(), flat.size());
        expect_eq(aw[static_cast<size_t>(blk)].data(), flat.data(),
                  flat.size(), "block weight");
    }
    std::vector<float> bfg(h), bfb(h);
    b.copy_final_ln(bfg.data(), bfb.data());
    expect_eq(afg.data(), bfg.data(), h, "final gamma");
    expect_eq(afb.data(), bfb.data(), h, "final beta");

    // Resume: A keeps training (holdout), B from the checkpoint. Attention's
    // global atomicAdd order is not run-to-run deterministic, so the deep
    // resumed trajectory matches to a tolerance, not bitwise.
    std::vector<float> holdout;
    for (int s = K + 1; s <= K + extra; ++s) {
        holdout.push_back(a.train_step(d_X.data(), Y.data(), m, 1, s));
    }
    for (int s = K + 1; s <= K + extra; ++s) {
        const float lb = b.train_step(d_X.data(), Y.data(), m, 1, s);
        EXPECT_NEAR(holdout[static_cast<size_t>(s - K - 1)], lb, 1e-4f)
            << "resumed deep loss at step " << s;
    }
}

// A checkpoint is geometry-bound: loading a mismatched dims, mismatched kind,
// corrupt magic or truncated stream must throw, never mis-restore.
TEST_F(CheckpointTest, RejectsGeometryMismatchAndCorruption) {
    const int h = 32, inter = 64;
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    MicroTrainer a(uninitialized_ctx(), h, inter, cfg);
    std::vector<float> Wg(static_cast<size_t>(h) * inter);
    std::vector<float> Wu(static_cast<size_t>(h) * inter);
    std::vector<float> Wd(static_cast<size_t>(inter) * h);
    fill_random(Wg.data(), Wg.size(), 701);
    fill_random(Wu.data(), Wu.size(), 702);
    fill_random(Wd.data(), Wd.size(), 703);
    a.set_weight(Wg.data(), Wu.data(), Wd.data());
    std::stringstream ss(std::ios::out | std::ios::binary);
    a.save_state(ss);
    std::string data = ss.str();

    // (a) dimension mismatch -> reject.
    {
        std::stringstream in(data);
        MicroTrainer wrong(uninitialized_ctx(), h / 2, inter, cfg);
        EXPECT_THROW(wrong.load_state(in), std::runtime_error);
    }
    // (b) kind mismatch: a MicroTrainer checkpoint is not a TransformerTrainer.
    {
        std::stringstream in(data);
        TransformerTrainer t(uninitialized_ctx(), 1, h, 1, 8, inter, cfg);
        EXPECT_THROW(t.load_state(in), std::runtime_error);
    }
    // (c) corrupt magic -> reject.
    {
        std::stringstream bad;
        bad.write("NOT A CKPT", 10);
        MicroTrainer c(uninitialized_ctx(), h, inter, cfg);
        EXPECT_THROW(c.load_state(bad), std::runtime_error);
    }
    // (d) truncated stream -> reject (EOF must surface as an error, not UB).
    {
        std::stringstream trunc(data.substr(0, data.size() / 2));
        MicroTrainer d(uninitialized_ctx(), h, inter, cfg);
        EXPECT_THROW(d.load_state(trunc), std::runtime_error);
    }
}
