/**
 * @file micro_trainer_test.cpp
 * @brief Single-GPU (tp == 1) tests for the MicroTrainer training driver
 *
 * Milestone v2.25 (TASK-028 / DEC-011): v2.24 proved the layers take a single
 * exact training step (weight parity vs host fp64), but the stack had never
 * been shown to actually learn. These tests pin the MicroTrainer contracts on
 * the tp <= 1 path with an uninitialized NcclContext (0 devices -> effective
 * tp degree 1, no NCCL): train_step returns a mean CE loss that descends over
 * steps, and accuracy (top-1 argmax vs labels) rises toward the learnable-task
 * ceiling. The multi-GPU full-run trajectory parity lives in
 * micro_trainer_multigpu_test.cpp.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/training.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::neural::training;
using namespace cuda::neural::optimizers;

namespace {

cuda::nccl::NcclContext& uninitialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

}  // namespace

class MicroTrainerSingleGpuTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        cudaFree(nullptr);  // Ensure a CUDA context for device-dependent calls
    }
};

// A learnable synthetic classification task: a fixed random low-rank map from
// the input to class logits. hidden == num_classes == 8, so the MLP's output
// [m x hidden] is treated as logits over 8 classes.
namespace {

// Generate the dataset: gaussian inputs + a random full-rank label matrix
// (the "teacher"). The 2-layer siLU MLP must learn to invert it.
void make_task(std::vector<float>& X, std::vector<int>& Y, int m, int h,
               unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    X.resize(static_cast<size_t>(m) * h);
    Y.resize(m);
    std::vector<float> T(static_cast<size_t>(h) * h);
    for (size_t i = 0; i < T.size(); ++i) {
        T[i] = norm(rng);
    }
    for (int i = 0; i < m; ++i) {
        for (int d = 0; d < h; ++d) {
            X[i * h + d] = norm(rng);
        }
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

void fill_random(float* data, size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

}  // namespace

// The MicroTrainer must drive the loss down: after a fixed number of AdamW
// steps on a learnable task, the mean CE loss decreases meaningfully and the
// top-1 accuracy rises above chance (1/8).
TEST_F(MicroTrainerSingleGpuTest, LossDescentsAndAccuracyRises) {
    const int h = 8;
    const int m = 32;
    std::vector<float> X;
    std::vector<int> Y;
    make_task(X, Y, m, h, 20260813);

    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
    MicroTrainer trainer(uninitialized_ctx(), h, 16, cfg);

    // Random init weights (deterministic seed).
    std::vector<float> Wg(static_cast<size_t>(h) * 16);
    std::vector<float> Wu(static_cast<size_t>(h) * 16);
    std::vector<float> Wd(static_cast<size_t>(16) * h);
    fill_random(Wg.data(), Wg.size(), 7);
    fill_random(Wu.data(), Wu.size(), 8);
    fill_random(Wd.data(), Wd.size(), 9);
    trainer.set_weight(Wg.data(), Wu.data(), Wd.data());

    cuda::memory::Buffer<float> d_X(m * h);

    float loss_0 = 0.0f, acc_0 = 0.0f, loss_k = 0.0f, acc_k = 0.0f;
    d_X.copy_from(X.data(), X.size());
    loss_0 = trainer.evaluate(d_X.data(), Y.data(), m, 1, &acc_0);
    for (int s = 1; s <= 60; ++s) {
        trainer.train_step(d_X.data(), Y.data(), m, 1, s);
    }
    loss_k = trainer.evaluate(d_X.data(), Y.data(), m, 1, &acc_k);

    // Chance accuracy for 8 classes is 1/8 = 0.125.
    EXPECT_LT(loss_k, loss_0) << "loss must descend over training";
    EXPECT_GT(acc_k, 0.5f) << "accuracy must rise well above chance (1/8)";
    EXPECT_GT(acc_k, acc_0 + 0.1f);
}

// The returned train_step loss must be finite and sane; train_step must also
// advance the weights (a second train_step changes the evaluation loss).
TEST_F(MicroTrainerSingleGpuTest, TrainStepReturnsFiniteLossAndAdvancesWeights) {
    const int h = 8;
    const int m = 16;
    std::vector<float> X;
    std::vector<int> Y;
    make_task(X, Y, m, h, 20260814);

    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
    MicroTrainer trainer(uninitialized_ctx(), h, 16, cfg);
    std::vector<float> Wg(static_cast<size_t>(h) * 16);
    std::vector<float> Wu(static_cast<size_t>(h) * 16);
    std::vector<float> Wd(static_cast<size_t>(16) * h);
    fill_random(Wg.data(), Wg.size(), 11);
    fill_random(Wu.data(), Wu.size(), 12);
    fill_random(Wd.data(), Wd.size(), 13);
    trainer.set_weight(Wg.data(), Wu.data(), Wd.data());

    cuda::memory::Buffer<float> d_X(m * h);
    d_X.copy_from(X.data(), X.size());

    float l1 = trainer.train_step(d_X.data(), Y.data(), m, 1, 1);
    EXPECT_TRUE(std::isfinite(l1));
    EXPECT_GT(l1, 0.0f);
    float l2 = trainer.train_step(d_X.data(), Y.data(), m, 1, 2);
    EXPECT_TRUE(std::isfinite(l2));
    EXPECT_NE(l1, l2) << "the optimizer must move the weights between steps";

    // Accessors.
    EXPECT_EQ(trainer.hidden_dim(), h);
    EXPECT_EQ(trainer.intermediate_size(), 16);
    EXPECT_EQ(trainer.tp_degree(), 1);
}
