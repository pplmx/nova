#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/loss/loss_functions.h>
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <algorithm>
#include <random>
#include <vector>

namespace cuda::neural::loss::test {

class LossFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        cudaStreamDestroy(stream_);
    }

    int device_ = 0;
    cudaStream_t stream_ = nullptr;
};

TEST_F(LossFunctionsTest, CrossEntropyLoss) {
    CrossEntropyConfig config;
    config.num_classes = 10;
    config.reduction_mean = true;

    int batch_size = 4;
    int num_classes = 10;

    std::vector<float> predictions(batch_size * num_classes);
    std::vector<int> targets(batch_size);
    std::vector<float> output(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
            predictions[b * num_classes + c] = (c == b % num_classes) ? 2.0f : 0.0f;
        }
        targets[b] = b % num_classes;
    }

    float loss = cross_entropy_loss(
        predictions.data(), targets.data(), output.data(),
        batch_size, num_classes, config, stream_
    );

    cudaStreamSynchronize(stream_);
    EXPECT_GT(loss, 0.0f);
    EXPECT_LT(loss, 10.0f);
}

TEST_F(LossFunctionsTest, FocalLoss) {
    FocalLossConfig config;
    config.num_classes = 5;
    config.alpha = 1.0f;
    config.gamma = 2.0f;

    int batch_size = 4;
    int num_classes = 5;

    std::vector<float> predictions(batch_size * num_classes);
    std::vector<int> targets(batch_size);
    std::vector<float> output(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_classes; ++c) {
            predictions[b * num_classes + c] = 1.0f / num_classes;
        }
        targets[b] = 0;
    }

    float loss = focal_loss(
        predictions.data(), targets.data(), output.data(),
        batch_size, num_classes, config, stream_
    );

    cudaStreamSynchronize(stream_);
    EXPECT_GT(loss, 0.0f);
}

// Regression for the focal_loss softmax denominator bug (TASK-077, the same
// class cross_entropy_loss was already fixed for): the old code divided by the
// running partial sum, so probs[class0] was always 1.0 and only the last class
// saw the correct denominator. Non-uniform logits + host fp64 reference of the
// two-pass softmax must match element-for-element — RED against the broken
// implementation.
TEST_F(LossFunctionsTest, FocalLoss_SoftmaxReferenceParity) {
    FocalLossConfig config;
    config.num_classes = 5;
    config.alpha = 1.0f;
    config.gamma = 2.0f;

    int batch_size = 4;
    int num_classes = 5;

    // Distinct logits so the wrong partial-sum denominator is observable: for
    // row 0 the broken code would give probs[class0]==1.0 (loss ~0) regardless.
    std::vector<float> preds = {
        2.0f, 1.0f, 0.5f, -0.5f, -2.0f,
        0.1f, 3.0f, 0.2f, -1.0f, 0.8f,
        -3.0f, -1.5f, 2.5f, 1.2f, 0.0f,
        0.4f, 0.4f, 0.4f, 0.4f, 0.4f,
    };
    std::vector<int> targets = {0, 2, 3, 1};
    std::vector<float> output(batch_size);

    float mean = focal_loss(
        preds.data(), targets.data(), output.data(),
        batch_size, num_classes, config, stream_);
    (void)cudaGetLastError();  // focal_loss is host-side; drain sticky errors

    constexpr double eps = 1e-8;
    double ref_total = 0.0;
    for (int b = 0; b < batch_size; ++b) {
        const float* row = preds.data() + static_cast<size_t>(b) * num_classes;
        double max_logit = -INFINITY;
        for (int c = 0; c < num_classes; ++c) {
            max_logit = std::max(max_logit, static_cast<double>(row[c]));
        }
        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += std::exp(static_cast<double>(row[c]) - max_logit);
        }
        const int t = targets[b];
        double p = std::exp(static_cast<double>(row[t]) - max_logit) / sum_exp;
        p = std::max(std::min(p, 1.0 - eps), eps);
        double ref = -config.alpha * std::pow(1.0 - p, config.gamma) * std::log(p);
        SCOPED_TRACE("batch " + std::to_string(b));
        EXPECT_NEAR(output[b], static_cast<float>(ref), 1e-3f)
            << "per-element focal loss must match the two-pass softmax reference";
        ref_total += ref;
    }
    EXPECT_NEAR(mean, static_cast<float>(ref_total / batch_size), 1e-3f)
        << "mean focal loss must match the reference mean";
}

TEST_F(LossFunctionsTest, ContrastiveLoss) {
    ContrastiveLossConfig config;
    config.temperature = 0.1f;
    config.embedding_dim = 64;
    config.normalize_embeddings = true;

    int batch_size = 4;
    int embedding_dim = 64;

    std::vector<float> embeddings1(batch_size * embedding_dim, 0.1f);
    std::vector<float> embeddings2(batch_size * embedding_dim, 0.1f);
    std::vector<float> output(batch_size);

    float loss = contrastive_loss(
        embeddings1.data(), embeddings2.data(), output.data(),
        batch_size, embedding_dim, config, stream_
    );

    cudaStreamSynchronize(stream_);
    EXPECT_GE(loss, 0.0f);
}

TEST_F(LossFunctionsTest, CrossEntropyLossFunction) {
    CrossEntropyConfig config;
    config.num_classes = 3;

    CrossEntropyLossFunction loss_fn(config);
    EXPECT_EQ(loss_fn.get_type(), LossType::CrossEntropy);
    EXPECT_EQ(loss_fn.get_name(), "CrossEntropy");
}

TEST_F(LossFunctionsTest, FocalLossFunction) {
    FocalLossConfig config;
    config.num_classes = 3;

    FocalLossFunction loss_fn(config);
    EXPECT_EQ(loss_fn.get_type(), LossType::FocalLoss);
    EXPECT_EQ(loss_fn.get_name(), "FocalLoss");
}

TEST_F(LossFunctionsTest, ContrastiveLossFunction) {
    ContrastiveLossConfig config;
    config.embedding_dim = 64;

    ContrastiveLossFunction loss_fn(config);
    EXPECT_EQ(loss_fn.get_type(), LossType::ContrastiveLoss);
    EXPECT_EQ(loss_fn.get_name(), "ContrastiveLoss");
}

TEST_F(LossFunctionsTest, LossTypeEnum) {
    EXPECT_EQ(static_cast<int>(LossType::CrossEntropy), 0);
    EXPECT_EQ(static_cast<int>(LossType::FocalLoss), 1);
    EXPECT_EQ(static_cast<int>(LossType::ContrastiveLoss), 2);
}

TEST_F(LossFunctionsTest, CrossEntropyConfig) {
    CrossEntropyConfig config;
    config.num_classes = 10;
    config.label_smoothing = 0.1f;
    config.reduction_mean = true;

    EXPECT_EQ(config.num_classes, 10);
    EXPECT_EQ(config.label_smoothing, 0.1f);
    EXPECT_TRUE(config.reduction_mean);
}

// ============================================================================
// Milestone v2.24 / TASK-024 RED: device cross-entropy-with-logits backward.
// The host cross_entropy_loss() computes only the scalar (its device kernels
// are dead code), so no grad-logits exist to seed a layer backward chain. This
// test pins the analytic contract grad_logits[b*C+c] = (softmax_c - onehot)/B
// (reduction_mean) against a host double-precision reference. RED against the
// provisional throw-stub; GREEN with the P2 device kernel.
// ============================================================================
TEST_F(LossFunctionsTest, CrossEntropyLogitsBackwardDevice) {
    CrossEntropyConfig config;
    config.num_classes = 10;
    config.reduction_mean = true;

    const int batch = 6;
    const int C = 10;
    std::vector<float> logits(static_cast<size_t>(batch) * C);
    std::vector<int> targets(batch);
    std::mt19937 rng(20260813);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = dist(rng);
    }
    for (int b = 0; b < batch; ++b) {
        targets[b] = static_cast<int>(rng() % C);
    }

    // Host double-precision analytic grad: softmax(logits) - onehot, /batch.
    std::vector<float> ref(static_cast<size_t>(batch) * C);
    for (int b = 0; b < batch; ++b) {
        double m = logits[b * C];
        for (int c = 1; c < C; ++c) {
            m = std::max(m, static_cast<double>(logits[b * C + c]));
        }
        double sum = 0.0;
        for (int c = 0; c < C; ++c) {
            sum += std::exp(static_cast<double>(logits[b * C + c]) - m);
        }
        for (int c = 0; c < C; ++c) {
            double sm = std::exp(static_cast<double>(logits[b * C + c]) - m) / sum;
            double grad = sm - (c == targets[b] ? 1.0 : 0.0);
            ref[b * C + c] = static_cast<float>(grad / batch);
        }
    }

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_grad(logits.size());
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    std::vector<float> grad(logits.size(), 12345.0f);
    EXPECT_NO_THROW(
        cross_entropy_logits_backward(
            d_logits.data(), d_targets.data(), d_grad.data(),
            batch, C, config, stream_));
    d_grad.copy_to(grad.data(), grad.size());

    for (size_t i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(ref[i], grad[i], 1e-5f)
            << "grad_logits mismatch at " << i;
    }
}

// ============================================================================
// Milestone v2.30 (DEC-016 / TASK-048) — device cross-entropy-loss scalar.
// cross_entropy_loss_device computes the identical mean CE loss as the host
// cross_entropy_loss but from device logits/targets, writing a single scalar
// to [1] so the training drivers no longer round-trip the full m*C logits
// buffer through the host every step (issue-v30-ce-loss-host-roundtrip).
// Pinned against a host double-precision reference. RED against the
// provisional throw-stub; GREEN with the P2 device reduction kernel.
// ============================================================================
namespace {

// Host double-precision reference for the mean CE loss (max-subtracted
// softmax, target log-probability), matching the host cross_entropy_loss()
// math exactly (the same full-class-sum ordering; no label smoothing).
double host_ce_loss_reference(const std::vector<float>& logits,
                              const std::vector<int>& targets, int batch,
                              int C, bool reduction_mean) {
    double total = 0.0;
    for (int b = 0; b < batch; ++b) {
        double m = logits[static_cast<size_t>(b) * C];
        for (int c = 1; c < C; ++c) {
            m = std::max(m, static_cast<double>(logits[static_cast<size_t>(b) * C + c]));
        }
        double sum_exp = 0.0;
        for (int c = 0; c < C; ++c) {
            sum_exp += std::exp(static_cast<double>(
                logits[static_cast<size_t>(b) * C + c]) - m);
        }
        const double log_sum_exp = std::log(sum_exp);
        const int t = targets[b];
        const double log_prob =
            static_cast<double>(logits[static_cast<size_t>(b) * C + t]) - m -
            log_sum_exp;
        total += -log_prob;
    }
    return reduction_mean ? total / batch : total;
}

}  // namespace

// Batch>1, classes>1; mean reduction. The primary contract for the training
// drivers (MicroTrainer/TransformerTrainer evaluate mean CE every step).
TEST_F(LossFunctionsTest, DeviceCrossEntropyLossMatchesFp64Reference) {
    CrossEntropyConfig config;
    config.num_classes = 10;
    config.reduction_mean = true;

    const int batch = 6;
    const int C = 10;
    std::vector<float> logits(static_cast<size_t>(batch) * C);
    std::vector<int> targets(batch);
    std::mt19937 rng(20260823);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = dist(rng);
    }
    for (int b = 0; b < batch; ++b) {
        targets[b] = static_cast<int>(rng() % C);
    }
    const double expected = host_ce_loss_reference(logits, targets, batch, C, true);

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_loss(1);
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    EXPECT_NO_THROW(
        cross_entropy_loss_device(
            d_logits.data(), d_targets.data(), d_loss.data(),
            batch, C, config, stream_));
    float dev = 0.0f;
    d_loss.copy_to(&dev, 1);

    // fp32 reduction-order drift vs the fp64 reference (same convention as
    // the existing kernel-parity tests: 1e-4 absorbs block-sum reordering).
    EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
}

// Single-row edge (batch == 1) — the degenerate case the trainers can hit at
// batch*seq == 1; must still produce the exact single-row loss.
TEST_F(LossFunctionsTest, DeviceCrossEntropyLossSingleRow) {
    CrossEntropyConfig config;
    config.num_classes = 8;
    config.reduction_mean = true;

    const int batch = 1;
    const int C = 8;
    std::vector<float> logits{3.0f, -1.0f, 0.5f, 2.2f, -4.0f, 1.1f, 0.0f, 0.7f};
    std::vector<int> targets{2};
    const double expected = host_ce_loss_reference(logits, targets, batch, C, true);

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_loss(1);
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    EXPECT_NO_THROW(
        cross_entropy_loss_device(
            d_logits.data(), d_targets.data(), d_loss.data(),
            batch, C, config, stream_));
    float dev = 0.0f;
    d_loss.copy_to(&dev, 1);

    EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
}

// Sum reduction (reduction_mean == false) must return the raw summed loss,
// matching the host cross_entropy_loss convention.
TEST_F(LossFunctionsTest, DeviceCrossEntropyLossSumReduction) {
    CrossEntropyConfig config;
    config.num_classes = 6;
    config.reduction_mean = false;

    const int batch = 3;
    const int C = 6;
    std::vector<float> logits(static_cast<size_t>(batch) * C);
    std::vector<int> targets(batch);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = dist(rng);
    }
    for (int b = 0; b < batch; ++b) {
        targets[b] = static_cast<int>(rng() % C);
    }
    const double expected = host_ce_loss_reference(logits, targets, batch, C, false);

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_loss(1);
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    EXPECT_NO_THROW(
        cross_entropy_loss_device(
            d_logits.data(), d_targets.data(), d_loss.data(),
            batch, C, config, stream_));
    float dev = 0.0f;
    d_loss.copy_to(&dev, 1);

    EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
}

// accuracy_device must count rows whose argmax class == target, matching the
// host top-1 loop the training drivers previously ran on copied-back logits.
// Includes a tie (row with two equal max logits): host and device argmax both
// pick the earliest max, so the count must agree exactly.
TEST_F(LossFunctionsTest, DeviceAccuracyMatchesHostArgmax) {
    const int batch = 5;
    const int C = 4;
    // Row 0: target 1 == argmax(3.0 at class 1)  -> correct
    // Row 1: target 2, argmax class 3            -> wrong
    // Row 2: tie between class 0 and class 1 (both 2.0); earliest is class 0;
    //        target 0 -> correct
    // Row 3: all equal; earliest max is class 0; target 0 -> correct
    // Row 4: target 3 == argmax(9.0 at class 3)  -> correct
    std::vector<float> logits{0.0f, 3.0f, 0.0f, 0.5f,
                              1.0f, 0.0f, 0.0f, 4.0f,
                              2.0f, 2.0f, 0.0f, 0.0f,
                              1.0f, 1.0f, 1.0f, 1.0f,
                              -1.0f, 0.0f, 0.5f, 9.0f};
    std::vector<int> targets{1, 2, 0, 0, 3};

    int host_correct = 0;
    for (int b = 0; b < batch; ++b) {
        const float* row = logits.data() + static_cast<size_t>(b) * C;
        int argmax = 0;
        for (int c = 1; c < C; ++c) {
            if (row[c] > row[argmax]) argmax = c;
        }
        if (argmax == targets[b]) ++host_correct;
    }

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_correct(1);
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    EXPECT_NO_THROW(
        accuracy_device(d_logits.data(), d_targets.data(), d_correct.data(),
                        batch, C, stream_));
    float dev_correct = -1.0f;
    d_correct.copy_to(&dev_correct, 1);

    EXPECT_EQ(static_cast<float>(host_correct), dev_correct);
}

// C > 256: with more classes than the block size, every thread holds a
// nonzero partial in the grid-stride max/exp-sum loops, so the block
// reductions actually combine multiple partials (the C<=10 tests above are
// all single-thread-row partials). Locks the multi-iteration path.
TEST_F(LossFunctionsTest, DeviceCrossEntropyLossLargeClassCount) {
    CrossEntropyConfig config;
    config.num_classes = 1024;
    config.reduction_mean = true;

    const int batch = 3;
    const int C = 1024;
    std::vector<float> logits(static_cast<size_t>(batch) * C);
    std::vector<int> targets(batch);
    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = dist(rng);
    }
    for (int b = 0; b < batch; ++b) {
        targets[b] = static_cast<int>(rng() % C);
    }
    const double expected = host_ce_loss_reference(logits, targets, batch, C, true);

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_loss(1);
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    EXPECT_NO_THROW(
        cross_entropy_loss_device(
            d_logits.data(), d_targets.data(), d_loss.data(),
            batch, C, config, stream_));
    float dev = 0.0f;
    d_loss.copy_to(&dev, 1);

    EXPECT_NEAR(static_cast<float>(expected), dev, 1e-4f);
}

// Non-uniform logits: a target pinned to the max-logit class must give a
// small loss (< 1), a target on the min-logit class a large loss (> 2) —
// sanity that the device path computes log-probabilities, not something
// else, before the strict fp64 parity above.
TEST_F(LossFunctionsTest, DeviceCrossEntropyLossDistinguishesTargets) {
    CrossEntropyConfig config;
    config.num_classes = 3;
    config.reduction_mean = true;

    const int batch = 2;
    const int C = 3;
    // Row 0: target 0 is the max class. Row 1: target 1 is the min class.
    std::vector<float> logits{10.0f, 0.0f, 0.0f, 0.0f, -10.0f, 0.0f};
    std::vector<int> targets{0, 1};

    cuda::memory::Buffer<float> d_logits(logits.size());
    cuda::memory::Buffer<int> d_targets(targets.size());
    cuda::memory::Buffer<float> d_loss(1);
    d_logits.copy_from(logits.data(), logits.size());
    d_targets.copy_from(targets.data(), targets.size());

    EXPECT_NO_THROW(
        cross_entropy_loss_device(
            d_logits.data(), d_targets.data(), d_loss.data(),
            batch, C, config, stream_));
    float dev = 0.0f;
    d_loss.copy_to(&dev, 1);

    // mean of (loss~1e-10 for row0, loss~10 for row1) ≈ 5.
    EXPECT_GT(dev, 1.0f);
    EXPECT_LT(dev, 9.0f);
}

}  // namespace cuda::neural::loss::test
