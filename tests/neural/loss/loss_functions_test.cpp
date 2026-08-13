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

}  // namespace cuda::neural::loss::test
