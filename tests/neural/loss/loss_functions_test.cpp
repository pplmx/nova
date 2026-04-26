#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/loss/loss_functions.h>

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

}  // namespace cuda::neural::loss::test
