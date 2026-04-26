#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

namespace cuda::neural::loss {

enum class LossType {
    CrossEntropy,
    FocalLoss,
    ContrastiveLoss
};

struct CrossEntropyConfig {
    int num_classes = 10;
    float label_smoothing = 0.0f;
    bool reduction_mean = true;
    float epsilon = 1e-7f;
};

float cross_entropy_loss(
    const float* predictions,
    const int* targets,
    float* output,
    int batch_size,
    int num_classes,
    const CrossEntropyConfig& config = {},
    cudaStream_t stream = nullptr
);

struct FocalLossConfig {
    int num_classes = 10;
    float alpha = 1.0f;
    float gamma = 2.0f;
    float epsilon = 1e-7f;
};

float focal_loss(
    const float* predictions,
    const int* targets,
    float* output,
    int batch_size,
    int num_classes,
    const FocalLossConfig& config = {},
    cudaStream_t stream = nullptr
);

struct ContrastiveLossConfig {
    float temperature = 0.07f;
    int embedding_dim = 128;
    bool normalize_embeddings = true;
    bool use_cosine_similarity = true;
};

float contrastive_loss(
    const float* embeddings1,
    const float* embeddings2,
    float* output,
    int batch_size,
    int embedding_dim,
    const ContrastiveLossConfig& config = {},
    cudaStream_t stream = nullptr
);

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual float forward(
        const float* predictions,
        const void* targets,
        float* output,
        int batch_size,
        cudaStream_t stream = nullptr
    ) = 0;
    virtual LossType get_type() const = 0;
    virtual std::string get_name() const = 0;
};

class CrossEntropyLossFunction : public LossFunction {
public:
    explicit CrossEntropyLossFunction(const CrossEntropyConfig& config);

    float forward(
        const float* predictions,
        const void* targets,
        float* output,
        int batch_size,
        cudaStream_t stream = nullptr
    ) override;

    LossType get_type() const override { return LossType::CrossEntropy; }
    std::string get_name() const override { return "CrossEntropy"; }

private:
    CrossEntropyConfig config_;
};

class FocalLossFunction : public LossFunction {
public:
    explicit FocalLossFunction(const FocalLossConfig& config);

    float forward(
        const float* predictions,
        const void* targets,
        float* output,
        int batch_size,
        cudaStream_t stream = nullptr
    ) override;

    LossType get_type() const override { return LossType::FocalLoss; }
    std::string get_name() const override { return "FocalLoss"; }

private:
    FocalLossConfig config_;
};

class ContrastiveLossFunction : public LossFunction {
public:
    explicit ContrastiveLossFunction(const ContrastiveLossConfig& config);

    float forward(
        const float* predictions,
        const void* targets,
        float* output,
        int batch_size,
        cudaStream_t stream = nullptr
    ) override;

    LossType get_type() const override { return LossType::ContrastiveLoss; }
    std::string get_name() const override { return "ContrastiveLoss"; }

private:
    ContrastiveLossConfig config_;
};

}  // namespace cuda::neural::loss
