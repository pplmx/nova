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

/**
 * @brief Device cross-entropy-with-logits backward (milestone v2.24)
 *
 * The missing seed gradient of the training stack: the host-side
 * cross_entropy_loss() computes only the scalar (its device kernels are
 * unused / dead code), so no grad-logits exist to feed a layer backward
 * chain. This kernel computes the standard softmax-CE gradient
 *
 *   grad_logits[b * C + c] = (softmax_c(logits_b) - [c == targets[b]]) / B
 *
 * for reduction_mean (or without the /B when sum reduction), where softmax is
 * computed with max-subtraction for numerical stability. It is the layer
 * backward chain's upstream seed in an end-to-end training step (TASK-025).
 *
 * @param logits Device logits [batch_size x num_classes]
 * @param targets Device targets [batch_size]
 * @param grad_logits Output device buffer [batch_size x num_classes]
 * @param batch_size Batch size
 * @param num_classes Number of classes
 * @param config Cross-entropy config (reduction must match the forward pass)
 * @param stream CUDA stream (default null = current stream)
 */
void cross_entropy_logits_backward(
    const float* logits,
    const int* targets,
    float* grad_logits,
    int batch_size,
    int num_classes,
    const CrossEntropyConfig& config = {},
    cudaStream_t stream = nullptr
);

/**
 * @brief Device cross-entropy-with-logits scalar (milestone v2.30 / DEC-016)
 *
 * The host cross_entropy_loss() computes the mean-loss scalar from HOST
 * predictions, so the training drivers (MicroTrainer, TransformerTrainer)
 * round-trip the whole m x hidden logits buffer back to the host every
 * train_step just to read this one number — the same D2H family v2.27
 * (optimizer) and v2.29 (SDPA/LN) removed. This entry computes the identical
 * mean CE loss on-device (per-row max-subtracted softmax -> target
 * log-probability -> block/warp reduction to a single scalar) and writes it
 * into `loss_out` (one float, [1]); the caller copies just that scalar back.
 *
 * @param logits Device logits [batch_size x num_classes]
 * @param targets Device targets [batch_size]
 * @param loss_out Device scalar output [1]
 * @param batch_size Batch size
 * @param num_classes Number of classes
 * @param config Cross-entropy config (reduction must match the host forward)
 * @param stream CUDA stream (default null = current stream)
 */
void cross_entropy_loss_device(
    const float* logits,
    const int* targets,
    float* loss_out,
    int batch_size,
    int num_classes,
    const CrossEntropyConfig& config = {},
    cudaStream_t stream = nullptr
);

/**
 * @brief Device top-1 accuracy (milestone v2.30 / DEC-016)
 *
 * Companion to cross_entropy_loss_device(): the training drivers' evaluate()
 * also round-trip the whole logits buffer to compute a host argmax loop. This
 * entry counts rows whose argmax class equals the target on-device and writes
 * the count into `correct_out` (one float, [1]) — the caller divides by
 * batch_size. Used with cross_entropy_loss_device so MicroTrainer /
 * TransformerTrainer stop copying logits across the host entirely.
 *
 * The count is stored as a float so it shares the trainers' float scalar
 * buffer; every increment is exactly +1.0f and the running sum is always an
 * integer, so the result is exact for any batch < 2^24 (~16.7M rows).
 *
 * @param logits Device logits [batch_size x num_classes]
 * @param targets Device targets [batch_size]
 * @param correct_out Device scalar count of correct rows [1]
 * @param batch_size Batch size
 * @param num_classes Number of classes
 * @param stream CUDA stream (default null = current stream)
 */
void accuracy_device(
    const float* logits,
    const int* targets,
    float* correct_out,
    int batch_size,
    int num_classes,
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
