#include "cuda/neural/loss/loss_functions.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace cuda::neural::loss {

namespace {

template <typename T>
__global__ void cross_entropy_logits_kernel(
    const T* logits, T* softmax_out, int batch_size, int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes) return;

    int batch = idx / num_classes;
    int class_idx = idx % num_classes;

    T max_logit = -INFINITY;
    for (int c = 0; c < num_classes; ++c) {
        max_logit = max(max_logit, logits[batch * num_classes + c]);
    }

    T sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        sum_exp += exp(logits[batch * num_classes + c] - max_logit);
    }

    softmax_out[idx] = exp(logits[idx] - max_logit) / sum_exp;
}

template <typename T>
__global__ void focal_loss_gradient_kernel(
    const T* probs, const int* targets, T* output,
    int batch_size, int num_classes, T gamma, T alpha, T epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int target_class = targets[idx];
    T p = probs[idx * num_classes + target_class];
    p = max(min(p, static_cast<T>(1.0f) - epsilon), epsilon);

    T focal_weight = powf(1.0f - p, gamma);
    output[idx] = -alpha * focal_weight * logf(p);
}

template <typename T>
__global__ void contrastive_cosine_kernel(
    const T* emb1, const T* emb2, T* similarities,
    int batch_size, int embedding_dim, T temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    T dot = 0.0f;
    T norm1 = 0.0f;
    T norm2 = 0.0f;

    for (int i = 0; i < embedding_dim; ++i) {
        T a = emb1[idx * embedding_dim + i];
        T b = emb2[idx * embedding_dim + i];
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    norm1 = sqrtf(norm1);
    norm2 = sqrtf(norm2);

    similarities[idx] = (norm1 > 0.0f && norm2 > 0.0f)
        ? expf(dot / (norm1 * norm2) / temperature)
        : 0.0f;
}

}  // namespace

float cross_entropy_loss(
    const float* predictions,
    const int* targets,
    float* output,
    int batch_size,
    int num_classes,
    const CrossEntropyConfig& config,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (batch_size * num_classes + block_size - 1) / block_size;

    std::vector<float> log_probs(batch_size * num_classes);
    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        float max_logit = -INFINITY;
        for (int c = 0; c < num_classes; ++c) {
            max_logit = std::max(max_logit, predictions[b * num_classes + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float logit = predictions[b * num_classes + c] - max_logit;
            float exp_logit = expf(logit);
            sum_exp += exp_logit;
            log_probs[b * num_classes + c] = logit - logf(sum_exp);
        }

        int target_class = targets[b];
        float loss = -log_probs[b * num_classes + target_class];

        if (config.reduction_mean) {
            output[b] = loss / static_cast<float>(batch_size);
        } else {
            output[b] = loss;
        }

        total_loss += loss;
    }

    return config.reduction_mean ? total_loss / batch_size : total_loss;
}

float focal_loss(
    const float* predictions,
    const int* targets,
    float* output,
    int batch_size,
    int num_classes,
    const FocalLossConfig& config,
    cudaStream_t stream
) {
    std::vector<float> probs(batch_size * num_classes);

    for (int b = 0; b < batch_size; ++b) {
        float max_logit = -INFINITY;
        for (int c = 0; c < num_classes; ++c) {
            max_logit = std::max(max_logit, predictions[b * num_classes + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float logit = predictions[b * num_classes + c] - max_logit;
            float exp_logit = expf(logit);
            sum_exp += exp_logit;
            probs[b * num_classes + c] = exp_logit / sum_exp;
        }
    }

    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        int target_class = targets[b];
        float p = probs[b * num_classes + target_class];
        p = std::max(std::min(p, 1.0f - config.epsilon), config.epsilon);

        float focal_weight = powf(1.0f - p, config.gamma);
        float loss = -config.alpha * focal_weight * logf(p);

        output[b] = loss;
        total_loss += loss;
    }

    return total_loss / batch_size;
}

float contrastive_loss(
    const float* embeddings1,
    const float* embeddings2,
    float* output,
    int batch_size,
    int embedding_dim,
    const ContrastiveLossConfig& config,
    cudaStream_t stream
) {
    std::vector<float> similarities(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;

        for (int i = 0; i < embedding_dim; ++i) {
            float a = embeddings1[b * embedding_dim + i];
            float b_emb = embeddings2[b * embedding_dim + i];
            dot += a * b_emb;
            norm1 += a * a;
            norm2 += b_emb * b_emb;
        }

        norm1 = sqrtf(norm1);
        norm2 = sqrtf(norm2);

        float cos_sim = (norm1 > 0.0f && norm2 > 0.0f)
            ? dot / (norm1 * norm2)
            : 0.0f;
        similarities[b] = cos_sim / config.temperature;
    }

    float sum_exp_pos = 0.0f;
    float sum_exp_neg = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        sum_exp_pos += expf(similarities[b]);

        float neg_sim = 0.0f;
        for (int j = 0; j < batch_size; ++j) {
            if (b != j) {
                float dot = 0.0f;
                float norm1 = 0.0f;
                float norm2 = 0.0f;

                for (int i = 0; i < embedding_dim; ++i) {
                    float a = embeddings1[b * embedding_dim + i];
                    float b_emb = embeddings1[j * embedding_dim + i];
                    dot += a * b_emb;
                    norm1 += a * a;
                    norm2 += b_emb * b_emb;
                }

                norm1 = sqrtf(norm1);
                norm2 = sqrtf(norm2);

                float cos_sim = (norm1 > 0.0f && norm2 > 0.0f)
                    ? dot / (norm1 * norm2)
                    : 0.0f;
                neg_sim += expf(cos_sim / config.temperature);
            }
        }
        sum_exp_neg += neg_sim;
    }

    float loss = -logf(sum_exp_pos / (sum_exp_pos + sum_exp_neg));

    for (int b = 0; b < batch_size; ++b) {
        output[b] = loss;
    }

    return loss;
}

CrossEntropyLossFunction::CrossEntropyLossFunction(const CrossEntropyConfig& config)
    : config_(config) {}

float CrossEntropyLossFunction::forward(
    const float* predictions,
    const void* targets,
    float* output,
    int batch_size,
    cudaStream_t stream
) {
    return cross_entropy_loss(
        predictions,
        static_cast<const int*>(targets),
        output,
        batch_size,
        config_.num_classes,
        config_,
        stream
    );
}

FocalLossFunction::FocalLossFunction(const FocalLossConfig& config)
    : config_(config) {}

float FocalLossFunction::forward(
    const float* predictions,
    const void* targets,
    float* output,
    int batch_size,
    cudaStream_t stream
) {
    return focal_loss(
        predictions,
        static_cast<const int*>(targets),
        output,
        batch_size,
        config_.num_classes,
        config_,
        stream
    );
}

ContrastiveLossFunction::ContrastiveLossFunction(const ContrastiveLossConfig& config)
    : config_(config) {}

float ContrastiveLossFunction::forward(
    const float* predictions,
    const void* targets,
    float* output,
    int batch_size,
    cudaStream_t stream
) {
    return contrastive_loss(
        predictions,
        static_cast<const float*>(targets),
        output,
        batch_size,
        config_.embedding_dim,
        config_,
        stream
    );
}

}  // namespace cuda::neural::loss
