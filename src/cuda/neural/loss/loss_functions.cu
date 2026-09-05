#include "cuda/neural/loss/loss_functions.h"
#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
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

        // The softmax denominator must be the FULL sum over every class.
        // Compute it first, then fill log_probs with it — using a running
        // sum while filling (the pre-fix bug) made the "log-prob" for a target
        // at position c use only the partial sum c' <= c, systematically
        // under-reporting the loss for early targets (found by the v2.25
        // MicroTrainer trajectory parity: device vs host fp64 loss diverged by
        // ~1.7, the exact average of this bias over uniform targets).
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float logit = predictions[b * num_classes + c] - max_logit;
            sum_exp += expf(logit);
        }
        const float log_sum_exp = logf(sum_exp);
        for (int c = 0; c < num_classes; ++c) {
            float logit = predictions[b * num_classes + c] - max_logit;
            log_probs[b * num_classes + c] = logit - log_sum_exp;
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

// Device cross-entropy-with-logits backward (v2.24 / TASK-025):
//   grad_logits[b*C+c] = (softmax_c(logits_b) - [c == targets[b]]) * scale
// with scale = 1/B for reduction_mean (matching the host forward), 1 otherwise.
// One thread per (b, c); the per-row max/sum is recomputed redundantly per
// class (C <= hundreds in practice, so the O(B*C^2) redundant work is cheap
// and keeps the kernel grid-stride-free and dependency-light).
template <typename T>
__global__ void cross_entropy_logits_backward_kernel(
    const T* logits,
    const int* targets,
    T* grad_logits,
    int batch_size,
    int num_classes,
    T scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_classes;
    if (idx >= total) return;

    const int b = idx / num_classes;
    const int c = idx % num_classes;

    T max_logit = -INFINITY;
    for (int j = 0; j < num_classes; ++j) {
        max_logit = max(max_logit, logits[b * num_classes + j]);
    }

    T sum_exp = static_cast<T>(0.0);
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += exp(logits[b * num_classes + j] - max_logit);
    }

    const T softmax_c = exp(logits[idx] - max_logit) / sum_exp;
    const T onehot = (c == targets[b]) ? static_cast<T>(1.0) : static_cast<T>(0.0);
    grad_logits[idx] = (softmax_c - onehot) * scale;
}

void cross_entropy_logits_backward(
    const float* logits,
    const int* targets,
    float* grad_logits,
    int batch_size,
    int num_classes,
    const CrossEntropyConfig& config,
    cudaStream_t stream
) {
    if (batch_size <= 0 || num_classes <= 0) {
        throw std::invalid_argument(
            "cross_entropy_logits_backward requires positive batch_size and "
            "num_classes");
    }
    const float scale =
        config.reduction_mean ? 1.0f / static_cast<float>(batch_size) : 1.0f;
    const int block_size = 256;
    const int total = batch_size * num_classes;
    const int grid_size = (total + block_size - 1) / block_size;
    cross_entropy_logits_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        logits, targets, grad_logits, batch_size, num_classes, scale);
    // async launch errors surface on the next sync; check immediately so a
    // bad launch is reported here, not in a caller's later cudaStreamSync.
    CUDA_CHECK(cudaGetLastError());
}

namespace {

constexpr int kCeLossBlock = 256;  // multiple of 32 (full reduction masks)
constexpr unsigned kCeFull = 0xffffffffu;

// Lane collapse of a per-thread max across the block. `red` must hold at
// least blockDim/32 floats; the reduced value is valid in ALL threads on
// return (kCeLossBlock threads are always active, so no partial-warp UB).
//
// The entry __syncthreads() is load-bearing — same convention as the v2.29
// block reductions (attention_kernels.cu): the helper both returns red[0]
// (read after the final barrier) and writes red[wid] at its start, so a
// subsequent call on the same buffer would race the previous call's readers
// without the entry barrier (compute-sanitizer racecheck-verified there).
__device__ float ce_block_reduce_max(float v, float* red) {
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(kCeFull, v, o));
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float acc = (threadIdx.x < kCeLossBlock / 32) ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) acc = fmaxf(acc, __shfl_down_sync(kCeFull, acc, o));
        if (lane == 0) red[0] = acc;
    }
    __syncthreads();
    return red[0];
}

// Same lane-collapse for a sum. Identity value is 0.0f.
__device__ float ce_block_reduce_sum(float v, float* red) {
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kCeFull, v, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float acc = (threadIdx.x < kCeLossBlock / 32) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(kCeFull, acc, o);
        if (lane == 0) red[0] = acc;
    }
    __syncthreads();
    return red[0];
}

// Mean/sum CE loss over a logits matrix, computed on-device (v2.30 / DEC-016).
// One block per row: block-reduce the row max, then the row exp-sum, then
// thread 0 atomicAdds -log_softmax(target), scaled by 1/batch (mean) or 1
// (sum), into loss_out[0] (the wrapper zeroes it first). This is the exact
// host cross_entropy_loss math, just on device, so the training drivers can
// read ONE scalar back instead of the whole m*C logits buffer. The float
// atomicAdd ordering across blocks only perturbs the final sum at ~ulp level,
// inside the parity tolerance (same convention as v2.29's dV/dK atomics).
__global__ void ce_loss_rows_kernel(const float* logits, const int* targets,
                                    float* out, int batch, int C,
                                    float loss_scale) {
    extern __shared__ float red[];
    const int b = blockIdx.x;
    if (b >= batch) return;
    const float* row = logits + static_cast<size_t>(b) * C;

    float mx = -INFINITY;
    for (int c = threadIdx.x; c < C; c += kCeLossBlock) {
        mx = fmaxf(mx, row[c]);
    }
    mx = ce_block_reduce_max(mx, red);

    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < C; c += kCeLossBlock) {
        sum_exp += __expf(row[c] - mx);
    }
    sum_exp = ce_block_reduce_sum(sum_exp, red);

    if (threadIdx.x == 0) {
        const int t = targets[b];
        const float log_prob = row[t] - mx - logf(sum_exp);
        atomicAdd(&out[0], -log_prob * loss_scale);
    }
}

// Per-row argmax vs the target class; count of correct rows → the caller
// divides by batch_size (v2.30 evaluate companion).
__global__ void accuracy_rows_kernel(const float* logits, const int* targets,
                                     float* correct, int batch, int C) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float* row = logits + static_cast<size_t>(b) * C;
    int argmax = 0;
    for (int c = 1; c < C; ++c) {
        if (row[c] > row[argmax]) argmax = c;
    }
    if (argmax == targets[b]) {
        atomicAdd(&correct[0], 1.0f);
    }
}

}  // namespace

void cross_entropy_loss_device(
    const float* logits,
    const int* targets,
    float* loss_out,
    int batch_size,
    int num_classes,
    const CrossEntropyConfig& config,
    cudaStream_t stream
) {
    // Milestone v2.30 (DEC-016 / TASK-049 P2): device mean/sum CE loss. The
    // public host cross_entropy_loss() keeps its signature for host callers;
    // this entry computes the identical scalar from device logits so the
    // training drivers no longer round-trip the full logits buffer (the same
    // D2H family v2.27/v2.29 removed from the other hot paths).
    if (batch_size <= 0 || num_classes <= 0) {
        throw std::invalid_argument(
            "cross_entropy_loss_device requires positive batch_size and "
            "num_classes");
    }
    const float loss_scale =
        config.reduction_mean ? 1.0f / static_cast<float>(batch_size) : 1.0f;
    const size_t shmem = (kCeLossBlock / 32) * sizeof(float);

    // Same-stream cudaMemsetAsync is ordered before the kernel (in-order
    // streams), so loss_out[0] is guaranteed zero before any atomicAdd. Check
    // its return value like the launches — a failing memset would otherwise be
    // silently swallowed (v2.30 reviewer LOW L1).
    CUDA_CHECK(cudaMemsetAsync(loss_out, 0, sizeof(float), stream));
    ce_loss_rows_kernel<<<batch_size, kCeLossBlock, shmem, stream>>>(
        logits, targets, loss_out, batch_size, num_classes, loss_scale);
    // async launch errors surface on the next sync; check immediately so a
    // bad launch is reported here, not in a caller's later cudaStreamSync.
    CUDA_CHECK(cudaGetLastError());
}

void accuracy_device(
    const float* logits,
    const int* targets,
    float* correct_out,
    int batch_size,
    int num_classes,
    cudaStream_t stream
) {
    if (batch_size <= 0 || num_classes <= 0) {
        throw std::invalid_argument(
            "accuracy_device requires positive batch_size and num_classes");
    }
    // Same-stream memset before the atomicAdd kernel (v2.30 reviewer LOW L1).
    CUDA_CHECK(cudaMemsetAsync(correct_out, 0, sizeof(float), stream));
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;
    accuracy_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        logits, targets, correct_out, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
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

        // Two passes per row: accumulate the FULL softmax denominator first,
        // then divide. The old code divided by the running partial sum while
        // still accumulating — probs[class 0] was exp(logit0)/exp(logit0) ==
        // 1.0 for every row regardless of the data, and only the final class
        // got the correct denominator (the same bug class cross_entropy_loss
        // was already fixed for, TASK-077).
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += expf(predictions[b * num_classes + c] - max_logit);
        }
        for (int c = 0; c < num_classes; ++c) {
            probs[b * num_classes + c] =
                expf(predictions[b * num_classes + c] - max_logit) / sum_exp;
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
