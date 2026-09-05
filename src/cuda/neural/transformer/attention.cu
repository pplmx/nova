#include "cuda/neural/transformer/attention.h"
#include "cuda/neural/matmul.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>

#include "cuda/device/error.h"

namespace cuda::neural::transformer {

namespace {

template <typename T>
__global__ void scale_kernel(T* data, int num_elements, T scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] *= scale;
    }
}

template <typename T>
__global__ void apply_causal_mask_kernel(
    T* scores, int batch_size, int num_heads,
    int seq_len, T mask_value
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;

    if (b < batch_size && h < num_heads && i < seq_len) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] = mask_value;
            }
        }
    }
}

template <typename T>
__global__ void softmax_kernel(T* data, int batch_size, int num_heads, int seq_len) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;

    if (b < batch_size && h < num_heads && i < seq_len) {
        T* row = &data[(b * num_heads + h) * seq_len * seq_len + i * seq_len];
        T max_val = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            max_val = max(max_val, row[j]);
        }

        T sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            row[j] = exp(row[j] - max_val);
            sum += row[j];
        }

        for (int j = 0; j < seq_len; ++j) {
            row[j] /= sum;
        }
    }
}

// Adds the positional encoding to every batch row's embeddings (TASK-077 /
// ISS-013): each of the `batch_size` rows receives the same per-position
// encoding, so output[(b*seq_len+p)*embed_dim + d] =
// input[..] + encoding[p*embed_dim + d]. Previously forward() memcpy'd the
// encoding OVER the input (losing the embeddings) and only touched the first
// batch row's span.
template <typename T>
__global__ void add_positional_kernel(
    const T* input, T* output, const T* encoding,
    int batch_size, int seq_len, int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * embed_dim;
    if (idx < total) {
        const int d = idx % embed_dim;
        const int p = (idx / embed_dim) % seq_len;
        output[idx] = input[idx] + encoding[p * embed_dim + d];
    }
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(const MultiHeadAttentionConfig& config)
    : config_(config), d_qkv_buffer_(nullptr), d_attn_weights_(nullptr),
      d_output_buffer_(nullptr), buffer_size_(0) {}

MultiHeadAttention::~MultiHeadAttention() {
    if (d_qkv_buffer_) cudaFree(d_qkv_buffer_);
    if (d_attn_weights_) cudaFree(d_attn_weights_);
    if (d_output_buffer_) cudaFree(d_output_buffer_);
}

void MultiHeadAttention::forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch_size,
    int seq_len,
    int qkv_dim,
    cudaStream_t stream
) {
    // This single-GPU attention path is deliberately NOT implemented (it was
    // superseded by TensorParallelMultiHeadAttention in v2.26 — see
    // tensor_parallel_attention.cpp's "anti-pattern" note). The old shell used
    // to scale an uninitialized scratch buffer and return without ever
    // computing attention or writing `output` — a silent garbage-producing
    // no-op (issue-v24-mha-incomplete). Fail fast instead, so a caller gets an
    // explicit error rather than reading untouched/garbage output.
    (void)query;
    (void)key;
    (void)value;
    (void)output;
    (void)batch_size;
    (void)seq_len;
    (void)qkv_dim;
    (void)stream;
    throw std::runtime_error(
        "MultiHeadAttention::forward: single-GPU multi-head attention is not "
        "implemented (superseded by TensorParallelMultiHeadAttention in "
        "v2.26); this class is a non-functional shell and must not be used");
}

void MultiHeadAttention::forward_self_attention(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    cudaStream_t stream
) {
    forward(input, input, input, output, batch_size, seq_len, hidden_dim, stream);
}

void MultiHeadAttention::set_dropout(float rate) {
    config_.dropout_rate = rate;
}

PositionalEncoding::PositionalEncoding(const PositionalEncodingConfig& config)
    : config_(config), d_encoding_buffer_(nullptr), buffer_size_(0) {
    compute_sinusoidal_encoding(config_.max_seq_len);
}

PositionalEncoding::~PositionalEncoding() {
    if (d_encoding_buffer_) cudaFree(d_encoding_buffer_);
}

void PositionalEncoding::forward(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    cudaStream_t stream
) {
    if (seq_len > config_.max_seq_len) {
        compute_sinusoidal_encoding(seq_len);
    }
    if (batch_size <= 0 || seq_len <= 0 || config_.embed_dim <= 0) {
        return;
    }
    if (d_encoding_buffer_ == nullptr) {
        // No encoding computed (e.g. an embed-dim of a type that produced no
        // buffer); behave as identity rather than overwriting with garbage.
        CUDA_CHECK(cudaMemcpyAsync(output, input,
                                   static_cast<size_t>(batch_size) * seq_len *
                                       config_.embed_dim * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }

    // output = input + positional encoding, repeated per batch row (TASK-077 /
    // ISS-013): previously the encoding was memcpy'd OVER the input — the first
    // sequence's embeddings were discarded and later batch rows got no encoding
    // at all.
    const int total = batch_size * seq_len * config_.embed_dim;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    add_positional_kernel<float><<<grid, block, 0, stream>>>(
        input, output, d_encoding_buffer_, batch_size, seq_len,
        config_.embed_dim);
    CUDA_CHECK(cudaGetLastError());
}

void PositionalEncoding::get_encoding(float* output, int seq_len, cudaStream_t stream) {
    if (seq_len > config_.max_seq_len) {
        compute_sinusoidal_encoding(seq_len);
    }

    if (d_encoding_buffer_) {
        CUDA_CHECK(cudaMemcpyAsync(output, d_encoding_buffer_, seq_len * config_.embed_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
    }
}

void PositionalEncoding::set_dropout(float rate) {
    config_.dropout_rate = rate;
}

void PositionalEncoding::compute_sinusoidal_encoding(int seq_len) {
    h_encoding_buffer_.resize(seq_len * config_.embed_dim);

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < config_.embed_dim; ++i) {
            float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / config_.embed_dim);
            if (i % 2 == 0) {
                h_encoding_buffer_[pos * config_.embed_dim + i] = sinf(angle);
            } else {
                h_encoding_buffer_[pos * config_.embed_dim + i] = cosf(angle);
            }
        }
    }

    if (d_encoding_buffer_) cudaFree(d_encoding_buffer_);
    CUDA_CHECK(cudaMalloc(&d_encoding_buffer_, seq_len * config_.embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_encoding_buffer_, h_encoding_buffer_.data(),
                          seq_len * config_.embed_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    buffer_size_ = seq_len * config_.embed_dim;
}

}  // namespace cuda::neural::transformer
