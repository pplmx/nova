#include "cuda/neural/transformer/attention.h"
#include "cuda/neural/matmul.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>

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
    int total_heads = config_.num_heads;
    int head_dim = config_.head_dim;
    size_t qkv_size = batch_size * seq_len * qkv_dim;
    size_t attn_size = batch_size * total_heads * seq_len * seq_len;
    size_t output_size = batch_size * seq_len * qkv_dim;

    size_t needed = qkv_size * 3 + attn_size + output_size;
    if (needed > buffer_size_) {
        if (d_qkv_buffer_) cudaFree(d_qkv_buffer_);
        cudaMalloc(&d_qkv_buffer_, needed * sizeof(float));
        buffer_size_ = needed;
        d_attn_weights_ = d_qkv_buffer_ + qkv_size;
        d_output_buffer_ = d_attn_weights_ + attn_size;
    }

    int block = 256;
    int grid = (batch_size * total_heads * seq_len + block - 1) / block;

    if (config_.scale_outputs) {
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        scale_kernel<float><<<grid, block, 0, stream>>>(
            d_attn_weights_, attn_size, scale);
    }
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

    int num_elements = batch_size * seq_len * config_.embed_dim;
    cudaMemcpyAsync(output, input, num_elements * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    if (d_encoding_buffer_) {
        cudaMemcpyAsync(output, d_encoding_buffer_, seq_len * config_.embed_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

void PositionalEncoding::get_encoding(float* output, int seq_len, cudaStream_t stream) {
    if (seq_len > config_.max_seq_len) {
        compute_sinusoidal_encoding(seq_len);
    }

    if (d_encoding_buffer_) {
        cudaMemcpyAsync(output, d_encoding_buffer_, seq_len * config_.embed_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
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
    cudaMalloc(&d_encoding_buffer_, seq_len * config_.embed_dim * sizeof(float));
    cudaMemcpy(d_encoding_buffer_, h_encoding_buffer_.data(),
               seq_len * config_.embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    buffer_size_ = seq_len * config_.embed_dim;
}

}  // namespace cuda::neural::transformer
