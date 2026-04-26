#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace cuda::neural::transformer {

struct MultiHeadAttentionConfig {
    int num_heads = 8;
    int head_dim = 64;
    float dropout_rate = 0.0f;
    bool use_causal_mask = false;
    bool scale_outputs = true;
};

class MultiHeadAttention {
public:
    explicit MultiHeadAttention(const MultiHeadAttentionConfig& config);
    ~MultiHeadAttention();

    void forward(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        int batch_size,
        int seq_len,
        int qkv_dim,
        cudaStream_t stream = nullptr
    );

    void forward_self_attention(
        const float* input,
        float* output,
        int batch_size,
        int seq_len,
        int hidden_dim,
        cudaStream_t stream = nullptr
    );

    void set_dropout(float rate);
    float get_dropout() const { return config_.dropout_rate; }

    int get_num_heads() const { return config_.num_heads; }
    int get_head_dim() const { return config_.head_dim; }

private:
    MultiHeadAttentionConfig config_;
    float* d_qkv_buffer_ = nullptr;
    float* d_attn_weights_ = nullptr;
    float* d_output_buffer_ = nullptr;
    size_t buffer_size_ = 0;
};

enum class PositionalEncodingType {
    Sinusoidal,
    Learned
};

struct PositionalEncodingConfig {
    PositionalEncodingType type = PositionalEncodingType::Sinusoidal;
    int max_seq_len = 512;
    int embed_dim = 512;
    float dropout_rate = 0.0f;
};

class PositionalEncoding {
public:
    explicit PositionalEncoding(const PositionalEncodingConfig& config);
    ~PositionalEncoding();

    void forward(
        const float* input,
        float* output,
        int batch_size,
        int seq_len,
        cudaStream_t stream = nullptr
    );

    void get_encoding(float* output, int seq_len, cudaStream_t stream = nullptr);

    void set_dropout(float rate);
    PositionalEncodingType get_type() const { return config_.type; }

private:
    void compute_sinusoidal_encoding(int seq_len);

    PositionalEncodingConfig config_;
    float* d_encoding_buffer_ = nullptr;
    std::vector<float> h_encoding_buffer_;
    size_t buffer_size_ = 0;
};

}  // namespace cuda::neural::transformer
