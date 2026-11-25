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

        // Copy and move are deleted.
        //
        // The class owns three cudaMalloc'd float buffers and its destructor cudaFree()s them, but it
        // declared none of the special members, so the compiler generated copy and move:
        // std::is_copy_constructible_v<MultiHeadAttention> reported true. A copy shared the three
        // pointers, so whichever object died second freed memory that was already gone.
        //
        // Copy would have to deep-copy three device buffers to be correct, which no caller asked for
        // and which would silently change this from a layer into a copier. Deleting it makes the
        // ownership explicit; callers that need two layers construct two.
        MultiHeadAttention(const MultiHeadAttention&) = delete;
        MultiHeadAttention& operator=(const MultiHeadAttention&) = delete;
        MultiHeadAttention(MultiHeadAttention&&) = delete;
        MultiHeadAttention& operator=(MultiHeadAttention&&) = delete;

        void forward(const float* query, const float* key, const float* value, float* output, int batch_size, int seq_len, int qkv_dim, cudaStream_t stream = nullptr);

        void forward_self_attention(const float* input, float* output, int batch_size, int seq_len, int hidden_dim, cudaStream_t stream = nullptr);

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

    enum class PositionalEncodingType { Sinusoidal, Learned };

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

        // Copy and move are deleted: the destructor cudaFree()s d_encoding_buffer_, so a copy would
        // share it and free it twice. Same shape as MultiHeadAttention above.
        PositionalEncoding(const PositionalEncoding&) = delete;
        PositionalEncoding& operator=(const PositionalEncoding&) = delete;
        PositionalEncoding(PositionalEncoding&&) = delete;
        PositionalEncoding& operator=(PositionalEncoding&&) = delete;

        void forward(const float* input, float* output, int batch_size, int seq_len, cudaStream_t stream = nullptr);

        void get_encoding(float* output, int seq_len, cudaStream_t stream = nullptr);

        void set_dropout(float rate);
        /// The dropout rate currently in effect. Exposed so a caller can confirm what it
        /// set: without it the setter wrote a private field that nothing could read back,
        /// and no test could distinguish a working setter from a no-op.
        [[nodiscard]] float get_dropout() const { return config_.dropout_rate; }
        PositionalEncodingType get_type() const { return config_.type; }

    private:
        void compute_sinusoidal_encoding(int seq_len);

        PositionalEncodingConfig config_;
        float* d_encoding_buffer_ = nullptr;
        std::vector<float> h_encoding_buffer_;
        size_t buffer_size_ = 0;
    };

}  // namespace cuda::neural::transformer
