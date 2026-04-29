#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/stream/stream.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuda::algo {

enum class FlashAttentionVersion {
    V2,
    V3
};

struct FlashAttentionConfig {
    int num_heads = 8;
    int num_kv_heads = 8;
    int head_dim = 64;
    int seq_len = 512;
    int batch_size = 1;
    float dropout_rate = 0.0f;
    bool causal = true;
    bool is_fp16 = true;
    int num_splits = 1;
    FlashAttentionVersion version = FlashAttentionVersion::V2;
    int tile_size = 64;
    uint64_t dropout_seed = 0;
    uint64_t dropout_offset = 0;
};

class FlashAttention {
public:
    explicit FlashAttention(const FlashAttentionConfig& config);
    ~FlashAttention();

    FlashAttention(const FlashAttention&) = delete;
    FlashAttention& operator=(const FlashAttention&) = delete;
    FlashAttention(FlashAttention&&) = default;
    FlashAttention& operator=(FlashAttention&&) = default;

    void forward(
        memory::Buffer<float>& output,
        memory::Buffer<float>& softmax_lse,
        const memory::Buffer<float>& query,
        const memory::Buffer<float>& key,
        const memory::Buffer<float>& value,
        const stream::Stream& stream
    );

    void forward_bf16(
        memory::Buffer<void>& output,
        memory::Buffer<float>& softmax_lse,
        const memory::Buffer<void>& query,
        const memory::Buffer<void>& key,
        const memory::Buffer<void>& value,
        const stream::Stream& stream
    );

    void backward(
        memory::Buffer<float>& dq,
        memory::Buffer<float>& dk,
        memory::Buffer<float>& dv,
        const memory::Buffer<float>& output,
        const memory::Buffer<float>& dout,
        const memory::Buffer<float>& query,
        const memory::Buffer<float>& key,
        const memory::Buffer<float>& value,
        const memory::Buffer<float>& softmax_lse,
        const stream::Stream& stream
    );

    void set_dropout(float rate, uint64_t seed = 0);
    void set_causal(bool causal) { config_.causal = causal; }
    bool get_causal() const { return config_.causal; }

    size_t get_workspace_size() const;
    void ensure_workspace(size_t bytes);

    FlashAttentionConfig config() const { return config_; }

private:
    FlashAttentionConfig config_;
    memory::Buffer<void> workspace_;
    void* dropout_state_ = nullptr;
};

std::unique_ptr<FlashAttention> create_flash_attention(
    const FlashAttentionConfig& config
);

namespace flash_attention_kernel {

struct alignas(16) AttentionParams {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* output_ptr;
    float* softmax_lse_ptr;
    float scale;
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int seq_len;
    int q_stride_h;
    int k_stride_h;
    int v_stride_h;
    int softmax_lse_stride;
    bool causal;
    bool is_fp16;
    int num_splits;
    void* workspace_ptr;
    uint64_t dropout_seed;
    uint64_t dropout_offset;
    float dropout_scale;
};

__global__ void flash_attention_fwd_kernel(AttentionParams params);
__global__ void flash_attention_bwd_kernel(AttentionParams params);

}  // namespace flash_attention_kernel

}  // namespace cuda::algo
