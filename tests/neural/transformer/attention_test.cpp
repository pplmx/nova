#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/transformer/attention.h>

namespace cuda::neural::transformer::test {

class MultiHeadAttentionTest : public ::testing::Test {
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

TEST_F(MultiHeadAttentionTest, BasicConstruction) {
    MultiHeadAttentionConfig config;
    config.num_heads = 8;
    config.head_dim = 64;
    config.dropout_rate = 0.1f;
    config.use_causal_mask = false;

    MultiHeadAttention attn(config);
    EXPECT_EQ(attn.get_num_heads(), 8);
    EXPECT_EQ(attn.get_head_dim(), 64);
}

TEST_F(MultiHeadAttentionTest, SetDropout) {
    MultiHeadAttentionConfig config;
    config.num_heads = 4;
    config.head_dim = 32;

    MultiHeadAttention attn(config);
    attn.set_dropout(0.5f);
    EXPECT_EQ(attn.get_dropout(), 0.5f);
}

TEST_F(MultiHeadAttentionTest, ForwardSelfAttention) {
    MultiHeadAttentionConfig config;
    config.num_heads = 2;
    config.head_dim = 32;

    MultiHeadAttention attn(config);

    int batch_size = 2;
    int seq_len = 4;
    int hidden_dim = 64;

    std::vector<float> input(batch_size * seq_len * hidden_dim, 0.1f);
    std::vector<float> output(batch_size * seq_len * hidden_dim, 0.0f);

    attn.forward_self_attention(
        input.data(), output.data(),
        batch_size, seq_len, hidden_dim, stream_
    );

    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(MultiHeadAttentionTest, ScaleOutputs) {
    MultiHeadAttentionConfig config1;
    config1.scale_outputs = true;

    MultiHeadAttentionConfig config2;
    config2.scale_outputs = false;

    MultiHeadAttention attn1(config1);
    MultiHeadAttention attn2(config2);
}

class PositionalEncodingTest : public ::testing::Test {
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

TEST_F(PositionalEncodingTest, SinusoidalEncoding) {
    PositionalEncodingConfig config;
    config.type = PositionalEncodingType::Sinusoidal;
    config.max_seq_len = 128;
    config.embed_dim = 64;

    PositionalEncoding pos_enc(config);
    EXPECT_EQ(pos_enc.get_type(), PositionalEncodingType::Sinusoidal);
}

TEST_F(PositionalEncodingTest, LearnedEncoding) {
    PositionalEncodingConfig config;
    config.type = PositionalEncodingType::Learned;
    config.max_seq_len = 128;
    config.embed_dim = 64;

    PositionalEncoding pos_enc(config);
    EXPECT_EQ(pos_enc.get_type(), PositionalEncodingType::Learned);
}

TEST_F(PositionalEncodingTest, SetDropout) {
    PositionalEncodingConfig config;
    config.type = PositionalEncodingType::Sinusoidal;

    PositionalEncoding pos_enc(config);
    pos_enc.set_dropout(0.2f);
}

TEST_F(PositionalEncodingTest, GetEncoding) {
    PositionalEncodingConfig config;
    config.type = PositionalEncodingType::Sinusoidal;
    config.max_seq_len = 16;
    config.embed_dim = 32;

    PositionalEncoding pos_enc(config);

    std::vector<float> encoding(16 * 32);
    pos_enc.get_encoding(encoding.data(), 16, stream_);

    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(PositionalEncodingTest, Forward) {
    PositionalEncodingConfig config;
    config.type = PositionalEncodingType::Sinusoidal;
    config.max_seq_len = 8;
    config.embed_dim = 16;

    PositionalEncoding pos_enc(config);

    int batch_size = 2;
    int seq_len = 8;

    std::vector<float> input(batch_size * seq_len * 16, 0.5f);
    std::vector<float> output(batch_size * seq_len * 16, 0.0f);

    pos_enc.forward(input.data(), output.data(), batch_size, seq_len, stream_);

    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace cuda::neural::transformer::test
