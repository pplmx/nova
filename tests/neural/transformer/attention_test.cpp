#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/transformer/attention.h>
#include <cuda/memory/buffer.h>
#include <cuda/memory/buffer-inl.h>

#include <vector>

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

// The single-GPU MultiHeadAttention is a superseded non-functional shell
// (v2.26 moved to TensorParallelMultiHeadAttention) that used to silently scale
// uninitialized scratch and return without computing attention or writing
// output — a garbage-producing no-op (issue-v24-mha-incomplete). It must fail
// fast instead, so a caller never reads untouched/garbage output.
TEST_F(MultiHeadAttentionTest, ForwardFailsFast_NotSilentNoop) {
    MultiHeadAttentionConfig config;
    config.num_heads = 2;
    config.head_dim = 32;

    MultiHeadAttention attn(config);

    int batch_size = 2;
    int seq_len = 4;
    int hidden_dim = 64;

    cuda::memory::Buffer<float> input(batch_size * seq_len * hidden_dim);
    cuda::memory::Buffer<float> output(batch_size * seq_len * hidden_dim);
    input.fill(0.1f);
    output.fill(0.0f);

    EXPECT_THROW(
        attn.forward_self_attention(
            input.data(), output.data(),
            batch_size, seq_len, hidden_dim, stream_),
        std::exception)
        << "unimplemented single-GPU attention must fail fast, not return as "
           "if it computed something";
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // The fail-fast must hold regardless of the scale_outputs knob.
    MultiHeadAttentionConfig c1;
    c1.scale_outputs = true;
    MultiHeadAttentionConfig c2;
    c2.scale_outputs = false;
    MultiHeadAttention attn1(c1), attn2(c2);
    EXPECT_THROW(
        attn1.forward_self_attention(
            input.data(), output.data(),
            batch_size, seq_len, hidden_dim, stream_),
        std::exception);
    EXPECT_THROW(
        attn2.forward_self_attention(
            input.data(), output.data(),
            batch_size, seq_len, hidden_dim, stream_),
        std::exception);
}

class PositionalEncodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
        cudaStreamCreate(&stream_);
        // This fixture asserts cudaGetLastError()==cudaSuccess after its own
        // async work (e.g. GetEncoding). Drain any stale sticky error left by an
        // earlier test so the assertion only reflects THIS test's operations.
        (void)cudaGetLastError();
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

    // get_encoding copies device-to-device into `output`, so it must be a GPU buffer.
    cuda::memory::Buffer<float> encoding(16 * 32);
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
    const int embed_dim = 16;

    // forward() must ADD the per-position encoding to every batch row (it used
    // to overwrite the first row's embeddings with the raw encoding and leave
    // later rows unencoded). Compute the reference from the encoding buffer.
    cuda::memory::Buffer<float> input(batch_size * seq_len * embed_dim);
    cuda::memory::Buffer<float> output(batch_size * seq_len * embed_dim);
    input.fill(0.5f);
    output.fill(0.0f);

    cuda::memory::Buffer<float> h_enc(seq_len * embed_dim);
    pos_enc.get_encoding(h_enc.data(), seq_len, stream_);
    cudaStreamSynchronize(stream_);
    std::vector<float> encoding(seq_len * embed_dim);
    h_enc.copy_to(encoding.data(), seq_len * embed_dim);

    pos_enc.forward(input.data(), output.data(), batch_size, seq_len, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<float> h_out(batch_size * seq_len * embed_dim);
    output.copy_to(h_out.data(), h_out.size());
    for (int b = 0; b < batch_size; ++b) {
        for (int p = 0; p < seq_len; ++p) {
            for (int d = 0; d < embed_dim; ++d) {
                // Every batch row gets input + the same per-position encoding.
                EXPECT_NEAR(
                    h_out[static_cast<size_t>(b * seq_len + p) * embed_dim + d],
                    0.5f + encoding[static_cast<size_t>(p) * embed_dim + d],
                    1e-4f)
                    << "output must equal input + positional encoding at batch "
                    << b << " pos " << p << " dim " << d
                    << " (row " << b << " was previously left unencoded)";
            }
        }
    }
}

TEST_F(PositionalEncodingTest, ForwardIdentityWhenNoInputRequired) {
    // Forward over a range that must not resize the encoding below seq_len and
    // must not crash; the add-semantics above already pins the numerics.
    PositionalEncodingConfig config;
    config.type = PositionalEncodingType::Sinusoidal;
    config.max_seq_len = 16;
    config.embed_dim = 8;

    PositionalEncoding pos_enc(config);
    cuda::memory::Buffer<float> input(2 * 4 * 8);
    cuda::memory::Buffer<float> output(2 * 4 * 8);
    input.fill(1.0f);
    output.fill(0.0f);
    pos_enc.forward(input.data(), output.data(), 2, 4, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace cuda::neural::transformer::test
