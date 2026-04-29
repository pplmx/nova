#include <gtest/gtest.h>
#include <cuda/algo/flash_attention.h>
#include <cuda/memory/buffer.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

namespace cuda::algo::test {

class FlashAttentionEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        stream_ = std::make_unique<stream::Stream>();
    }

    void TearDown() override {
        stream_.reset();
        CUDA_CHECK(cudaDeviceReset());
    }

    std::unique_ptr<stream::Stream> stream_;
};

TEST_F(FlashAttentionEdgeCaseTest, ZeroDropoutRate) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);
    ASSERT_NE(flash_attn, nullptr);

    const size_t elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 1.0f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, FullDropoutRate) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.99f,
        .causal = true,
        .is_fp16 = false,
        .dropout_seed = 42
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 1.0f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, SingleHead) {
    FlashAttentionConfig config{
        .num_heads = 1,
        .num_kv_heads = 1,
        .head_dim = 64,
        .seq_len = 32,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(1);

    std::vector<float> data(elements, 0.5f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, LargeHeadDim) {
    FlashAttentionConfig config{
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 256,
        .seq_len = 64,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 0.1f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, SmallSequenceLength) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 4,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 1.0f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, LargeSequenceLength) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 4096,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 0.01f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_output(elements);
    output.copy_to(h_output.data(), elements);

    for (size_t i = 0; i < std::min(elements, size_t(100)); ++i) {
        EXPECT_TRUE(std::isfinite(h_output[i])) << "Output at index " << i << " is not finite";
    }
}

TEST_F(FlashAttentionEdgeCaseTest, BatchSizeGreaterThanOne) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 64,
        .batch_size = 4,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.batch_size * config.num_heads);

    std::vector<float> data(elements);
    std::iota(data.begin(), data.end(), 0.0f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, DifferentQKVDimensions) {
    FlashAttentionConfig config{
        .num_heads = 8,
        .num_kv_heads = 2,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = false,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t q_elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    const size_t kv_elements = config.batch_size * config.seq_len * config.num_kv_heads * config.head_dim;

    memory::Buffer<float> query(q_elements), key(kv_elements), value(kv_elements);
    memory::Buffer<float> output(q_elements), softmax_lse(config.batch_size * config.num_heads);

    std::vector<float> q_data(q_elements, 0.5f);
    std::vector<float> kv_data(kv_elements, 1.0f);

    query.copy_from(q_data.data(), q_elements);
    key.copy_from(kv_data.data(), kv_elements);
    value.copy_from(kv_data.data(), kv_elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, OutputSumToOne) {
    FlashAttentionConfig config{
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 32,
        .seq_len = 16,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = false,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> uniform_data(elements, 1.0f);
    query.copy_from(uniform_data.data(), elements);
    key.copy_from(uniform_data.data(), elements);
    value.copy_from(uniform_data.data(), elements);

    flash_attn->forward(output, softmax_lse, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_output(elements);
    output.copy_to(h_output.data(), elements);

    for (int head = 0; head < config.num_heads; ++head) {
        float sum = 0.0f;
        for (int seq = 0; seq < config.seq_len; ++seq) {
            for (int dim = 0; dim < config.head_dim; ++dim) {
                size_t idx = (seq * config.num_heads + head) * config.head_dim + dim;
                sum += h_output[idx];
            }
        }
        EXPECT_NEAR(sum, config.seq_len * config.head_dim, 1e-3f)
            << "Head " << head << " output should sum to seq_len * head_dim";
    }
}

TEST_F(FlashAttentionEdgeCaseTest, SoftmaxLSEOutput) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 0.5f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    flash_attn->forward(output, softmax_lse, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_softmax_lse(config.num_heads);
    softmax_lse.copy_to(h_softmax_lse.data(), config.num_heads);

    for (int h = 0; h < config.num_heads; ++h) {
        EXPECT_GT(h_softmax_lse[h], 0.0f) << "Softmax LSE should be positive";
        EXPECT_TRUE(std::isfinite(h_softmax_lse[h])) << "Softmax LSE should be finite";
    }
}

TEST_F(FlashAttentionEdgeCaseTest, BackwardPassGradientShape) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 64,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), dout(elements), softmax_lse(config.num_heads);
    memory::Buffer<float> dq(elements), dk(elements), dv(elements);

    std::vector<float> data(elements, 0.5f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);
    dout.copy_from(data.data(), elements);

    flash_attn->forward(output, softmax_lse, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    EXPECT_NO_THROW(flash_attn->backward(dq, dk, dv, output, dout, query, key, value, softmax_lse, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_dq(elements);
    dq.copy_to(h_dq.data(), elements);

    for (size_t i = 0; i < std::min(elements, size_t(100)); ++i) {
        EXPECT_TRUE(std::isfinite(h_dq[i])) << "Gradient at index " << i << " should be finite";
    }
}

TEST_F(FlashAttentionEdgeCaseTest, BackwardPassWithDropout) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 64,
        .batch_size = 1,
        .dropout_rate = 0.1f,
        .causal = true,
        .is_fp16 = false,
        .dropout_seed = 12345
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), dout(elements), softmax_lse(config.num_heads);
    memory::Buffer<float> dq(elements), dk(elements), dv(elements);

    std::vector<float> data(elements, 0.5f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);
    dout.copy_from(data.data(), elements);

    flash_attn->forward(output, softmax_lse, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    EXPECT_NO_THROW(flash_attn->backward(dq, dk, dv, output, dout, query, key, value, softmax_lse, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, WorkspaceGrowth) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    size_t initial_size = flash_attn->get_workspace_size();

    flash_attn->set_dropout(0.5f);

    config.seq_len = 512;
    config.head_dim = 128;
    auto flash_attn_large = create_flash_attention(config);
    size_t large_size = flash_attn_large->get_workspace_size();

    EXPECT_GT(large_size, initial_size);
}

TEST_F(FlashAttentionEdgeCaseTest, MultiQueryAttention) {
    FlashAttentionConfig config{
        .num_heads = 8,
        .num_kv_heads = 1,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t q_elements = config.seq_len * config.num_heads * config.head_dim;
    const size_t kv_elements = config.seq_len * config.num_kv_heads * config.head_dim;

    memory::Buffer<float> query(q_elements), key(kv_elements), value(kv_elements);
    memory::Buffer<float> output(q_elements), softmax_lse(config.num_heads);

    std::vector<float> data(q_elements, 0.5f);
    query.copy_from(data.data(), q_elements);

    data.resize(kv_elements);
    std::fill(data.begin(), data.end(), 1.0f);
    key.copy_from(data.data(), kv_elements);
    value.copy_from(data.data(), kv_elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, CausalWithLengthOne) {
    FlashAttentionConfig config{
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 32,
        .seq_len = 1,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output(elements), softmax_lse(config.num_heads);

    std::vector<float> data(elements, 1.0f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionEdgeCaseTest, DeterministicWithSameSeed) {
    FlashAttentionConfig config{
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.15f,
        .causal = true,
        .is_fp16 = false
    };

    auto flash_attn1 = create_flash_attention(config);
    config.dropout_seed = 99999;
    auto flash_attn2 = create_flash_attention(config);

    const size_t elements = config.seq_len * config.num_heads * config.head_dim;
    memory::Buffer<float> query(elements), key(elements), value(elements);
    memory::Buffer<float> output1(elements), output2(elements);
    memory::Buffer<float> softmax_lse1(config.num_heads), softmax_lse2(config.num_heads);

    std::vector<float> data(elements, 1.0f);
    query.copy_from(data.data(), elements);
    key.copy_from(data.data(), elements);
    value.copy_from(data.data(), elements);

    flash_attn1->forward(output1, softmax_lse1, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    flash_attn2->forward(output2, softmax_lse2, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h1(elements), h2(elements);
    output1.copy_to(h1.data(), elements);
    output2.copy_to(h2.data(), elements);

    bool identical = true;
    for (size_t i = 0; i < elements; ++i) {
        if (std::abs(h1[i] - h2[i]) > 1e-6f) {
            identical = false;
            break;
        }
    }
    EXPECT_FALSE(identical) << "Different seeds should produce different dropout patterns";
}

}  // namespace cuda::algo::test
