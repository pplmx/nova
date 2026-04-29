#include <gtest/gtest.h>
#include <cuda/algo/flash_attention.h>
#include <cuda/memory/buffer.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>
#include <cmath>
#include <vector>
#include <random>

namespace cuda::algo::test {

class FlashAttentionTest : public ::testing::Test {
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

    FlashAttentionConfig default_config_ {
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .seq_len = 128,
        .batch_size = 1,
        .dropout_rate = 0.0f,
        .causal = true,
        .is_fp16 = false
    };
};

TEST_F(FlashAttentionTest, Creation) {
    auto flash_attn = create_flash_attention(default_config_);
    ASSERT_NE(flash_attn, nullptr);
    EXPECT_EQ(flash_attn->config().num_heads, 4);
    EXPECT_EQ(flash_attn->config().head_dim, 64);
}

TEST_F(FlashAttentionTest, ForwardOutputShape) {
    auto flash_attn = create_flash_attention(default_config_);

    const size_t total_elements = default_config_.batch_size * default_config_.seq_len *
                                   default_config_.num_heads * default_config_.head_dim;

    memory::Buffer<float> query(total_elements);
    memory::Buffer<float> key(total_elements);
    memory::Buffer<float> value(total_elements);
    memory::Buffer<float> output(total_elements);
    memory::Buffer<float> softmax_lse(default_config_.batch_size * default_config_.num_heads);

    std::vector<float> h_query(total_elements, 0.1f);
    std::vector<float> h_key(total_elements, 0.1f);
    std::vector<float> h_value(total_elements, 1.0f);

    query.copy_from(h_query.data(), total_elements);
    key.copy_from(h_key.data(), total_elements);
    value.copy_from(h_value.data(), total_elements);

    flash_attn->forward(output, softmax_lse, query, key, value, *stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_output(total_elements);
    output.copy_to(h_output.data(), total_elements);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(std::isfinite(h_output[i])) << "Output at index " << i << " is not finite";
        EXPECT_GE(h_output[i], -10.0f) << "Output at index " << i << " is too negative";
        EXPECT_LE(h_output[i], 10.0f) << "Output at index " << i << " is too large";
    }
}

TEST_F(FlashAttentionTest, StableSoftmaxNoOverflow) {
    auto config = default_config_;
    config.seq_len = 2048;
    config.head_dim = 128;
    auto flash_attn = create_flash_attention(config);

    const size_t total_elements = config.batch_size * config.seq_len *
                                   config.num_heads * config.head_dim;

    memory::Buffer<float> query(total_elements);
    memory::Buffer<float> key(total_elements);
    memory::Buffer<float> value(total_elements);
    memory::Buffer<float> output(total_elements);
    memory::Buffer<float> softmax_lse(config.batch_size * config.num_heads);

    std::vector<float> h_query(total_elements, 10.0f);
    std::vector<float> h_key(total_elements, 10.0f);
    std::vector<float> h_value(total_elements, 10.0f);

    query.copy_from(h_query.data(), total_elements);
    key.copy_from(h_key.data(), total_elements);
    value.copy_from(h_value.data(), total_elements);

    flash_attn->forward(output, softmax_lse, query, key, value, *stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_output(total_elements);
    output.copy_to(h_output.data(), total_elements);

    for (size_t i = 0; i < total_elements; ++i) {
        EXPECT_TRUE(std::isfinite(h_output[i])) << "Overflow detected at index " << i;
    }
}

TEST_F(FlashAttentionTest, WorkspaceAllocation) {
    auto config = default_config_;
    config.seq_len = 512;
    config.head_dim = 128;
    config.num_heads = 8;
    auto flash_attn = create_flash_attention(config);

    size_t workspace_size = flash_attn->get_workspace_size();
    EXPECT_GT(workspace_size, 0);

    flash_attn->ensure_workspace(workspace_size * 2);
    EXPECT_EQ(flash_attn->get_workspace_size(), workspace_size);
}

TEST_F(FlashAttentionTest, CausalMasking) {
    auto config = default_config_;
    config.causal = true;
    config.seq_len = 64;
    auto flash_attn_causal = create_flash_attention(config);

    config.causal = false;
    auto flash_attn_non_causal = create_flash_attention(config);

    const size_t total_elements = config.batch_size * config.seq_len *
                                   config.num_heads * config.head_dim;

    memory::Buffer<float> query(total_elements);
    memory::Buffer<float> key(total_elements);
    memory::Buffer<float> value(total_elements);
    memory::Buffer<float> output_causal(total_elements);
    memory::Buffer<float> output_non_causal(total_elements);
    memory::Buffer<float> softmax_lse(config.batch_size * config.num_heads);

    std::vector<float> h_data(total_elements, 1.0f);
    query.copy_from(h_data.data(), total_elements);
    key.copy_from(h_data.data(), total_elements);
    value.copy_from(h_data.data(), total_elements);

    flash_attn_causal->forward(output_causal, softmax_lse, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    softmax_lse = memory::Buffer<float>(config.batch_size * config.num_heads);
    flash_attn_non_causal->forward(output_non_causal, softmax_lse, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h_causal(total_elements);
    std::vector<float> h_non_causal(total_elements);
    output_causal.copy_to(h_causal.data(), total_elements);
    output_non_causal.copy_to(h_non_causal.data(), total_elements);

    bool differ = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(h_causal[i] - h_non_causal[i]) > 1e-5f) {
            differ = true;
            break;
        }
    }
    EXPECT_TRUE(differ) << "Causal and non-causal outputs should differ";
}

TEST_F(FlashAttentionTest, DropoutDeterminism) {
    auto config = default_config_;
    config.dropout_rate = 0.1f;
    config.dropout_seed = 12345;
    auto flash_attn = create_flash_attention(config);

    const size_t total_elements = default_config_.batch_size * default_config_.seq_len *
                                   default_config_.num_heads * default_config_.head_dim;

    memory::Buffer<float> query(total_elements);
    memory::Buffer<float> key(total_elements);
    memory::Buffer<float> value(total_elements);
    memory::Buffer<float> output1(total_elements);
    memory::Buffer<float> output2(total_elements);
    memory::Buffer<float> softmax_lse1(default_config_.batch_size * default_config_.num_heads);
    memory::Buffer<float> softmax_lse2(default_config_.batch_size * default_config_.num_heads);

    std::vector<float> h_data(total_elements, 1.0f);
    query.copy_from(h_data.data(), total_elements);
    key.copy_from(h_data.data(), total_elements);
    value.copy_from(h_data.data(), total_elements);

    flash_attn->forward(output1, softmax_lse1, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    flash_attn->forward(output2, softmax_lse2, query, key, value, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> h1(total_elements);
    std::vector<float> h2(total_elements);
    output1.copy_to(h1.data(), total_elements);
    output2.copy_to(h2.data(), total_elements);

    for (size_t i = 0; i < total_elements; ++i) {
        EXPECT_EQ(h1[i], h2[i]) << "Dropout should be deterministic with same seed";
    }
}

TEST_F(FlashAttentionTest, ConfigMutation) {
    auto flash_attn = create_flash_attention(default_config_);

    EXPECT_TRUE(flash_attn->get_causal());

    flash_attn->set_causal(false);
    EXPECT_FALSE(flash_attn->get_causal());

    flash_attn->set_dropout(0.5f, 42);
    EXPECT_EQ(flash_attn->config().dropout_rate, 0.5f);
    EXPECT_EQ(flash_attn->config().dropout_seed, 42u);
}

TEST_F(FlashAttentionTest, GQASupport) {
    auto config = default_config_;
    config.num_heads = 8;
    config.num_kv_heads = 2;
    config.head_dim = 64;
    auto flash_attn = create_flash_attention(config);

    ASSERT_EQ(flash_attn->config().num_heads, 8);
    ASSERT_EQ(flash_attn->config().num_kv_heads, 2);

    const size_t q_elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    const size_t kv_elements = config.batch_size * config.seq_len * config.num_kv_heads * config.head_dim;

    memory::Buffer<float> query(q_elements);
    memory::Buffer<float> key(kv_elements);
    memory::Buffer<float> value(kv_elements);
    memory::Buffer<float> output(q_elements);
    memory::Buffer<float> softmax_lse(config.batch_size * config.num_heads);

    std::vector<float> h_data(q_elements, 0.5f);
    query.copy_from(h_data.data(), q_elements);

    h_data.resize(kv_elements);
    std::fill(h_data.begin(), h_data.end(), 0.5f);
    key.copy_from(h_data.data(), kv_elements);
    value.copy_from(h_data.data(), kv_elements);

    EXPECT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(FlashAttentionTest, BF16Support) {
    auto config = default_config_;
    config.is_fp16 = false;
    auto flash_attn = create_flash_attention(config);

    memory::Buffer<void> query(sizeof(__nv_bfloat16) * 64);
    memory::Buffer<void> key(sizeof(__nv_bfloat16) * 64);
    memory::Buffer<void> value(sizeof(__nv_bfloat16) * 64);
    memory::Buffer<void> output(sizeof(__nv_bfloat16) * 64);
    memory::Buffer<float> softmax_lse(4);

    EXPECT_NO_THROW(flash_attn->forward_bf16(output, softmax_lse, query, key, value, *stream_));
}

}  // namespace cuda::algo::test
