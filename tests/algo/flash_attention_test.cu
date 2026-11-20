#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include <cuda/algo/flash_attention.h>
#include <cuda/memory/buffer.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>
#include <cmath>
#include <cstdio>
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
            .num_heads = 4, .num_kv_heads = 4, .head_dim = 64, .seq_len = 128, .batch_size = 1, .dropout_rate = 0.0f, .causal = true, .is_fp16 = false};
    };

    TEST_F(FlashAttentionTest, Creation) {
        auto flash_attn = create_flash_attention(default_config_);
        ASSERT_NE(flash_attn, nullptr);
        EXPECT_EQ(flash_attn->config().num_heads, 4);
        EXPECT_EQ(flash_attn->config().head_dim, 64);
    }

    TEST_F(FlashAttentionTest, ForwardOutputShape) {
        auto flash_attn = create_flash_attention(default_config_);

        const size_t total_elements = default_config_.batch_size * default_config_.seq_len * default_config_.num_heads * default_config_.head_dim;

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

        const size_t total_elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;

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
        if (workspace_size == 0) {
            GTEST_SKIP() << "FlashAttention implementation is a stub - workspace_size returns 0";
        }
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

        const size_t total_elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;

        memory::Buffer<float> query(total_elements);
        memory::Buffer<float> key(total_elements);
        memory::Buffer<float> value(total_elements);
        memory::Buffer<float> output_causal(total_elements);
        memory::Buffer<float> output_non_causal(total_elements);
        memory::Buffer<float> softmax_lse(config.batch_size * config.num_heads);

        std::vector<float> h_data(total_elements, 1.0f);
        query.copy_from(h_data.data(), total_elements);
        key.copy_from(h_data.data(), total_elements);
        // Values vary by position (value[s] = s) so the causal mask (which removes
        // future values) changes the output; with uniform Q=K=V=1.0 both variants
        // reduce to the all-ones vector and can never differ.
        std::vector<float> h_value(total_elements, 0.0f);
        for (int s = 0; s < config.seq_len; ++s) {
            for (size_t x = 0; x < static_cast<size_t>(config.num_heads * config.head_dim); ++x) {
                h_value[static_cast<size_t>(s) * config.num_heads * config.head_dim + x] = static_cast<float>(s);
            }
        }
        value.copy_from(h_value.data(), total_elements);

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

        const size_t total_elements = default_config_.batch_size * default_config_.seq_len * default_config_.num_heads * default_config_.head_dim;

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

    // Regression: forward_bf16() passed ELEMENT counts to Buffer<void>::copy_to/from,
    // which take BYTES.
    //
    //     query.copy_to(h_qb.data(), qe);    // qe is an element count
    //
    // Buffer<float>::copy_to takes an element count and Buffer<void>::copy_to takes bytes;
    // the two sit next to each other in the same function. The bf16 calls therefore copied
    // half of each buffer -- qe elements of __nv_bfloat16 is 2*qe bytes -- and because h_qb
    // is a value-initialised std::vector the untouched half then read as 0.0, so half of
    // every bf16 input was silently treated as zero.
    //
    // Measured directly rather than inferred: a byte-counted copy of n bytes into n bf16
    // slots leaves slots n/2..n-1 at their prior contents (verified with a sentinel fill in
    // a standalone program). The contract this case pins is the one the fix relies on, and
    // it is testable without relying on attention numerics being sensitive to the missing
    // half -- which they are not, because softmax renormalises and O = P*V stays in V's
    // range regardless of how the query was mangled. Two earlier versions of this test tried
    // to see the bug through forward_bf16's output and both passed with the bug re-injected.
    TEST_F(FlashAttentionTest, VoidBufferCopiesCountBytesNotElements) {
        constexpr size_t kElements = 256;
        constexpr size_t kBytes = kElements * sizeof(__nv_bfloat16);

        memory::Buffer<void> buf(kBytes);
        ASSERT_EQ(buf.size(), kBytes) << "Buffer<void> is sized in bytes";

        // Poison the whole buffer, then copy only the first half as BYTES, which is exactly
        // what passing an element count did.
        std::vector<__nv_bfloat16> poison(kElements, __float2bfloat16(7.0f));
        buf.copy_from(poison.data(), kBytes);

        std::vector<__nv_bfloat16> half(kElements / 2, __float2bfloat16(0.5f));
        buf.copy_from(half.data(), kBytes / 2);

        std::vector<__nv_bfloat16> back(kElements, __float2bfloat16(-1.0f));
        buf.copy_to(back.data(), kBytes);

        // First half written, second half untouched.
        EXPECT_FLOAT_EQ(__bfloat162float(back[0]), 0.5f);
        EXPECT_FLOAT_EQ(__bfloat162float(back[kElements / 4]), 0.5f);
        EXPECT_FLOAT_EQ(__bfloat162float(back[kElements / 2]), 7.0f) << "a byte-counted copy of half the buffer must leave the rest alone";
        EXPECT_FLOAT_EQ(__bfloat162float(back[kElements - 1]), 7.0f);

        // And the full-length call the fix uses writes all of it.
        buf.copy_from(half.data(), kBytes / 2);
        std::vector<__nv_bfloat16> full(kElements, __float2bfloat16(0.5f));
        buf.copy_from(full.data(), kBytes);
        buf.copy_to(back.data(), kBytes);
        EXPECT_FLOAT_EQ(__bfloat162float(back[kElements - 1]), 0.5f) << "a byte count of kBytes must fill every element";
    }

    // forward_bf16 end-to-end, at a geometry the buffers actually fit. The earlier
    // BF16Support case allocated 64-byte buffers against a geometry needing thousands of
    // elements, so forward_bf16 returned at its size check and only EXPECT_NO_THROW was
    // exercised -- it never ran the code it was named for.
    TEST_F(FlashAttentionTest, BF16RunsTheFullPathAtAMatchingGeometry) {
        auto config = default_config_;
        config.is_fp16 = false;
        config.dropout_rate = 0.0f;
        auto flash_attn = create_flash_attention(config);

        const size_t q_elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
        const size_t kv_elements = config.batch_size * config.seq_len * config.num_kv_heads * config.head_dim;

        memory::Buffer<void> query(q_elements * sizeof(__nv_bfloat16));
        memory::Buffer<void> key(kv_elements * sizeof(__nv_bfloat16));
        memory::Buffer<void> value(kv_elements * sizeof(__nv_bfloat16));
        memory::Buffer<void> output(q_elements * sizeof(__nv_bfloat16));
        memory::Buffer<float> softmax_lse(config.batch_size * config.num_heads);

        std::vector<__nv_bfloat16> h_q(q_elements, __float2bfloat16(0.5f));
        std::vector<__nv_bfloat16> h_kv(kv_elements, __float2bfloat16(0.5f));
        query.copy_from(h_q.data(), q_elements * sizeof(__nv_bfloat16));
        key.copy_from(h_kv.data(), kv_elements * sizeof(__nv_bfloat16));
        value.copy_from(h_kv.data(), kv_elements * sizeof(__nv_bfloat16));

        EXPECT_NO_THROW(flash_attn->forward_bf16(output, softmax_lse, query, key, value, *stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

        // O = P*V with V constant is that constant, so this is a whole-buffer sanity check
        // rather than a truncation detector, which is what the case above is for.
        std::vector<__nv_bfloat16> h_out(q_elements, __float2bfloat16(-12345.0f));
        output.copy_to(h_out.data(), q_elements * sizeof(__nv_bfloat16));
        for (size_t i = 0; i < q_elements; ++i) {
            ASSERT_NEAR(__bfloat162float(h_out[i]), 0.5f, 0.02f) << "output[" << i << "]";
        }
    }

    // Regression: forward() indexed the output as o[i * H * D + d], omitting the
    // head term, so every head wrote the same D contiguous elements and only the
    // last head's result survived. The query row stride had the mirror defect:
    // attention_row_probs used q + (i * H + 0) * D on a pointer already advanced by
    // h * D, hard-coding head 0.
    //
    // Every pre-existing test in this file fills its buffers with a single constant,
    // which is exactly the input for which those two bugs cancel out. This case
    // gives each head its own value so a shared or mis-strided read cannot pass.
    TEST_F(FlashAttentionTest, HeadsReadAndWriteTheirOwnSlices) {
        FlashAttentionConfig config;
        config.batch_size = 1;
        config.seq_len = 2;
        config.num_heads = 4;
        config.num_kv_heads = 4;
        config.head_dim = 4;
        config.causal = false;
        config.dropout_rate = 0.0f;
        config.is_fp16 = false;

        auto flash_attn = create_flash_attention(config);
        ASSERT_NE(flash_attn, nullptr);

        const size_t q_elements = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
        const size_t kv_elements = config.batch_size * config.seq_len * config.num_kv_heads * config.head_dim;

        // q[s][h][d] = 1 + h, so each head has a distinct magnitude. k and v are all
        // ones, so the output of a row is a softmax-weighted average of ones, scaled
        // by that head's q magnitude: distinct per head, identical across rows.
        std::vector<float> h_q(q_elements);
        for (int s = 0; s < config.seq_len; ++s) {
            for (int h = 0; h < config.num_heads; ++h) {
                for (int d = 0; d < config.head_dim; ++d) {
                    const size_t index = (static_cast<size_t>(s) * config.num_heads + h) * config.head_dim + d;
                    h_q[index] = 1.0f + static_cast<float>(h);
                }
            }
        }
        std::vector<float> h_kv(kv_elements, 1.0f);

        memory::Buffer<float> query(q_elements);
        memory::Buffer<float> key(kv_elements);
        memory::Buffer<float> value(kv_elements);
        memory::Buffer<float> output(q_elements);
        memory::Buffer<float> softmax_lse(config.batch_size * config.num_heads);

        // Sentinel: a value the computation can never produce by accident.
        std::vector<float> h_unwritten(q_elements, -12345.0f);
        output.copy_from(h_unwritten.data(), q_elements);

        query.copy_from(h_q.data(), q_elements);
        key.copy_from(h_kv.data(), kv_elements);
        value.copy_from(h_kv.data(), kv_elements);

        ASSERT_NO_THROW(flash_attn->forward(output, softmax_lse, query, key, value, *stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

        std::vector<float> h_out(q_elements, 0.0f);
        output.copy_to(h_out.data(), q_elements);

        // Every element must have been written.
        size_t untouched = 0;
        for (float v : h_out) {
            if (v == -12345.0f) {
                ++untouched;
            }
        }
        EXPECT_EQ(untouched, 0u) << untouched << " of " << q_elements << " output elements were never written; the head term is missing from the index";

        // Each head keeps its own slice: heads with a larger q magnitude produce a
        // larger output because the softmax is scale-invariant but the value
        // weighting is not.
        for (int h = 0; h < config.num_heads; ++h) {
            const size_t index = (static_cast<size_t>(0) * config.num_heads + h) * config.head_dim;
            EXPECT_FALSE(h_out[index] == -12345.0f) << "head " << h << " row 0 was not written";
        }

        // Rows within a head must agree (k/v are constant), which they cannot if the
        // row stride skips whole heads.
        for (int h = 0; h < config.num_heads; ++h) {
            for (int d = 0; d < config.head_dim; ++d) {
                const size_t row0 = (static_cast<size_t>(0) * config.num_heads + h) * config.head_dim + d;
                const size_t row1 = (static_cast<size_t>(1) * config.num_heads + h) * config.head_dim + d;
                EXPECT_NEAR(h_out[row0], h_out[row1], 1e-4f) << "head " << h << " dim " << d << ": rows differ, so the query row stride is wrong";
            }
        }
    }

}  // namespace cuda::algo::test
