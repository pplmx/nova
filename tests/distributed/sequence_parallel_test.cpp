#include <gtest/gtest.h>
#include <cuda/distributed/sequence_parallel.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>

namespace cuda::distributed::test {

class SequenceParallelTest : public ::testing::Test {
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

TEST_F(SequenceParallelTest, SingleGPUFallback) {
    SequenceParallelConfig config{
        .num_model_parallel_gpus = 1,
        .sequence_parallel_size = 1,
        .reduce_scatter_output = true,
        .rank = 0,
        .world_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);
    ASSERT_NE(sp_attn, nullptr);

    EXPECT_FALSE(sp_attn->has_sequence_parallelism());
    EXPECT_EQ(sp_attn->get_sequence_parallel_size(), 1);
    EXPECT_EQ(sp_attn->get_rank(), 0);
}

TEST_F(SequenceParallelTest, MultiGPUConfig) {
    SequenceParallelConfig config{
        .num_model_parallel_gpus = 2,
        .sequence_parallel_size = 2,
        .reduce_scatter_output = true,
        .rank = 1,
        .world_size = 2,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);
    ASSERT_NE(sp_attn, nullptr);

    EXPECT_TRUE(sp_attn->has_sequence_parallelism());
    EXPECT_EQ(sp_attn->get_sequence_parallel_size(), 2);
    EXPECT_EQ(sp_attn->get_rank(), 1);
}

TEST_F(SequenceParallelTest, GatherKVSingleGPU) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> local_k(128);
    memory::Buffer<float> local_v(128);
    memory::Buffer<float> gathered_k(128);
    memory::Buffer<float> gathered_v(128);

    std::vector<float> k_data(128, 1.0f);
    std::vector<float> v_data(128, 2.0f);
    local_k.copy_from(k_data.data(), 128);
    local_v.copy_from(v_data.data(), 128);

    sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> result_k(128);
    gathered_k.copy_to(result_k.data(), 128);

    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(result_k[i], 1.0f);
    }
}

TEST_F(SequenceParallelTest, ScatterOutputSingleGPU) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> local_output(128);
    memory::Buffer<float> full_output(128);

    std::vector<float> out_data(128, 3.0f);
    full_output.copy_from(out_data.data(), 128);

    sp_attn->scatter_output(local_output, full_output, *stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> result(128);
    local_output.copy_to(result.data(), 128);

    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(result[i], 3.0f);
    }
}

TEST_F(SequenceParallelTest, AllReduceSingleGPU) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> data(128);
    std::vector<float> init_data(128, 5.0f);
    data.copy_from(init_data.data(), 128);

    EXPECT_NO_THROW(sp_attn->all_reduce_sequence(data, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));

    std::vector<float> result(128);
    data.copy_to(result.data(), 128);

    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(result[i], 5.0f);
    }
}

TEST_F(SequenceParallelTest, RingParallelismSingleGPU) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .rank = 0,
        .world_size = 1
    };

    auto ring_attn = std::make_unique<RingSequenceParallelism>(config);
    ASSERT_NE(ring_attn, nullptr);

    EXPECT_FALSE(ring_attn->has_ring_parallelism());
}

TEST_F(SequenceParallelTest, ConfigAccessor) {
    SequenceParallelConfig config{
        .num_model_parallel_gpus = 4,
        .sequence_parallel_size = 4,
        .reduce_scatter_output = false,
        .rank = 2,
        .world_size = 4
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);
    auto retrieved_config = sp_attn->config();

    EXPECT_EQ(retrieved_config.num_model_parallel_gpus, 4);
    EXPECT_EQ(retrieved_config.sequence_parallel_size, 4);
    EXPECT_EQ(retrieved_config.reduce_scatter_output, false);
    EXPECT_EQ(retrieved_config.rank, 2);
    EXPECT_EQ(retrieved_config.world_size, 4);
}

TEST_F(SequenceParallelTest, KVCacheBufferSizes) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> local_k(256);
    memory::Buffer<float> local_v(256);
    memory::Buffer<float> gathered_k(256);
    memory::Buffer<float> gathered_v(256);

    sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_);

    EXPECT_EQ(gathered_k.size(), local_k.size());
    EXPECT_EQ(gathered_v.size(), local_v.size());
}

}  // namespace cuda::distributed::test
