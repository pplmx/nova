#include <gtest/gtest.h>
#include <cuda/distributed/sequence_parallel.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>

namespace cuda::distributed::test {

class SequenceParallelEdgeTest : public ::testing::Test {
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

TEST_F(SequenceParallelEdgeTest, SingleGPUConfigValidation) {
    SequenceParallelConfig config{
        .num_model_parallel_gpus = 1,
        .sequence_parallel_size = 1,
        .reduce_scatter_output = true,
        .rank = 0,
        .world_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    EXPECT_TRUE(config.num_model_parallel_gpus >= 1);
    EXPECT_TRUE(config.sequence_parallel_size >= 1);
    EXPECT_EQ(config.rank, 0);
    EXPECT_EQ(config.world_size, 1);
}

TEST_F(SequenceParallelEdgeTest, MultiGPUConfigValidation) {
    SequenceParallelConfig config{
        .num_model_parallel_gpus = 4,
        .sequence_parallel_size = 4,
        .reduce_scatter_output = true,
        .rank = 3,
        .world_size = 4,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    EXPECT_EQ(config.rank, 3);
    EXPECT_EQ(config.world_size, 4);
    EXPECT_GE(config.num_model_parallel_gpus, 1);
    EXPECT_GE(config.sequence_parallel_size, 1);
}

TEST_F(SequenceParallelEdgeTest, GatherKVWithEmptyBuffers) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> local_k(0);
    memory::Buffer<float> local_v(0);
    memory::Buffer<float> gathered_k(0);
    memory::Buffer<float> gathered_v(0);

    EXPECT_NO_THROW(sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_));
}

TEST_F(SequenceParallelEdgeTest, GatherKVWithLargeBuffers) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t large_size = 1024 * 1024;
    memory::Buffer<float> local_k(large_size);
    memory::Buffer<float> local_v(large_size);
    memory::Buffer<float> gathered_k(large_size);
    memory::Buffer<float> gathered_v(large_size);

    std::vector<float> data(large_size, 1.0f);
    local_k.copy_from(data.data(), large_size);
    local_v.copy_from(data.data(), large_size);

    EXPECT_NO_THROW(sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, ScatterOutputWithEmptyBuffer) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> local_output(0);
    memory::Buffer<float> full_output(0);

    EXPECT_NO_THROW(sp_attn->scatter_output(local_output, full_output, *stream_));
}

TEST_F(SequenceParallelEdgeTest, ScatterOutputWithReduceScatterDisabled) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .reduce_scatter_output = false,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t size = 256;
    memory::Buffer<float> local_output(size);
    memory::Buffer<float> full_output(size);

    std::vector<float> data(size, 3.0f);
    full_output.copy_from(data.data(), size);

    EXPECT_NO_THROW(sp_attn->scatter_output(local_output, full_output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, AllReduceWithEmptyBuffer) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    memory::Buffer<float> data(0);

    EXPECT_NO_THROW(sp_attn->all_reduce_sequence(data, *stream_));
}

TEST_F(SequenceParallelEdgeTest, AllReduceWithLargeBuffer) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t size = 1024 * 1024;
    memory::Buffer<float> data(size);

    std::vector<float> init_data(size, 5.0f);
    data.copy_from(init_data.data(), size);

    EXPECT_NO_THROW(sp_attn->all_reduce_sequence(data, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, RingAttentionSingleGPU) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .rank = 0,
        .world_size = 1
    };

    auto ring_attn = std::make_unique<RingSequenceParallelism>(config);

    memory::Buffer<float> query(128);
    memory::Buffer<float> key(128);
    memory::Buffer<float> value(128);
    memory::Buffer<float> output(128);

    EXPECT_NO_THROW(ring_attn->ring_attention(query, key, value, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, RingAttentionWithLargerBuffers) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .rank = 0,
        .world_size = 1
    };

    auto ring_attn = std::make_unique<RingSequenceParallelism>(config);

    const size_t size = 1024;
    memory::Buffer<float> query(size);
    memory::Buffer<float> key(size);
    memory::Buffer<float> value(size);
    memory::Buffer<float> output(size);

    std::vector<float> data(size, 2.0f);
    query.copy_from(data.data(), size);
    key.copy_from(data.data(), size);
    value.copy_from(data.data(), size);

    EXPECT_NO_THROW(ring_attn->ring_attention(query, key, value, output, *stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, ConfigEqualityCheck) {
    SequenceParallelConfig config1{
        .num_model_parallel_gpus = 2,
        .sequence_parallel_size = 2,
        .reduce_scatter_output = true,
        .rank = 0,
        .world_size = 2
    };

    SequenceParallelConfig config2{
        .num_model_parallel_gpus = 2,
        .sequence_parallel_size = 2,
        .reduce_scatter_output = true,
        .rank = 0,
        .world_size = 2
    };

    auto sp_attn1 = std::make_unique<SequenceParallelAttention>(config1);
    auto sp_attn2 = std::make_unique<SequenceParallelAttention>(config2);

    EXPECT_EQ(sp_attn1->config().num_model_parallel_gpus,
             sp_attn2->config().num_model_parallel_gpus);
    EXPECT_EQ(sp_attn1->config().sequence_parallel_size,
             sp_attn2->config().sequence_parallel_size);
}

TEST_F(SequenceParallelEdgeTest, ConfigModificationThroughAccessor) {
    SequenceParallelConfig config{
        .num_model_parallel_gpus = 2,
        .sequence_parallel_size = 2,
        .rank = 1,
        .world_size = 2
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    auto retrieved = sp_attn->config();
    EXPECT_EQ(retrieved.rank, 1);
    EXPECT_EQ(retrieved.world_size, 2);
    EXPECT_EQ(sp_attn->get_rank(), 1);
    EXPECT_EQ(sp_attn->get_sequence_parallel_size(), 2);
}

TEST_F(SequenceParallelEdgeTest, StreamSynchronizationAfterOperations) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t size = 512;
    memory::Buffer<float> local_k(size);
    memory::Buffer<float> local_v(size);
    memory::Buffer<float> gathered_k(size);
    memory::Buffer<float> gathered_v(size);

    sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
    SUCCEED();
}

TEST_F(SequenceParallelEdgeTest, MultipleGatherOperations) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t size = 256;
    for (int i = 0; i < 10; ++i) {
        memory::Buffer<float> local_k(size);
        memory::Buffer<float> local_v(size);
        memory::Buffer<float> gathered_k(size);
        memory::Buffer<float> gathered_v(size);

        EXPECT_NO_THROW(sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, MultipleScatterOperations) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t size = 256;
    for (int i = 0; i < 10; ++i) {
        memory::Buffer<float> local_output(size);
        memory::Buffer<float> full_output(size);

        EXPECT_NO_THROW(sp_attn->scatter_output(local_output, full_output, *stream_));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, MultipleAllReduceOperations) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t size = 256;
    for (int i = 0; i < 10; ++i) {
        memory::Buffer<float> data(size);
        EXPECT_NO_THROW(sp_attn->all_reduce_sequence(data, *stream_));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_->get()));
}

TEST_F(SequenceParallelEdgeTest, KVBufferSizesMatch) {
    SequenceParallelConfig config{
        .sequence_parallel_size = 1,
        .comm = nullptr
    };

    auto sp_attn = std::make_unique<SequenceParallelAttention>(config);

    const size_t kv_size = 512;
    memory::Buffer<float> local_k(kv_size);
    memory::Buffer<float> local_v(kv_size);
    memory::Buffer<float> gathered_k(kv_size);
    memory::Buffer<float> gathered_v(kv_size);

    sp_attn->gather_kv(gathered_k, gathered_v, local_k, local_v, *stream_);

    EXPECT_EQ(gathered_k.size(), local_k.size());
    EXPECT_EQ(gathered_v.size(), local_v.size());
}

TEST_F(SequenceParallelEdgeTest, ReduceScatterOutputFlag) {
    SequenceParallelConfig config_rs{
        .sequence_parallel_size = 1,
        .reduce_scatter_output = true,
        .comm = nullptr
    };

    SequenceParallelConfig config_no_rs{
        .sequence_parallel_size = 1,
        .reduce_scatter_output = false,
        .comm = nullptr
    };

    auto sp_attn_rs = std::make_unique<SequenceParallelAttention>(config_rs);
    auto sp_attn_no_rs = std::make_unique<SequenceParallelAttention>(config_no_rs);

    EXPECT_TRUE(sp_attn_rs->config().reduce_scatter_output);
    EXPECT_FALSE(sp_attn_no_rs->config().reduce_scatter_output);
}

TEST_F(SequenceParallelEdgeTest, SequenceParallelSizeAccessor) {
    SequenceParallelConfig config1{
        .sequence_parallel_size = 1
    };

    SequenceParallelConfig config4{
        .sequence_parallel_size = 4
    };

    auto sp_attn1 = std::make_unique<SequenceParallelAttention>(config1);
    auto sp_attn4 = std::make_unique<SequenceParallelAttention>(config4);

    EXPECT_EQ(sp_attn1->get_sequence_parallel_size(), 1);
    EXPECT_EQ(sp_attn4->get_sequence_parallel_size(), 4);
    EXPECT_FALSE(sp_attn1->has_sequence_parallelism());
    EXPECT_TRUE(sp_attn4->has_sequence_parallelism());
}

}  // namespace cuda::distributed::test
