#pragma once

#include "cuda/distributed/common.h"
#include "cuda/memory/buffer.h"
#include <memory>

#if defined(NOVA_NCCL_ENABLED)
#include <nccl.h>
#endif

namespace cuda::distributed {

struct SequenceParallelConfig {
    int num_model_parallel_gpus = 1;
    int sequence_parallel_size = 1;
    bool reduce_scatter_output = true;
    int rank = 0;
    int world_size = 1;
    void* comm = nullptr;
};

class SequenceParallelAttention {
public:
    explicit SequenceParallelAttention(const SequenceParallelConfig& config);
    ~SequenceParallelAttention();

    SequenceParallelAttention(const SequenceParallelAttention&) = delete;
    SequenceParallelAttention& operator=(const SequenceParallelAttention&) = delete;
    SequenceParallelAttention(SequenceParallelAttention&&) = default;
    SequenceParallelAttention& operator=(SequenceParallelAttention&&) = default;

    void gather_kv(
        memory::Buffer<float>& gathered_k,
        memory::Buffer<float>& gathered_v,
        const memory::Buffer<float>& local_k,
        const memory::Buffer<float>& local_v,
        const Stream& stream
    );

    void scatter_output(
        memory::Buffer<float>& local_output,
        const memory::Buffer<float>& full_output,
        const Stream& stream
    );

    void all_reduce_sequence(
        memory::Buffer<float>& data,
        const Stream& stream
    );

    bool has_sequence_parallelism() const {
        return config_.sequence_parallel_size > 1;
    }

    int get_sequence_parallel_size() const {
        return config_.sequence_parallel_size;
    }

    int get_rank() const { return config_.rank; }

    SequenceParallelConfig config() const { return config_; }

private:
    void all_gather_internal(
        memory::Buffer<float>& output,
        const memory::Buffer<float>& input,
        size_t count,
        const Stream& stream
    );

    void reduce_scatter_internal(
        memory::Buffer<float>& output,
        const memory::Buffer<float>& input,
        size_t count,
        const Stream& stream
    );

    void all_reduce_internal(
        memory::Buffer<float>& data,
        size_t count,
        const Stream& stream
    );

    SequenceParallelConfig config_;
};

class RingSequenceParallelism {
public:
    explicit RingSequenceParallelism(const SequenceParallelConfig& config);
    ~RingSequenceParallelism();

    RingSequenceParallelism(const RingSequenceParallelism&) = delete;
    RingSequenceParallelism& operator=(const RingSequenceParallelism&) = delete;
    RingSequenceParallelism(RingSequenceParallelism&&) = default;
    RingSequenceParallelism& operator=(RingSequenceParallelism&&) = default;

    void ring_attention(
        memory::Buffer<float>& query,
        memory::Buffer<float>& key,
        memory::Buffer<float>& value,
        memory::Buffer<float>& output,
        const Stream& stream
    );

    bool has_ring_parallelism() const {
        return config_.sequence_parallel_size > 1;
    }

private:
    void send_recv_kv(
        memory::Buffer<float>& send_k,
        memory::Buffer<float>& send_v,
        memory::Buffer<float>& recv_k,
        memory::Buffer<float>& recv_v,
        size_t count,
        const Stream& stream
    );

    SequenceParallelConfig config_;
    int prev_rank_;
    int next_rank_;
};

#if defined(NOVA_NCCL_ENABLED)

namespace detail {

inline ncclDataType_t to_nccl_dtype(float) { return ncclFloat32; }
inline ncclDataType_t to_nccl_dtype(__half) { return ncclHalf; }

template <typename T>
ncclResult_t nccl_all_gather(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclAllGather(sendbuf, recvbuf, count, datatype, comm, stream);
}

template <typename T>
ncclResult_t nccl_reduce_scatter(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclReduceScatter(sendbuf, recvbuf, count, datatype, op, comm, stream);
}

template <typename T>
ncclResult_t nccl_all_reduce(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm, stream);
}

}  // namespace detail

#endif

SequenceParallelAttention::SequenceParallelAttention(
    const SequenceParallelConfig& config
) : config_(config) {}

SequenceParallelAttention::~SequenceParallelAttention() = default;

void SequenceParallelAttention::gather_kv(
    memory::Buffer<float>& gathered_k,
    memory::Buffer<float>& gathered_v,
    const memory::Buffer<float>& local_k,
    const memory::Buffer<float>& local_v,
    const Stream& stream
) {
    if (!has_sequence_parallelism()) {
        gathered_k = local_k;
        gathered_v = local_v;
        return;
    }

#if defined(NOVA_NCCL_ENABLED)
    if (config_.comm == nullptr) {
        gathered_k = local_k;
        gathered_v = local_v;
        return;
    }

    const size_t local_count = local_k.size();
    const size_t total_count = local_count * config_.sequence_parallel_size;

    if (gathered_k.size() < total_count) {
        gathered_k = memory::Buffer<float>(total_count);
    }
    if (gathered_v.size() < total_count) {
        gathered_v = memory::Buffer<float>(total_count);
    }

    auto dtype = detail::to_nccl_dtype(float{});
    detail::nccl_all_gather<float>(
        local_k.data(), gathered_k.data(), local_count, dtype,
        static_cast<ncclComm_t>(config_.comm), stream.get()
    );
    CUDA_CHECK(cudaGetLastError());

    detail::nccl_all_gather<float>(
        local_v.data(), gathered_v.data(), local_count, dtype,
        static_cast<ncclComm_t>(config_.comm), stream.get()
    );
    CUDA_CHECK(cudaGetLastError());
#else
    (void)gathered_k;
    (void)gathered_v;
    (void)local_k;
    (void)local_v;
    (void)stream;
#endif
}

void SequenceParallelAttention::scatter_output(
    memory::Buffer<float>& local_output,
    const memory::Buffer<float>& full_output,
    const Stream& stream
) {
    if (!has_sequence_parallelism()) {
        local_output = full_output;
        return;
    }

#if defined(NOVA_NCCL_ENABLED)
    if (config_.comm == nullptr) {
        local_output = full_output;
        return;
    }

    const size_t local_count = local_output.size();
    const size_t total_count = full_output.size();

    if (total_count != local_count * config_.sequence_parallel_size) {
        local_output = full_output;
        return;
    }

    if (config_.reduce_scatter_output) {
        auto dtype = detail::to_nccl_dtype(float{});
        detail::nccl_reduce_scatter<float>(
            full_output.data(), local_output.data(), local_count, dtype,
            ncclSum, static_cast<ncclComm_t>(config_.comm), stream.get()
        );
        CUDA_CHECK(cudaGetLastError());
    } else {
        const size_t offset = config_.rank * local_count;
        CUDA_CHECK(cudaMemcpyAsync(
            local_output.data(),
            static_cast<const float*>(full_output.data()) + offset,
            local_count * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream.get()
        ));
    }
#else
    (void)local_output;
    (void)full_output;
    (void)stream;
#endif
}

void SequenceParallelAttention::all_reduce_sequence(
    memory::Buffer<float>& data,
    const Stream& stream
) {
    if (!has_sequence_parallelism()) {
        return;
    }

#if defined(NOVA_NCCL_ENABLED)
    if (config_.comm == nullptr) {
        return;
    }

    auto dtype = detail::to_nccl_dtype(float{});
    detail::nccl_all_reduce<float>(
        data.data(), data.data(), data.size(), dtype,
        ncclSum, static_cast<ncclComm_t>(config_.comm), stream.get()
    );
    CUDA_CHECK(cudaGetLastError());
#else
    (void)data;
    (void)stream;
#endif
}

RingSequenceParallelism::RingSequenceParallelism(
    const SequenceParallelConfig& config
) : config_(config) {
    prev_rank_ = (config_.rank - 1 + config_.world_size) % config_.world_size;
    next_rank_ = (config_.rank + 1) % config_.world_size;
}

RingSequenceParallelism::~RingSequenceParallelism() = default;

void RingSequenceParallelism::ring_attention(
    memory::Buffer<float>& query,
    memory::Buffer<float>& key,
    memory::Buffer<float>& value,
    memory::Buffer<float>& output,
    const Stream& stream
) {
    if (!has_ring_parallelism()) {
        output = query;
        return;
    }

#if defined(NOVA_NCCL_ENABLED)
    (void)query;
    (void)key;
    (void)value;
    (void)output;
    (void)stream;
#else
    output = query;
#endif
}

}  // namespace cuda::distributed
