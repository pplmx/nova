#pragma once

#include "cuda/distributed/common.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/stream/stream.h"
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(NOVA_NCCL_ENABLED)
#include <nccl.h>
#include "cuda/device/error.h"
#endif

namespace cuda::distributed {

using cuda::stream::Stream;

// Device-kernel host entry points for the ring attention (defined in
// src/cuda/distributed/ring_attention.cu; the header only orchestrates).
void ring_attn_init(
    float* out, float* running_max, float* running_l,
    int local_seq, int hidden, cudaStream_t stream);
void ring_attn_block(
    const float* q, const float* k, const float* v,
    float* out, float* running_max, float* running_l,
    int local_seq, int keys, int hidden, float inv_sqrt, cudaStream_t stream);
void ring_attn_finalize(
    float* out, const float* running_l,
    int local_seq, int hidden, cudaStream_t stream);

struct SequenceParallelConfig {
    int num_model_parallel_gpus = 1;
    int sequence_parallel_size = 1;
    bool reduce_scatter_output = true;
    int rank = 0;
    int world_size = 1;
    void* comm = nullptr;
    /** Feature dim of the Q/K/V rows. Required by the multi-GPU ring attention
     *  (RingSequenceParallelism) to compute the per-token dot products; the
     *  all-gather / scatter ops do not need it. */
    int hidden_dim = 0;
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

inline SequenceParallelAttention::SequenceParallelAttention(
    const SequenceParallelConfig& config
) : config_(config) {}

inline SequenceParallelAttention::~SequenceParallelAttention() = default;

inline void SequenceParallelAttention::gather_kv(
    memory::Buffer<float>& gathered_k,
    memory::Buffer<float>& gathered_v,
    const memory::Buffer<float>& local_k,
    const memory::Buffer<float>& local_v,
    const Stream& stream
) {
    if (!has_sequence_parallelism()) {
        if (gathered_k.size() < local_k.size()) {
            gathered_k = memory::Buffer<float>(local_k.size());
        }
        if (gathered_v.size() < local_v.size()) {
            gathered_v = memory::Buffer<float>(local_v.size());
        }
        gathered_k.copy_from(local_k.data(), local_k.size());
        gathered_v.copy_from(local_v.data(), local_v.size());
        return;
    }

#if defined(NOVA_NCCL_ENABLED)
    if (config_.comm == nullptr) {
        if (gathered_k.size() < local_k.size()) {
            gathered_k = memory::Buffer<float>(local_k.size());
        }
        if (gathered_v.size() < local_v.size()) {
            gathered_v = memory::Buffer<float>(local_v.size());
        }
        gathered_k.copy_from(local_k.data(), local_k.size());
        gathered_v.copy_from(local_v.data(), local_v.size());
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

inline void SequenceParallelAttention::scatter_output(
    memory::Buffer<float>& local_output,
    const memory::Buffer<float>& full_output,
    const Stream& stream
) {
    if (!has_sequence_parallelism()) {
        if (local_output.size() < full_output.size()) {
            local_output = memory::Buffer<float>(full_output.size());
        }
        local_output.copy_from(full_output.data(), full_output.size());
        return;
    }

#if defined(NOVA_NCCL_ENABLED)
    if (config_.comm == nullptr) {
        if (local_output.size() < full_output.size()) {
            local_output = memory::Buffer<float>(full_output.size());
        }
        local_output.copy_from(full_output.data(), full_output.size());
        return;
    }

    const size_t local_count = local_output.size();
    const size_t total_count = full_output.size();

    if (total_count != local_count * config_.sequence_parallel_size) {
        if (local_output.size() < full_output.size()) {
            local_output = memory::Buffer<float>(full_output.size());
        }
        local_output.copy_from(full_output.data(), full_output.size());
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

inline void SequenceParallelAttention::all_reduce_sequence(
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

inline RingSequenceParallelism::RingSequenceParallelism(
    const SequenceParallelConfig& config
) : config_(config) {
    // A ring needs at least 2 ranks; world_size 0/1 would make the modulo
    // below UB (division by zero) or self-loops. The multi-GPU ring path also
    // fails fast without a valid world_size, so these stay -1 and are never
    // used on the single-GPU fallback.
    prev_rank_ = -1;
    next_rank_ = -1;
    if (config_.world_size >= 2) {
        prev_rank_ = (config_.rank - 1 + config_.world_size) % config_.world_size;
        next_rank_ = (config_.rank + 1) % config_.world_size;
    }
}

inline RingSequenceParallelism::~RingSequenceParallelism() = default;

inline void RingSequenceParallelism::send_recv_kv(
    memory::Buffer<float>& send_k,
    memory::Buffer<float>& send_v,
    memory::Buffer<float>& recv_k,
    memory::Buffer<float>& recv_v,
    size_t count,
    const Stream& stream
) {
#if defined(NOVA_NCCL_ENABLED)
    // Ring step: send this rank's block clockwise to next_rank_ while receiving
    // prev_rank_'s block counter-clockwise. The four P2P ops are enqueued as one
    // ncclGroupStart/GroupEnd group: without group semantics NCCL serializes
    // consecutive point-to-point ops on the communicator and the ring deadlocks
    // (each rank's send blocks on a peer progress that never comes). Next/prev
    // are resolved from config rank / world_size in the constructor.
    auto dtype = detail::to_nccl_dtype(float{});
    ncclComm_t comm = static_cast<ncclComm_t>(config_.comm);
    ncclGroupStart();
    const ncclResult_t r = ncclSend(
        send_k.data(), count, dtype, next_rank_, comm, stream.get());
    const ncclResult_t r2 = ncclRecv(
        recv_k.data(), count, dtype, prev_rank_, comm, stream.get());
    const ncclResult_t r3 = ncclSend(
        send_v.data(), count, dtype, next_rank_, comm, stream.get());
    const ncclResult_t r4 = ncclRecv(
        recv_v.data(), count, dtype, prev_rank_, comm, stream.get());
    const ncclResult_t group = ncclGroupEnd();
    if (r != ncclSuccess || r2 != ncclSuccess || r3 != ncclSuccess ||
        r4 != ncclSuccess || group != ncclSuccess) {
        // Preserve the per-op ncclResult_t so a ring KV failure is diagnosable
        // (which op + GPU it broke on). Full comm-poisoning recovery
        // (mark_comm_aborted) needs an NcclContext reference the config does
        // not carry — tracked as a library-level follow-up.
        throw std::runtime_error(
            "RingSequenceParallelism::send_recv_kv: NCCL send/recv failed "
            "(ring KV exchange; send-K=" + std::to_string(r) +
            " recv-K=" + std::to_string(r2) +
            " send-V=" + std::to_string(r3) +
            " recv-V=" + std::to_string(r4) +
            " group=" + std::to_string(group) + ")");
    }
    CUDA_CHECK(cudaGetLastError());
#else
    (void)send_k;
    (void)send_v;
    (void)recv_k;
    (void)recv_v;
    (void)count;
    (void)stream;
#endif
}

inline void RingSequenceParallelism::ring_attention(
    memory::Buffer<float>& query,
    memory::Buffer<float>& key,
    memory::Buffer<float>& value,
    memory::Buffer<float>& output,
    const Stream& stream
) {
    if (!has_ring_parallelism()) {
        if (output.size() < query.size()) {
            output = memory::Buffer<float>(query.size());
        }
        output.copy_from(query.data(), query.size());
        return;
    }

    // Multi-GPU ring sequence parallelism (milestone v2.22 / TASK-016): each
    // rank owns a local sequence shard of Q/K/V. The ring walks P-1 remote
    // blocks (send_recv_kv) and accumulates each with online softmax, so the
    // per-rank output is the local queries' attention over the FULL KV
    // sequence. Numeric scale 1/sqrt(hidden) matches the attention reference.
    const int hidden = config_.hidden_dim;
    if (hidden <= 0) {
        throw std::invalid_argument(
            "RingSequenceParallelism::ring_attention: config.hidden_dim must be "
            "> 0 for multi-GPU ring attention");
    }
    const size_t elems = query.size();
    if (elems == 0 || elems % static_cast<size_t>(hidden) != 0 ||
        key.size() != elems || value.size() != elems) {
        throw std::invalid_argument(
            "RingSequenceParallelism::ring_attention: query/key/value must have "
            "equal sizes divisible by config.hidden_dim");
    }
    if (config_.comm == nullptr) {
        throw std::invalid_argument(
            "RingSequenceParallelism::ring_attention: multi-GPU ring requires "
            "a live communicator (config.comm)");
    }
    // The ring must walk exactly the gates' parallelism degree: a config whose
    // world_size disagrees with sequence_parallel_size would silently attend
    // only a prefix of the full sequence — fail fast instead (the class of
    // "silent partial attention" this milestone set out to eliminate).
    const int sp = config_.world_size;
    if (sp < 2 || sp != config_.sequence_parallel_size) {
        throw std::invalid_argument(
            "RingSequenceParallelism::ring_attention: world_size and "
            "sequence_parallel_size must be equal and >= 2 for the multi-GPU "
            "ring");
    }
    const size_t local_seq_sz = elems / static_cast<size_t>(hidden);
    if (local_seq_sz > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "RingSequenceParallelism::ring_attention: local sequence size "
            "exceeds the supported int range");
    }
    const int local_seq = static_cast<int>(local_seq_sz);
    if (output.size() < elems) {
        output = memory::Buffer<float>(elems);
    }

    // Buffer geometry in size_t so the intermediate sizes cannot overflow int.
    const size_t block_elems = static_cast<size_t>(local_seq) * hidden;
    memory::Buffer<float> running_max(local_seq);
    memory::Buffer<float> running_l(local_seq);
    memory::Buffer<float> cur_k(block_elems), cur_v(block_elems);
    memory::Buffer<float> nx_k(block_elems), nx_v(block_elems);
    cur_k.copy_from(key.data(), elems);
    cur_v.copy_from(value.data(), elems);

    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hidden));
    ring_attn_init(output.data(), running_max.data(), running_l.data(),
                   local_seq, hidden, stream.get());

    for (int step = 0; step < sp; ++step) {
        ring_attn_block(query.data(), cur_k.data(), cur_v.data(),
                        output.data(), running_max.data(), running_l.data(),
                        local_seq, local_seq, hidden, inv_sqrt, stream.get());
        if (step < sp - 1) {
            send_recv_kv(cur_k, cur_v, nx_k, nx_v, elems, stream);
            std::swap(cur_k, nx_k);
            std::swap(cur_v, nx_v);
        }
    }
    ring_attn_finalize(output.data(), running_l.data(),
                       local_seq, hidden, stream.get());
}

}  // namespace cuda::distributed
