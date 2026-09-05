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
#include "cuda/nccl/nccl_recovery.h"
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
    /** Opaque pointer to the owning NcclContext (a cuda::nccl::NcclContext*),
     *  or null. When the caller also passes a live `comm`, set this so a failed
     *  collective aborts the communicator and flips the context's has_nccl()
     *  false — the self-healing initialize() path can then re-establish NCCL
     *  instead of letting every later NCCL user silently reuse a dead comm.
     *  May stay null when `comm` is null (single-GPU fallback). */
    void* nccl_context = nullptr;
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
    // The collective bodies live inline in gather_kv / scatter_output /
    // all_reduce_sequence; the private all_gather/reduce_scatter/all_reduce
    // _internal helpers that used to be declared here were removed — they were
    // declared but never defined or called (ISS-012).

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

// Surface a raw NCCL collective failure. `poison` distinguishes the two cases
// under the same policy as the comm-error layer (safe_nccl_call):
//   - poison == false: a synchronous caller error (e.g. a deliberate
//     invalid-argument call) — thrown WITHOUT poisoning, because a negative
//     test must not flip has_nccl() false for every later NCCL user in the
//     process.
//   - poison == true: the communicator itself is broken (async/comm-level
//     error) — abort + poison the owning context so has_nccl() goes false and
//     the self-healing initialize() path can recover instead of a silent hang
//     on the dead comm.
// ncclGetErrorString is only called on a real failure; `code` is a valid NCCL
// error code here.
[[noreturn]] inline void collect_failed(
    const char* where, const char* op, ncclResult_t code,
    void* comm, void* nccl_context, bool poison) {
    if (poison) {
        cuda::nccl::poison_failed_comm(comm, nccl_context);
    }
    throw std::runtime_error(
        std::string(where) + ": NCCL " + op + " failed (" +
        std::to_string(static_cast<int>(code)) + ": " +
        ncclGetErrorString(code) + ")");
}

// Run one raw NCCL collective (fn must return ncclResult_t) with comm-error
// layer-compatible handling. Never returns on failure: an immediate non-success
// is a caller error (thrown, not poisoned); an error detected AFTER a
// successful enqueue — the single-shot ncclCommGetAsyncError poll below — is a
// broken communicator and is poisoned (see collect_failed).
template <typename Fn>
inline void checked_collective(
    const char* where, const char* op, Fn&& fn,
    void* comm, void* nccl_context) {
    const ncclResult_t rc = fn();
    if (rc != ncclSuccess) {
        collect_failed(where, op, rc, comm, nccl_context, /*poison=*/false);
    }
    // Enqueue succeeded; one-shot async poll for a comm-level failure (e.g. a
    // peer rank's transport died mid-collective). A broken comm that has
    // ALREADY reported an error must be noticed here, not left for the
    // caller's cudaStreamSynchronize. ncclInProgress means the op is still in
    // flight with no error yet — proceed (the same "keep waiting, not a
    // failure" reading safe_nccl_call's poll loop uses; a single-shot poll
    // cannot hold for a hang, the caller's eventual stream sync is that
    // backstop). Only a reported non-in-progress error is a broken comm.
    ncclResult_t async_err = ncclSuccess;
    if (ncclCommGetAsyncError(static_cast<ncclComm_t>(comm), &async_err) !=
        ncclSuccess) {
        collect_failed(where, op, ncclInternalError, comm, nccl_context,
                       /*poison=*/true);
    }
    if (async_err != ncclSuccess && async_err != ncclInProgress) {
        collect_failed(where, op, async_err, comm, nccl_context, /*poison=*/true);
    }
}

// Fail-fast gate for a configured communicator (TASK-074). When the caller
// wired an owning NcclContext and that context is broken (has_nccl() false —
// a prior collective aborted and poisoned it), `config.comm` points at a DEAD
// communicator. Touching it is undefined (observed to SEGV in the ring-abort
// multi-GPU test), so throw a clear error before any NCCL call: the documented
// recovery is re-initializing the NcclContext (which re-establishes clean
// comms) rather than retrying a config that can never work again.
inline void check_comm_healthy(
    const char* where, const void* comm, const void* nccl_context) {
    if (comm != nullptr && !cuda::nccl::comm_context_healthy(
                               const_cast<void*>(nccl_context))) {
        throw std::runtime_error(
            std::string(where) + ": the owning NcclContext is broken "
            "(has_nccl() false) — config.comm points at a communicator a prior "
            "collective aborted; re-initialize the NcclContext and rebuild the "
            "config before issuing more sequence-parallel collectives");
    }
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
    detail::check_comm_healthy(
        "SequenceParallelAttention::gather_kv", config_.comm,
        config_.nccl_context);

    const size_t local_count = local_k.size();
    const size_t total_count = local_count * config_.sequence_parallel_size;

    if (gathered_k.size() < total_count) {
        gathered_k = memory::Buffer<float>(total_count);
    }
    if (gathered_v.size() < total_count) {
        gathered_v = memory::Buffer<float>(total_count);
    }

    auto dtype = detail::to_nccl_dtype(float{});
    // Route the raw AllGather through the comm-error policy (checked_collective
    // / nccl_recovery.h): a broken comm is aborted and poisons has_nccl() so
    // the self-healing initialize() path recovers, instead of the pre-TASK-018
    // behavior of silently discarding the ncclResult_t and scanning only the
    // CUDA error.
    detail::checked_collective(
        "SequenceParallelAttention::gather_kv", "AllGather(K)",
        [&]() {
            return detail::nccl_all_gather<float>(
                local_k.data(), gathered_k.data(), local_count, dtype,
                static_cast<ncclComm_t>(config_.comm), stream.get());
        },
        config_.comm, config_.nccl_context);
    CUDA_CHECK(cudaGetLastError());

    detail::checked_collective(
        "SequenceParallelAttention::gather_kv", "AllGather(V)",
        [&]() {
            return detail::nccl_all_gather<float>(
                local_v.data(), gathered_v.data(), local_count, dtype,
                static_cast<ncclComm_t>(config_.comm), stream.get());
        },
        config_.comm, config_.nccl_context);
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
    detail::check_comm_healthy(
        "SequenceParallelAttention::scatter_output", config_.comm,
        config_.nccl_context);

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
        // Comm-error layer routing (see gather_kv for the rationale): a dead
        // comm is aborted + poisons has_nccl() so self-healing can recover.
        detail::checked_collective(
            "SequenceParallelAttention::scatter_output", "ReduceScatter",
            [&]() {
                return detail::nccl_reduce_scatter<float>(
                    full_output.data(), local_output.data(), local_count,
                    dtype, ncclSum, static_cast<ncclComm_t>(config_.comm),
                    stream.get());
            },
            config_.comm, config_.nccl_context);
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
    detail::check_comm_healthy(
        "SequenceParallelAttention::all_reduce_sequence", config_.comm,
        config_.nccl_context);

    auto dtype = detail::to_nccl_dtype(float{});
    // Comm-error layer routing (see gather_kv for the rationale).
    detail::checked_collective(
        "SequenceParallelAttention::all_reduce_sequence", "AllReduce",
        [&]() {
            return detail::nccl_all_reduce<float>(
                data.data(), data.data(), data.size(), dtype,
                ncclSum, static_cast<ncclComm_t>(config_.comm), stream.get());
        },
        config_.comm, config_.nccl_context);
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
    // The group end is where an actual transport failure surfaces: per-op calls
    // inside a group only validate arguments and enqueue; the rendezvous errors
    // (dead peer / broken comm) are reported by ncclGroupEnd.
    const ncclResult_t group = ncclGroupEnd();
    if (r != ncclSuccess || r2 != ncclSuccess || r3 != ncclSuccess ||
        r4 != ncclSuccess) {
        // Immediately-rejected op = caller error (bad peer / count / dtype) —
        // preserve the per-op ncclResult_t for diagnosis, but do NOT poison:
        // a deliberate invalid-argument call must not break every later NCCL
        // user (same policy as safe_nccl_call's synchronous-error branch).
        throw std::runtime_error(
            "RingSequenceParallelism::send_recv_kv: NCCL send/recv rejected "
            "(ring KV exchange; send-K=" + std::to_string(r) +
            " recv-K=" + std::to_string(r2) +
            " send-V=" + std::to_string(r3) +
            " recv-V=" + std::to_string(r4) +
            " group=" + std::to_string(group) + ")");
    }
    if (group != ncclSuccess) {
        // The enqueued ring exchange failed en-route: the communicator is in an
        // indeterminate state and must never be reused — abort it and notify
        // the owning context (has_nccl() -> false) so later collectives fail
        // fast and the self-healing initialize() path can recover, instead of
        // the pre-TASK-018 behavior of throwing while leaving a poisoned comm
        // silently live for every subsequent user.
        cuda::nccl::poison_failed_comm(config_.comm, config_.nccl_context);
        throw std::runtime_error(
            "RingSequenceParallelism::send_recv_kv: NCCL ring KV group failed "
            "(comm aborted; group=" + std::to_string(group) + ")");
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
    detail::check_comm_healthy(
        "RingSequenceParallelism::ring_attention", config_.comm,
        config_.nccl_context);
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
