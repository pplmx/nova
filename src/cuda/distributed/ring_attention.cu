/**
 * @file ring_attention.cu
 * @brief Device kernels for RingSequenceParallelism multi-GPU ring attention
 *
 * Milestone v2.22 (TASK-016): the ring algorithm walks P-1 remote KV blocks
 * (sent clockwise / received counter-clockwise around the ring via NCCL P2P)
 * and accumulates partial attention over each block with the standard online
 * softmax (running max m_i + running denominator l_i), so the final output is
 * exactly the local queries' attention over the FULL KV sequence.
 *
 * The per-block kernel is deliberately simple: one thread per query token,
 * sequentially accumulating over that block's key rows with the rescale-on-
 * max-update rule. It is the reference-correct first implementation; a
 * tiled/reduction kernel is a performance follow-up, not a correctness one.
 *
 * Three host entry points (declared in sequence_parallel.h, consumed by both
 * the header's ring_attention orchestration and any .cpp TU):
 *  - ring_attn_init:      zero the accumulation / running max=-inf / l=0 state
 *  - ring_attn_block:     attend one KV block into the running state
 *  - ring_attn_finalize:  divide the accumulation by l_i
 */

#include "cuda/distributed/sequence_parallel.h"
#include "cuda/device/error.h"

#include <cmath>

namespace cuda::distributed {

namespace {

__global__ void ring_attn_init_kernel(
    float* out,
    float* running_max,
    float* running_l,
    int local_seq,
    int hidden
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_seq) {
        return;
    }
    float* o = out + static_cast<size_t>(i) * hidden;
    for (int c = 0; c < hidden; ++c) {
        o[c] = 0.0f;
    }
    running_max[i] = -INFINITY;
    running_l[i] = 0.0f;
}

__global__ void ring_attn_block_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float* running_max,
    float* running_l,
    int local_seq,
    int keys,
    int hidden,
    float inv_sqrt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_seq) {
        return;
    }
    const float* qi = q + static_cast<size_t>(i) * hidden;
    float* o = out + static_cast<size_t>(i) * hidden;
    float m = running_max[i];
    float l = running_l[i];

    for (int j = 0; j < keys; ++j) {
        const float* kj = k + static_cast<size_t>(j) * hidden;
        float s = 0.0f;
        for (int c = 0; c < hidden; ++c) {
            s += qi[c] * kj[c];
        }
        s *= inv_sqrt;

        // Online softmax: rescale the running accumulation on a max update.
        if (m < s) {
            const float corr = __expf(m - s);
            for (int c = 0; c < hidden; ++c) {
                o[c] *= corr;
            }
            l *= corr;
            m = s;
        }
        const float p = __expf(s - m);
        l += p;
        const float* vj = v + static_cast<size_t>(j) * hidden;
        for (int c = 0; c < hidden; ++c) {
            o[c] += p * vj[c];
        }
    }

    running_max[i] = m;
    running_l[i] = l;
}

__global__ void ring_attn_finalize_kernel(
    float* out,
    const float* running_l,
    int local_seq,
    int hidden
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_seq) {
        return;
    }
    const float inv = 1.0f / running_l[i];
    float* o = out + static_cast<size_t>(i) * hidden;
    for (int c = 0; c < hidden; ++c) {
        o[c] *= inv;
    }
}

}  // namespace

void ring_attn_init(
    float* out,
    float* running_max,
    float* running_l,
    int local_seq,
    int hidden,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (local_seq + threads - 1) / threads;
    ring_attn_init_kernel<<<blocks, threads, 0, stream>>>(
        out, running_max, running_l, local_seq, hidden);
    CUDA_CHECK(cudaGetLastError());
}

void ring_attn_block(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float* running_max,
    float* running_l,
    int local_seq,
    int keys,
    int hidden,
    float inv_sqrt,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (local_seq + threads - 1) / threads;
    if (blocks == 0) {
        return;
    }
    ring_attn_block_kernel<<<blocks, threads, 0, stream>>>(
        q, k, v, out, running_max, running_l, local_seq, keys, hidden, inv_sqrt);
    CUDA_CHECK(cudaGetLastError());
}

void ring_attn_finalize(
    float* out,
    const float* running_l,
    int local_seq,
    int hidden,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (local_seq + threads - 1) / threads;
    if (blocks == 0) {
        return;
    }
    ring_attn_finalize_kernel<<<blocks, threads, 0, stream>>>(
        out, running_l, local_seq, hidden);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda::distributed
