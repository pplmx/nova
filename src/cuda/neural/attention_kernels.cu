/**
 * @file attention_kernels.cu
 * @brief Device multi-head SDPA kernels (milestone v2.29, TASK-044/045)
 *
 * Milestone v2.29 (DEC-015): the public sdpa_forward/sdpa_backward in
 * tensor_parallel_attention.cpp used to compute multi-head scaled-dot-product
 * attention ON THE HOST — full D2H copies, host loops over
 * m x local_heads x m x head_dim, H2D copies back — once per block per
 * train_step. That was the binding deep-transformer training-latency
 * constraint (Round-28 LEARN M3; the v2.26 header deferred the move). This
 * module is the exact port of the host algorithm to CUDA, behind the detail::
 * entry points declared in tensor_parallel_attention.h and wired into the
 * public sdpa_forward/sdpa_backward in P2 (mirroring optimizers_kernels.cu).
 *
 * Layout (matches the public API): q/k/v/dq/dk/dv/out are [m x local_heads x
 * head_dim], position-major then head then dim; key j sits at
 * [j*local_heads*head_dim + h*head_dim + d].
 *
 * Forward: one block per (position, head) query row. Per-row max-subtracted
 * softmax over the m keys via a block reduction (two passes: max, then the
 * exp-normalized weighted sum of V accumulated into a shared accumulator with
 * float atomics), so m is bounded only by the number of keys, not the block.
 *
 * Backward: one block per (position, head). The softmax probs P[j] and
 * dP[j] = dout . v[j] are stored to per-call scratch so dscore[j] =
 * P[j]*(dP[j] - sum_k P[k]dP[k])/sqrt(d) can use the cross-key weighted sum;
 * dV[j,:] (+= P[j] dout) and dK[j,:] (+= dscore[j] q) accumulate across query
 * positions with global atomics (dQ is per-query-row, so it reduces in shared
 * then writes once). The wrapper zeros dq/dk/dv first — the public API
 * overwrites them, exactly like the host vectors did.
 */

#include "cuda/neural/tensor_parallel_attention.h"

#include "cuda/device/error.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cmath>
#include <stdexcept>

namespace cuda::neural {

namespace {

constexpr int kSdpaBlock = 256;  // multiple of 32 (reductions use full masks)
constexpr float kMinusInf = -1e30f;

// Lane collapse of a per-thread sum/max across the block. `red` must hold at
// least blockDim/32 floats; the reduced value is valid in ALL threads on
// return (kSdpaBlock threads are always active, so no partial-warp UB).
__device__ float block_reduce_sum(float v, float* red) {
    constexpr unsigned kFull = 0xffffffffu;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFull, v, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float acc = (threadIdx.x < kSdpaBlock / 32) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(kFull, acc, o);
        if (lane == 0) red[0] = acc;
    }
    __syncthreads();
    return red[0];
}

__device__ float block_reduce_max(float v, float* red) {
    constexpr unsigned kFull = 0xffffffffu;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(kFull, v, o));
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float acc = (threadIdx.x < kSdpaBlock / 32) ? red[threadIdx.x] : kMinusInf;
        for (int o = 16; o > 0; o >>= 1) acc = fmaxf(acc, __shfl_down_sync(kFull, acc, o));
        if (lane == 0) red[0] = acc;
    }
    __syncthreads();
    return red[0];
}

// Score of key j against query row qr: q[p,h] . k[j,h] / sqrt(hd).
__device__ float key_score(const float* qr, const float* k, int j,
                           size_t pos_stride, size_t hoff, int hd, float inv) {
    const float* kj = k + static_cast<size_t>(j) * pos_stride + hoff;
    float s = 0.0f;
    for (int d = 0; d < hd; ++d) s += qr[d] * kj[d];
    return s * inv;
}

// out = softmax_j( q[p,h] . k[j,h] / sqrt(d) ) v[j,h] per (p,h).
// Dynamic shared memory: red[kSdpaBlock/32] + out_acc[hd]. Scores are
// recomputed per pass (a cheap hd-dot) instead of held in shared, so m is
// bounded only by the number of keys, not the block.
__global__ void sdpa_forward_kernel(const float* q, const float* k,
                                    const float* v, float* out, int m,
                                    int local_heads, int hd) {
    const int grid_row = blockIdx.x;
    const int head = grid_row % local_heads;
    const int pos = grid_row / local_heads;
    const float inv = 1.0f / sqrtf(static_cast<float>(hd));
    extern __shared__ float sh[];
    float* red = sh;
    float* out_acc = sh + kSdpaBlock / 32;
    const size_t pos_stride = static_cast<size_t>(local_heads) * hd;
    const size_t hoff = static_cast<size_t>(head) * hd;
    const float* qr = q + static_cast<size_t>(pos) * pos_stride + hoff;

    for (int d = threadIdx.x; d < hd; d += kSdpaBlock) out_acc[d] = 0.0f;
    __syncthreads();

    // Pass A: max score over all keys (each thread reduces its own keys, then
    // one block reduction).
    float max_s = kMinusInf;
    for (int j = threadIdx.x; j < m; j += kSdpaBlock) {
        max_s = fmaxf(max_s, key_score(qr, k, j, pos_stride, hoff, hd, inv));
    }
    max_s = block_reduce_max(max_s, red);

    // Pass B: sum_j exp(s-max) and accumulate exp(s-max) * v[j] into the
    // shared out accumulator (float atomicAdd ordering only perturbs the sum
    // at ~ulp level, inside the parity tolerance).
    float sum = 0.0f;
    for (int j = threadIdx.x; j < m; j += kSdpaBlock) {
        const float expv = __expf(key_score(qr, k, j, pos_stride, hoff, hd, inv) - max_s);
        sum += expv;
        const float* vj = v + static_cast<size_t>(j) * pos_stride + hoff;
        for (int d = 0; d < hd; ++d) atomicAdd(&out_acc[d], expv * vj[d]);
    }
    sum = block_reduce_sum(sum, red);

    for (int d = threadIdx.x; d < hd; d += kSdpaBlock) {
        out[static_cast<size_t>(pos) * pos_stride + hoff + d] = out_acc[d] / sum;
    }
}

// Analytic backward: writes dq/dk/dv (must be zeroed by the caller/wrapper —
// the public API overwrites them). No cross-pass scratch: P and dP are
// recomputed in the final pass (a cheap hd-dot per key) rather than stored,
// because they are query-row-specific and a shared scratch would be raced by
// concurrent (position, head) blocks. Shared: red[kSdpaBlock/32] + qacc[hd].
__global__ void sdpa_backward_kernel(const float* q, const float* k,
                                     const float* v, const float* dout,
                                     float* dq, float* dk, float* dv, int m,
                                     int local_heads, int hd) {
    const int grid_row = blockIdx.x;
    const int head = grid_row % local_heads;
    const int pos = grid_row / local_heads;
    const float inv = 1.0f / sqrtf(static_cast<float>(hd));
    extern __shared__ float sh[];
    float* red = sh;
    float* qacc = sh + kSdpaBlock / 32;
    const size_t pos_stride = static_cast<size_t>(local_heads) * hd;
    const size_t hoff = static_cast<size_t>(head) * hd;
    const float* qr = q + static_cast<size_t>(pos) * pos_stride + hoff;
    const float* doutr = dout + static_cast<size_t>(pos) * pos_stride + hoff;
    const float* kbase = k + hoff;  // key j row: kbase + j*pos_stride
    const float* vbase = v + hoff;  // value j row: vbase + j*pos_stride
    const float* kq = qr;

    for (int d = threadIdx.x; d < hd; d += kSdpaBlock) qacc[d] = 0.0f;
    __syncthreads();

    // Pass A: max score over keys.
    float max_s = kMinusInf;
    for (int j = threadIdx.x; j < m; j += kSdpaBlock) {
        max_s = fmaxf(max_s, key_score(qr, k, j, pos_stride, hoff, hd, inv));
    }
    max_s = block_reduce_max(max_s, red);

    // Pass B: esum = sum_j exp(s-max); wsum = sum_j exp(s-max)*dP[j] with
    // dP[j] = dout . v[j]. weighted = wsum/esum.
    float esum = 0.0f, wsum = 0.0f;
    for (int j = threadIdx.x; j < m; j += kSdpaBlock) {
        const float expv =
            __expf(key_score(qr, k, j, pos_stride, hoff, hd, inv) - max_s);
        esum += expv;
        const float* vj = vbase + static_cast<size_t>(j) * pos_stride;
        float dp = 0.0f;
        for (int d = 0; d < hd; ++d) dp += doutr[d] * vj[d];
        wsum += expv * dp;
    }
    esum = block_reduce_sum(esum, red);
    wsum = block_reduce_sum(wsum, red);
    const float inv_sum = 1.0f / esum;
    const float weighted = wsum * inv_sum;

    // Pass C: dscore[j] = P[j]*(dP[j]-weighted)*inv (P, dP recomputed);
    // dV/dK accumulate over query positions -> global atomics; dQ is
    // per-query-row, accumulates in shared then writes once.
    for (int j = threadIdx.x; j < m; j += kSdpaBlock) {
        const float expv =
            __expf(key_score(qr, k, j, pos_stride, hoff, hd, inv) - max_s);
        const float p = expv * inv_sum;
        const float* vj = vbase + static_cast<size_t>(j) * pos_stride;
        float dp = 0.0f;
        for (int d = 0; d < hd; ++d) dp += doutr[d] * vj[d];
        const float ds = p * (dp - weighted) * inv;
        const float* kj = kbase + static_cast<size_t>(j) * pos_stride;
        for (int d = 0; d < hd; ++d) atomicAdd(&qacc[d], ds * kj[d]);
        float* dvj = dv + static_cast<size_t>(j) * pos_stride + hoff;
        float* dkj = dk + static_cast<size_t>(j) * pos_stride + hoff;
        for (int d = 0; d < hd; ++d) {
            atomicAdd(&dvj[d], p * doutr[d]);
            atomicAdd(&dkj[d], ds * kq[d]);
        }
    }
    __syncthreads();
    float* dqout = dq + static_cast<size_t>(pos) * pos_stride + hoff;
    for (int d = threadIdx.x; d < hd; d += kSdpaBlock) dqout[d] = qacc[d];
}

void validate_sdpa_dims(int m, int local_heads, int head_dim, const char* fn) {
    if (m <= 0 || local_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument(std::string(fn) +
                                    " requires positive m, local_heads, head_dim");
    }
}

}  // namespace

namespace detail {

void sdpa_forward_device(const float* q, const float* k, const float* v,
                         float* out, int m, int local_heads, int head_dim,
                         cudaStream_t stream) {
    validate_sdpa_dims(m, local_heads, head_dim, "sdpa_forward_device");
    const int grid = m * local_heads;
    const int shared = (kSdpaBlock / 32 + kSdpaBlock + head_dim) * sizeof(float);
    sdpa_forward_kernel<<<grid, kSdpaBlock, shared, stream>>>(
        q, k, v, out, m, local_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void sdpa_backward_device(const float* q, const float* k, const float* v,
                          const float* dout, float* dq, float* dk, float* dv,
                          int m, int local_heads, int head_dim,
                          cudaStream_t stream) {
    validate_sdpa_dims(m, local_heads, head_dim, "sdpa_backward_device");
    const size_t n = static_cast<size_t>(m) * local_heads * head_dim;
    // dq/dk/dv are overwritten by the public API (the host vectors were
    // zero-init and accumulated); zero first so the cross-query atomic adds
    // accumulate onto a clean base.
    CUDA_CHECK(cudaMemsetAsync(dq, 0, n * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dk, 0, n * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dv, 0, n * sizeof(float), stream));
    const int grid = m * local_heads;
    const int shared = (kSdpaBlock / 32 + head_dim) * sizeof(float);
    sdpa_backward_kernel<<<grid, kSdpaBlock, shared, stream>>>(
        q, k, v, dout, dq, dk, dv, m, local_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace detail

}  // namespace cuda::neural
