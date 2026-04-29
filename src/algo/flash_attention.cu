#include "cuda/algo/flash_attention.h"
#include "cuda/device/error.h"
#include "cuda/stream/stream.h"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <algorithm>

namespace cuda::algo {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE = 64;
constexpr int THREADS_PER_BLOCK = 256;

inline dim3 get_grid_dim(int total_blocks) {
    return dim3((total_blocks + 255) / 256);
}

inline dim3 get_block_dim() {
    return dim3(THREADS_PER_BLOCK);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <bool因果, bool IsFP16>
__global__ void flash_attention_fwd_kernel(
    const void* __restrict__ q_ptr,
    const void* __restrict__ k_ptr,
    const void* __restrict__ v_ptr,
    void* __restrict__ output_ptr,
    float* __restrict__ softmax_lse_ptr,
    float scale,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int q_stride_h,
    int k_stride_h,
    int v_stride_h,
    int softmax_lse_stride,
    bool causal,
    uint64_t dropout_seed,
    uint64_t dropout_offset,
    float dropout_scale
) {
    const int bid = blockIdx.x / num_heads;
    const int h = blockIdx.x % num_heads;

    if (bid >= batch_size || h >= num_heads) return;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int h_kv = h / kv_head_ratio;

    const int thread_idx = threadIdx.x;
    const int head_dim_v = head_dim / WARP_SIZE;

    extern __shared__ float shared[];
    float* sdata = shared;

    const void* q_h = reinterpret_cast<const char*>(q_ptr) + bid * seq_len * q_stride_h + h * head_dim;
    const void* k_h = reinterpret_cast<const char*>(k_ptr) + bid * seq_len * k_stride_h + h_kv * head_dim;
    const void* v_h = reinterpret_cast<const char*>(v_ptr) + bid * seq_len * v_stride_h + h_kv * head_dim;

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float q_local[TILE_SIZE / WARP_SIZE];

    if constexpr (IsFP16) {
        const __half* q_half = reinterpret_cast<const __half*>(q_h);
        #pragma unroll
        for (int i = 0; i < TILE_SIZE / WARP_SIZE; ++i) {
            q_local[i] = __half2float(q_half[thread_idx + i * WARP_SIZE]);
        }
    } else {
        const float* q_float = reinterpret_cast<const float*>(q_h);
        #pragma unroll
        for (int i = 0; i < TILE_SIZE / WARP_SIZE; ++i) {
            q_local[i] = q_float[thread_idx + i * WARP_SIZE];
        }
    }

    #pragma unroll
    for (int i = 0; i < TILE_SIZE / WARP_SIZE; ++i) {
        sdata[thread_idx + i * WARP_SIZE] = q_local[i];
    }
    __syncthreads();

    float block_max = warp_reduce_max(thread_max);
    __shared__ float max_shared;
    if (threadIdx.x == 0) max_shared = block_max;
    __syncthreads();
    block_max = max_shared;

    thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / WARP_SIZE; ++i) {
        thread_sum += expf(q_local[i] - block_max);
    }
    float block_sum = warp_reduce_sum(thread_sum);
    __shared__ float sum_shared;
    if (threadIdx.x == 0) sum_shared = block_sum;
    __syncthreads();
    block_sum = sum_shared;

    if (threadIdx.x == 0 && softmax_lse_ptr != nullptr) {
        softmax_lse_ptr[bid * softmax_lse_stride + h] = block_max + logf(block_sum);
    }

    float* out_h = reinterpret_cast<float*>(output_ptr) + bid * seq_len * q_stride_h + h * head_dim + thread_idx;
    *out_h = expf(q_local[0] - block_max) / block_sum;
}

template <bool因果, bool IsFP16>
__global__ void flash_attention_bwd_kernel(
    const void* __restrict__ q_ptr,
    const void* __restrict__ k_ptr,
    const void* __restrict__ v_ptr,
    const void* __restrict__ dout_ptr,
    const float* __restrict__ softmax_lse_ptr,
    void* __restrict__ dq_ptr,
    void* __restrict__ dk_ptr,
    void* __restrict__ dv_ptr,
    float scale,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int q_stride_h,
    int k_stride_h,
    int v_stride_h,
    uint64_t dropout_seed,
    uint64_t dropout_offset,
    float dropout_scale
) {
    const int bid = blockIdx.x / num_heads;
    const int h = blockIdx.x % num_heads;

    if (bid >= batch_size || h >= num_heads) return;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int h_kv = h / kv_head_ratio;

    const int thread_idx = threadIdx.x;

    float* dq_h = reinterpret_cast<float*>(dq_ptr) + bid * seq_len * q_stride_h + h * head_dim + thread_idx;
    float* dk_h = reinterpret_cast<float*>(dk_ptr) + bid * seq_len * k_stride_h + h_kv * head_dim + thread_idx;
    float* dv_h = reinterpret_cast<float*>(dv_ptr) + bid * seq_len * v_stride_h + h_kv * head_dim + thread_idx;

    float dq_local = 0.0f;

    float lse = softmax_lse_ptr[bid * seq_len * num_heads + h];
    float p_max = -INFINITY;

    float p_normalized = expf(0.0f - p_max) / 1.0f;
    float dout_val = 0.0f;

    dq_local = (p_normalized - 1.0f) * dout_val;

    *dq_h = dq_local;
}

}  // anonymous namespace

FlashAttention::FlashAttention(const FlashAttentionConfig& config)
    : config_(config) {
    if (config_.dropout_seed == 0) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;
        config_.dropout_seed = dis(gen);
    }
}

FlashAttention::~FlashAttention() = default;

size_t FlashAttention::get_workspace_size() const {
    const size_t elements_per_head = config_.seq_len * config_.head_dim;
    const size_t total_elements = elements_per_head * config_.num_heads;
    return total_elements * sizeof(float);
}

void FlashAttention::ensure_workspace(size_t bytes) {
    if (workspace_.size() < bytes) {
        workspace_ = memory::Buffer<void>(bytes);
    }
}

void FlashAttention::set_dropout(float rate, uint64_t seed) {
    config_.dropout_rate = rate;
    if (seed != 0) {
        config_.dropout_seed = seed;
    }
}

void FlashAttention::forward(
    memory::Buffer<float>& output,
    memory::Buffer<float>& softmax_lse,
    const memory::Buffer<float>& query,
    const memory::Buffer<float>& key,
    const memory::Buffer<float>& value,
    const stream::Stream& stream
) {
    ensure_workspace(get_workspace_size());

    const float scale = 1.0f / sqrtf(static_cast<float>(config_.head_dim));
    const float dropout_scale = config_.dropout_rate > 0.0f ? (1.0f / (1.0f - config_.dropout_rate)) : 0.0f;

    const int total_blocks = config_.batch_size * config_.num_heads;
    const int smem_size = TILE_SIZE * sizeof(float);

    flash_attention_kernel::AttentionParams params;
    params.q_ptr = query.data();
    params.k_ptr = key.data();
    params.v_ptr = value.data();
    params.output_ptr = output.data();
    params.softmax_lse_ptr = softmax_lse.data();
    params.scale = scale;
    params.batch_size = config_.batch_size;
    params.num_heads = config_.num_heads;
    params.num_kv_heads = config_.num_kv_heads;
    params.head_dim = config_.head_dim;
    params.seq_len = config_.seq_len;
    params.q_stride_h = config_.num_heads * config_.head_dim;
    params.k_stride_h = config_.num_kv_heads * config_.head_dim;
    params.v_stride_h = config_.num_kv_heads * config_.head_dim;
    params.softmax_lse_stride = config_.num_heads;
    params.causal = config_.causal;
    params.is_fp16 = false;
    params.dropout_seed = config_.dropout_seed;
    params.dropout_offset = config_.dropout_offset;
    params.dropout_scale = dropout_scale;

    if (config_.causal) {
        flash_attention_fwd_kernel<true, false><<<get_grid_dim(total_blocks), get_block_dim(), smem_size, stream.get()>>>(
            params.q_ptr, params.k_ptr, params.v_ptr, params.output_ptr,
            params.softmax_lse_ptr, params.scale, params.batch_size,
            params.num_heads, params.num_kv_heads, params.head_dim,
            params.seq_len, params.q_stride_h, params.k_stride_h,
            params.v_stride_h, params.softmax_lse_stride, params.causal,
            params.dropout_seed, params.dropout_offset, params.dropout_scale
        );
    } else {
        flash_attention_fwd_kernel<false, false><<<get_grid_dim(total_blocks), get_block_dim(), smem_size, stream.get()>>>(
            params.q_ptr, params.k_ptr, params.v_ptr, params.output_ptr,
            params.softmax_lse_ptr, params.scale, params.batch_size,
            params.num_heads, params.num_kv_heads, params.head_dim,
            params.seq_len, params.q_stride_h, params.k_stride_h,
            params.v_stride_h, params.softmax_lse_stride, params.causal,
            params.dropout_seed, params.dropout_offset, params.dropout_scale
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

void FlashAttention::forward_bf16(
    memory::Buffer<void>& output,
    memory::Buffer<float>& softmax_lse,
    const memory::Buffer<void>& query,
    const memory::Buffer<void>& key,
    const memory::Buffer<void>& value,
    const stream::Stream& stream
) {
    ensure_workspace(get_workspace_size());
}

void FlashAttention::backward(
    memory::Buffer<float>& dq,
    memory::Buffer<float>& dk,
    memory::Buffer<float>& dv,
    const memory::Buffer<float>& output,
    const memory::Buffer<float>& dout,
    const memory::Buffer<float>& query,
    const memory::Buffer<float>& key,
    const memory::Buffer<float>& value,
    const memory::Buffer<float>& softmax_lse,
    const stream::Stream& stream
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(config_.head_dim));
    const float dropout_scale = config_.dropout_rate > 0.0f ? (1.0f / (1.0f - config_.dropout_rate)) : 0.0f;

    const int total_blocks = config_.batch_size * config_.num_heads;

    if (config_.causal) {
        flash_attention_bwd_kernel<true, false><<<get_grid_dim(total_blocks), get_block_dim(), 0, stream.get()>>>(
            query.data(), key.data(), value.data(), dout.data(),
            softmax_lse.data(), dq.data(), dk.data(), dv.data(),
            scale, config_.batch_size, config_.num_heads,
            config_.num_kv_heads, config_.head_dim, config_.seq_len,
            config_.num_heads * config_.head_dim,
            config_.num_kv_heads * config_.head_dim,
            config_.num_kv_heads * config_.head_dim,
            config_.dropout_seed, config_.dropout_offset, dropout_scale
        );
    } else {
        flash_attention_bwd_kernel<false, false><<<get_grid_dim(total_blocks), get_block_dim(), 0, stream.get()>>>(
            query.data(), key.data(), value.data(), dout.data(),
            softmax_lse.data(), dq.data(), dk.data(), dv.data(),
            scale, config_.batch_size, config_.num_heads,
            config_.num_kv_heads, config_.head_dim, config_.seq_len,
            config_.num_heads * config_.head_dim,
            config_.num_kv_heads * config_.head_dim,
            config_.num_kv_heads * config_.head_dim,
            config_.dropout_seed, config_.dropout_offset, dropout_scale
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

std::unique_ptr<FlashAttention> create_flash_attention(
    const FlashAttentionConfig& config
) {
    return std::make_unique<FlashAttention>(config);
}

}  // namespace cuda::algo
