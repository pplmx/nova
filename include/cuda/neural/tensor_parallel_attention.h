#pragma once

/**
 * @file tensor_parallel_attention.h
 * @brief Weight-sharded tensor-parallel Multi-Head Attention block (v2.26)
 *
 * Milestone v2.26 (TASK-031, DEC-012): the v2.21-25 stack can train a plain
 * MLP on the parallel path, but no attention exists there — the single-GPU
 * MultiHeadAttention (transformer/attention.h) is an incomplete shell
 * (issue-v24-mha-incomplete: it scales a buffer and returns without computing
 * QK^T/softmax/V). This block builds attention on the *verified* layer
 * conventions so a real transformer-block can be trained end-to-end:
 *
 * - Q/K/V projections: ColumnParallelLayer (hidden -> heads*head_dim). Each
 *   rank owns contiguous head columns (heads/tp per rank), so the forward
 *   produces the rank's local heads with NO cross-rank communication.
 * - Multi-head SDPA: per-head softmax(Q K^T / sqrt(d)) V over the full
 *   sequence. All heads of a rank are complete on that rank, so the attention
 *   math itself is local (only the output projection below is collective).
 * - Output projection: RowParallelLayer (heads*head_dim -> hidden): local
 *   partial + one block-wise AllReduce -> replicated [m x hidden] output.
 *
 * Backward follows the v2.23 pattern: the output row-layer backward produces
 * the local head-gradients (sharded); SDPA backward yields dQ/dK/dV (analytic
 * scaled-dot-product chain); each of the QKV column layers scatters its
 * column-slice grad-weight and AllReduces the (replicated-input) grad-input,
 * which are summed to the full block grad-input.
 */

#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/matmul.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cublas_v2.h>

#include <cstddef>
#include <memory>

namespace cuda::neural {

/**
 * @class TensorParallelMultiHeadAttention
 * @brief Weight-sharded multi-head self-attention block for transformers
 *
 * Rank r owns the Q/K/V column shards (heads [r*heads/tp, (r+1)*heads/tp))
 * and the output row shard. forward() computes the replicated [m x hidden]
 * attention output; backward() fills the caller's full grad-weight buffers
 * (rank slices) and the full AllReduced grad-input. Composes with
 * TensorParallelMLP to form a trainable transformer block.
 */
class TensorParallelMultiHeadAttention {
public:
    /**
     * @brief Construct the attention block
     * @param ctx NCCL context (rank membership for shards; tp <= 1 -> the
     *        block is a plain single-shard attention, no NCCL)
     * @param num_heads Number of heads (must divide evenly by tp when tp > 1)
     * @param head_dim Per-head dimension
     * @param hidden_dim Input/output hidden width (num_heads*head_dim must
     *        divide evenly by tp for the sharded QKV/output projections)
     * @throws std::invalid_argument when a feature dim is not divisible by tp
     */
    TensorParallelMultiHeadAttention(
        ::cuda::nccl::NcclContext& ctx,
        int num_heads,
        int head_dim,
        int hidden_dim);

    // Non-copyable/non-movable (owns device buffers + the layer sub-stack).
    TensorParallelMultiHeadAttention(const TensorParallelMultiHeadAttention&) = delete;
    TensorParallelMultiHeadAttention& operator=(const TensorParallelMultiHeadAttention&) = delete;
    TensorParallelMultiHeadAttention(TensorParallelMultiHeadAttention&&) = delete;
    TensorParallelMultiHeadAttention& operator=(TensorParallelMultiHeadAttention&&) = delete;

    ~TensorParallelMultiHeadAttention();

    /**
     * @brief Upload the full Q/K/V/output weights
     * @param wq [hidden x heads*head_dim] row-major
     * @param wk [hidden x heads*head_dim] row-major
     * @param wv [hidden x heads*head_dim] row-major
     * @param wo [heads*head_dim x hidden] row-major
     */
    void set_weight(const float* wq, const float* wk, const float* wv,
                    const float* wo);

    /**
     * @brief Upload this rank's Q/K/V/output weight shards directly (v2.33)
     *
     * The no-slice dual of copy_weights: wq/wk/wv column shards
     * [hidden x local_heads*head_dim], wo row shard
     * [local_heads*head_dim x hidden] (== the full weights at tp == 1).
     */
    void set_weight_shards(const float* wq, const float* wk, const float* wv,
                           const float* wo);

    /**
     * @brief Forward pass (synchronous)
     * @param input Device input [batch*seq x hidden_dim] (replicated)
     * @param output Device output [batch*seq x hidden_dim] (replicated)
     * @param batch Batch size
     * @param seq Sequence length
     */
    void forward(const float* input, float* output, int batch, int seq);

    /**
     * @brief Backward pass (synchronous)
     * @param input Forward input [batch*seq x hidden] (replicated)
     * @param grad_output Upstream gradient [batch*seq x hidden] (replicated)
     * @param grad_input Full grad-input [batch*seq x hidden] (replicated,
     *        AllReduced across ranks)
     * @param grad_wq/grad_wk/grad_wv Full [hidden x heads*head_dim] (rank
     *        writes its column shard slice)
     * @param grad_wo Full [heads*head_dim x hidden] (rank writes its row
     *        shard slice)
     * @param batch Batch size
     * @param seq Sequence length
     */
    void backward(
        const float* input,
        const float* grad_output,
        float* grad_input,
        float* grad_wq,
        float* grad_wk,
        float* grad_wv,
        float* grad_wo,
        int batch,
        int seq);

    /**
     * @brief Apply one optimizer step to the Q/K/V/output weight shards
     *        (v2.26)
     *
     * One AdamW per weight tensor: each projection's m/v moments must belong
     * to exactly one weight (the per-tensor-optimizer convention from the
     * v2.24 cpp-review). Each rank extracts its shard slice from the caller's
     * full grad-weight buffers (column slices for Q/K/V, row block for Wo)
     * and steps its private shard in place.
     *
     * @param opt_q/opt_k/opt_v/opt_o AdamW instances (one per weight tensor)
     * @param grad_wq/grad_wk/grad_wv Full [hidden x qkv], as from backward()
     * @param grad_wo Full [qkv x hidden]
     * @param step_no Optimizer step counter
     * @param stream CUDA stream
     */
    void step(
        optimizers::AdamWOptimizer& opt_q, optimizers::AdamWOptimizer& opt_k,
        optimizers::AdamWOptimizer& opt_v, optimizers::AdamWOptimizer& opt_o,
        const float* grad_wq, const float* grad_wk, const float* grad_wv,
        const float* grad_wo, int step_no, cudaStream_t stream = nullptr);

    /**
     * @brief Copy this rank's Q/K/V/output weight shards to host memory
     */
    void copy_weights(float* wq, float* wk, float* wv, float* wo) const;

    /**
     * @brief Number of heads (full)
     */
    [[nodiscard]] int num_heads() const;

    /**
     * @brief Per-head dimension
     */
    [[nodiscard]] int head_dim() const;

    /**
     * @brief Hidden (input/output) width
     */
    [[nodiscard]] int hidden_dim() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    void ensure_scratch(int m);

    int num_heads_ = 0;
    int head_dim_ = 0;
    int hidden_dim_ = 0;
    std::unique_ptr<ColumnParallelLayer> q_proj_;   // hidden -> heads*head_dim
    std::unique_ptr<ColumnParallelLayer> k_proj_;
    std::unique_ptr<ColumnParallelLayer> v_proj_;
    std::unique_ptr<RowParallelLayer> out_proj_;    // heads*head_dim -> hidden
    // Scratch: per-activation buffers sized to the current batch.
    std::unique_ptr<cuda::memory::Buffer<float>> q_buf_, k_buf_, v_buf_;
    std::unique_ptr<cuda::memory::Buffer<float>> qkv_out_;
    int scratch_m_ = 0;
};

/**
 * @brief Local multi-head scaled-dot-product self-attention on one rank's
 *        head shard (v2.26)
 *
 * Computes out = softmax(Q K^T / sqrt(d)) V for the rank-local heads, and its
 * analytic backward. Called by TensorParallelMultiHeadAttention with the
 * per-head [m x heads_local x head_dim] buffers. Exposed for reference testing.
 *
 * @param q [m x local_heads*head_dim] (rank-local Q shard)
 * @param k [m x local_heads*head_dim]
 * @param v [m x local_heads*head_dim]
 * @param out [m x local_heads*head_dim]
 * @param dout Upstream grad [m x local_heads*head_dim]
 * @param dq/dk/dv Output grads [m x local_heads*head_dim]
 * @param m Sequence/batch-elements (self-attention over m keys)
 * @param local_heads Heads owned by this rank
 * @param head_dim Per-head dimension
 * @param stream CUDA stream
 */
void sdpa_forward(const float* q, const float* k, const float* v, float* out,
                  int m, int local_heads, int head_dim,
                  cudaStream_t stream = nullptr);

void sdpa_backward(const float* q, const float* k, const float* v,
                   const float* out, const float* dout, float* dq, float* dk,
                   float* dv, int m, int local_heads, int head_dim,
                   cudaStream_t stream = nullptr);

namespace detail {
// v2.29 device multi-head SDPA (TASK-044/045): the GPU kernels that replace
// the host D2H/H2D round-trip in sdpa_forward/sdpa_backward (the v2.26 header
// deferred them to 'a later performance pass'; Round-28 LEARN M3 named the
// per-block host round-trip the binding deep-trainer latency). Same semantics
// and layout as the public functions — q/k/v/dq/dk/dv/out are [m x
// local_heads*head_dim] device buffers, self-attention over m keys, dims must
// be positive. This is the P1 RED contract surface: tested directly against
// fp64 references in attention_kernels_test.cpp, then wired into the public
// API in P2 (attention_kernels.cu).
void sdpa_forward_device(const float* q, const float* k, const float* v,
                         float* out, int m, int local_heads, int head_dim,
                         cudaStream_t stream = nullptr);

void sdpa_backward_device(const float* q, const float* k, const float* v,
                          const float* dout, float* dq, float* dk, float* dv,
                          int m, int local_heads, int head_dim,
                          cudaStream_t stream = nullptr);
}  // namespace detail

}  // namespace cuda::neural
