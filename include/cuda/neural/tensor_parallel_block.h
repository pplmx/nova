#pragma once

/**
 * @file tensor_parallel_block.h
 * @brief Pre-LN residual transformer block on the parallel stack (v2.28)
 *
 * Milestone v2.28 (TASK-039, DEC-014): the v2.21-27 stack trains exactly one
 * model — an unnormalized single block (attention -> MLP, logits =
 * MLP(attn(X))) — because the layer_norm module was forward-only (untrainable)
 * and no residual connection existed anywhere. A real transformer block is
 * pre-norm with residuals:
 *
 *     h  = x + MHA(LN1(x))
 *     y  = h + MLP(LN2(h))
 *
 * This class composes the verified pieces — LayerNorm (v2.28, trainable,
 * collective-free), TensorParallelMultiHeadAttention (v2.26) and
 * TensorParallelMLP (v2.21) — with residual adds (elementwise_add). All
 * activations are replicated [m x hidden] so normalization and residuals are
 * collective-free; the only communicator in the block stays the MHA output
 * projection's AllReduce (v2.26). Weights per block: LN1 gamma/beta, MHA
 * Wq/Wk/Wv/Wo, LN2 gamma/beta, MLP gate/up/down — one AdamW per tensor,
 * stepped by the caller (v2.24 one-optimizer-per-tensor convention).
 */

#include "cuda/neural/tensor_parallel_attention.h"
#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/layer_norm.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cstddef>
#include <memory>

namespace cuda::neural {

/**
 * @class TransformerBlock
 * @brief One pre-LN, residual, tensor-parallel transformer block (v2.28)
 *
 * forward():  y = x + MLP(LN2( x + MHA(LN1(x)) )). backward() fills the
 * caller's full grad-weight buffers (rank slices for the sharded MHA/MLP
 * weights; full [hidden] vectors for the replicated LN gamma/beta) and the
 * full AllReduced grad-input. step() applies one AdamW per weight tensor
 * (11 per block: 4 LN affine + 4 MHA + 3 MLP).
 */
class TransformerBlock {
public:
    /**
     * @brief Construct a pre-LN residual transformer block
     * @param ctx NCCL context (rank membership for the MHA/MLP shards;
     *        tp <= 1 -> plain single-shard block, no NCCL)
     * @param hidden Hidden (input/output) width
     * @param heads Number of attention heads (must divide evenly by tp)
     * @param head_dim Per-head dimension (heads*head_dim must divide by tp)
     * @param intermediate_size FFN intermediate dim
     * @param ln_eps LayerNorm stability epsilon
     * @throws std::invalid_argument on invalid dims (sub-layer ctors reject
     *        non-divisible feature dims when tp > 1)
     */
    TransformerBlock(
        ::cuda::nccl::NcclContext& ctx,
        int hidden,
        int heads,
        int head_dim,
        int intermediate_size,
        float ln_eps = 1e-5f);

    // Non-copyable/non-movable (owns device buffers + the layer sub-stack).
    TransformerBlock(const TransformerBlock&) = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;
    TransformerBlock(TransformerBlock&&) = delete;
    TransformerBlock& operator=(TransformerBlock&&) = delete;

    ~TransformerBlock();

    /**
     * @brief Upload the full block weights (all 11 tensors)
     * @param ln1_gamma/ln1_beta [hidden] (replicated)
     * @param wq/wk/wv [hidden x qkv] (column-parallel QKV)
     * @param wo [qkv x hidden] (row-parallel output)
     * @param ln2_gamma/ln2_beta [hidden] (replicated)
     * @param wg/wu [hidden x inter], wd [inter x hidden] (gated MLP)
     */
    void set_weight(const float* ln1_gamma, const float* ln1_beta,
                    const float* wq, const float* wk, const float* wv,
                    const float* wo, const float* ln2_gamma,
                    const float* ln2_beta, const float* wg, const float* wu,
                    const float* wd);

    /**
     * @brief Forward pass: y = x + MLP(LN2(x + MHA(LN1(x))))
     * @param input Device input [batch*seq x hidden] (replicated)
     * @param output Device output [batch*seq x hidden] (replicated)
     * @param batch Batch size
     * @param seq Sequence length
     */
    void forward(const float* input, float* output, int batch, int seq);

    /**
     * @brief Backward pass (synchronous)
     *
     * Residual backprop: output = h + m_out with h = x + a, so the direct
     * grad-input carries the residual dL/dh2 plus the LN1-path term.
     *
     * @param input Forward input [batch*seq x hidden] (replicated)
     * @param grad_output Upstream gradient [batch*seq x hidden] (replicated)
     * @param grad_input Full grad-input [batch*seq x hidden] (replicated)
     * @param dln1_gamma/dln1_beta Full [hidden] LN grads (sum over rows)
     * @param dwq/dwk/dwv Full [hidden x qkv], dwo Full [qkv x hidden]
     *        (rank writes its shard slice)
     * @param dln2_gamma/dln2_beta Full [hidden] LN grads (sum over rows)
     * @param dwg/dwu Full [hidden x inter], dwd Full [inter x hidden]
     * @param batch Batch size
     * @param seq Sequence length
     */
    void backward(
        const float* input,
        const float* grad_output,
        float* grad_input,
        float* dln1_gamma, float* dln1_beta,
        float* dwq, float* dwk, float* dwv, float* dwo,
        float* dln2_gamma, float* dln2_beta,
        float* dwg, float* dwu, float* dwd,
        int batch, int seq);

    /**
     * @brief Apply one AdamW step to every weight tensor of the block
     *        (11 optimizers; one per tensor)
     *
     * LN gamma/beta are replicated full-[hidden] vectors stepped with the
     * full (identical) grads on every rank; MHA/MLP shards step in place per
     * v2.24.
     *
     * @param o_l1g/o_l1b/o_q/o_k/o_v/o_o/o_l2g/o_l2b/o_g/o_u/o_d AdamW
     *        instances, one per weight tensor
     * @param dln1_gamma/dln1_beta/dwq/dwk/dwv/dwo/dln2_gamma/dln2_beta/
     *        dwg/dwu/dwd Full grads as produced by backward()
     * @param step_no Optimizer step counter
     * @param stream CUDA stream
     */
    void step(
        optimizers::AdamWOptimizer& o_l1g, optimizers::AdamWOptimizer& o_l1b,
        optimizers::AdamWOptimizer& o_q, optimizers::AdamWOptimizer& o_k,
        optimizers::AdamWOptimizer& o_v, optimizers::AdamWOptimizer& o_o,
        optimizers::AdamWOptimizer& o_l2g, optimizers::AdamWOptimizer& o_l2b,
        optimizers::AdamWOptimizer& o_g, optimizers::AdamWOptimizer& o_u,
        optimizers::AdamWOptimizer& o_d,
        const float* dln1_gamma, const float* dln1_beta,
        const float* dwq, const float* dwk, const float* dwv, const float* dwo,
        const float* dln2_gamma, const float* dln2_beta,
        const float* dwg, const float* dwu, const float* dwd,
        int step_no, cudaStream_t stream = nullptr);

    /**
     * @brief Copy this rank's block weights to host memory (parity)
     */
    void copy_weights(float* ln1_gamma, float* ln1_beta,
                      float* wq, float* wk, float* wv, float* wo,
                      float* ln2_gamma, float* ln2_beta,
                      float* wg, float* wu, float* wd) const;

    /**
     * @brief Hidden width
     */
    [[nodiscard]] int hidden() const;

    /**
     * @brief Attention heads
     */
    [[nodiscard]] int num_heads() const;

    /**
     * @brief FFN intermediate width
     */
    [[nodiscard]] int intermediate_size() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    void ensure_scratch(int m);

    int hidden_ = 0;
    int heads_ = 0;
    int head_dim_ = 0;
    int intermediate_ = 0;
    std::unique_ptr<LayerNorm> ln1_;                  // pre-attention norm
    std::unique_ptr<TensorParallelMultiHeadAttention> attn_;
    std::unique_ptr<LayerNorm> ln2_;                  // pre-MLP norm
    std::unique_ptr<TensorParallelMLP> mlp_;
    // Forward intermediates (kept for backward) sized to the current batch.
    std::unique_ptr<cuda::memory::Buffer<float>> h1_;   // LN1(x)
    std::unique_ptr<cuda::memory::Buffer<float>> a_;    // MHA(h1)
    std::unique_ptr<cuda::memory::Buffer<float>> h2_;   // x + a (LN2 input)
    std::unique_ptr<cuda::memory::Buffer<float>> h3_;   // LN2(h2) (MLP input)
    std::unique_ptr<cuda::memory::Buffer<float>> m_out_; // MLP(h3)
    // Backward scratch: MLP grad-input, LN2-path grad-input, residual sums,
    // attention grad-input, LN1-path grad-input (all [m x hidden]).
    std::unique_ptr<cuda::memory::Buffer<float>> d_mlp_gi_;
    std::unique_ptr<cuda::memory::Buffer<float>> d_r1ln_;
    std::unique_ptr<cuda::memory::Buffer<float>> d_h2_;
    std::unique_ptr<cuda::memory::Buffer<float>> d_attn_gi_;
    std::unique_ptr<cuda::memory::Buffer<float>> d_xln_;
    int scratch_m_ = 0;
};

}  // namespace cuda::neural
