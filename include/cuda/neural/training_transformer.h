#pragma once

/**
 * @file training_transformer.h
 * @brief Deep multi-block transformer training driver on the parallel stack
 *        (v2.28)
 *
 * Milestone v2.28 (TASK-039, DEC-014): MicroTrainer (v2.25) trains a single
 * unnormalized MLP; v2.26/27 added TP attention and device optimizers but the
 * trainable model is still one unnormalized block (attention -> MLP). This
 * driver turns the v2.28 TransformerBlock into a reusable deep training loop:
 * it owns N pre-LN residual blocks, a final LayerNorm (the hidden == classes
 * "head"), one AdamW per weight tensor (11 per block + 2 final-LN), and the
 * device scratch; train_step() chains
 *
 *     y = LN_final( Block_N( ... Block_1(X) ... ) )
 *     dlogits -> CE-logits backward -> per-block backward -> per-tensor AdamW
 *
 * and returns the mean CE loss. The acceptance (like MicroTrainer) is real
 * learning: loss descent + accuracy rise on single-GPU, and a K-step multi-GPU
 * sharded run reproducing the single-GPU full-weight training trajectory
 * (weights AND loss curve) on 2 & 4 GPUs.
 */

#include "cuda/neural/tensor_parallel_block.h"
#include "cuda/neural/layer_norm.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/neural/loss/loss_functions.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <array>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

namespace cuda::neural::training {

/**
 * @class TransformerTrainer
 * @brief Reusable deep N-block transformer training loop (v2.28)
 *
 * Owns N TransformerBlocks + one final LayerNorm. Weights are uploaded per
 * block (set_block_weight) and for the final LN (set_final_ln_weight); one
 * AdamW per weight tensor (11N + 2) keeps moments private per tensor (v2.24
 * cpp-review convention). train_step() returns the host mean CE loss.
 */
class TransformerTrainer {
public:
    /**
     * @brief Construct the deep transformer trainer
     * @param ctx NCCL context (rank membership for the block shards; tp == 1
     *        when it reports <= 1 device)
     * @param num_blocks Number of TransformerBlocks in the stack (> 0)
     * @param hidden Hidden (input/output) dim == number of classes
     * @param heads Attention heads per block
     * @param head_dim Per-head dimension
     * @param intermediate_size FFN intermediate dim per block
     * @param opt_cfg AdamW config (one optimizer per weight tensor)
     * @throws std::invalid_argument on non-positive dims (sub-layer ctors
     *        reject non-divisible feature dims when tp > 1)
     */
    TransformerTrainer(
        ::cuda::nccl::NcclContext& ctx,
        int num_blocks,
        int hidden,
        int heads,
        int head_dim,
        int intermediate_size,
        const optimizers::OptimizerConfig& opt_cfg = optimizers::OptimizerConfig{});

    // Non-copyable/non-movable (owns device buffers + the block stack).
    TransformerTrainer(const TransformerTrainer&) = delete;
    TransformerTrainer& operator=(const TransformerTrainer&) = delete;
    TransformerTrainer(TransformerTrainer&&) = delete;
    TransformerTrainer& operator=(TransformerTrainer&&) = delete;

    ~TransformerTrainer();

    /**
     * @brief Upload block b's 11 weights (LN1 g/b, Wq/Wk/Wv/Wo, LN2 g/b,
     *        Wg/Wu/Wd), 0-indexed
     */
    void set_block_weight(int block, const float* ln1_gamma,
                          const float* ln1_beta, const float* wq,
                          const float* wk, const float* wv, const float* wo,
                          const float* ln2_gamma, const float* ln2_beta,
                          const float* wg, const float* wu, const float* wd);

    /**
     * @brief Upload the final LayerNorm gamma/beta (the classes head)
     */
    void set_final_ln_weight(const float* gamma, const float* beta);

    /**
     * @brief Run one gradient training step over a batch
     *
     * forward (N blocks + final LN) -> device cross_entropy_logits_backward
     * -> per-block backward -> per-tensor AdamW.
     *
     * @param input Device activations [batch*seq x hidden] (replicated)
     * @param targets Host labels [batch*seq] in [0, hidden)
     * @param batch Batch size
     * @param seq Sequence length
     * @param step_no Optimizer step counter (AdamW bias correction)
     * @return Mean cross-entropy loss over the batch (host scalar)
     */
    float train_step(const float* input, const int* targets, int batch, int seq,
                     int step_no);

    /**
     * @brief Forward + loss (optionally top-1 accuracy), no weight update
     * @param input Device activations [batch*seq x hidden]
     * @param targets Host labels [batch*seq]
     * @param batch Batch size
     * @param seq Sequence length
     * @param accuracy Optional output for top-1 accuracy in [0,1]
     * @return Mean cross-entropy loss (host scalar)
     */
    float evaluate(const float* input, const int* targets, int batch, int seq,
                   float* accuracy = nullptr);

    /**
     * @brief Copy block b's weights (per-rank shards) to host memory (parity)
     */
    void copy_block_weights(int block, float* ln1_gamma, float* ln1_beta,
                            float* wq, float* wk, float* wv, float* wo,
                            float* ln2_gamma, float* ln2_beta, float* wg,
                            float* wu, float* wd) const;

    /**
     * @brief Copy the final LN gamma/beta to host memory (parity)
     */
    void copy_final_ln(float* gamma, float* beta) const;

    /**
     * @brief Serialize the full trainer state to a binary stream (v2.32)
     *
     * Deterministic NSCK-v1 format: kind, dims, all 11N+2 weight tensors and
     * every optimizer's m/v moment pair. The optimizer step counter is NOT
     * part of the checkpoint — train_step() takes step_no from the caller —
     * so a restored trainer resumes exactly only if stepping continues at the
     * interrupted count (K+1 after a K-step run).
     */
    void save_state(std::ostream& out) const;

    /**
     * @brief Restore trainer state from a save_state() stream (v2.32)
     *
     * Read + validate every record (kind, dims, per-tensor sizes, moment
     * sizes) before applying anything: a geometry mismatch, corrupt magic or
     * truncated stream throws std::runtime_error and leaves this trainer
     * untouched (no partial restore). Only then are weights and moments set
     * (via set_block_weight/set_final_ln_weight/copy_moments_from), so a
     * successfully restored trainer resumes with byte-identical weights and
     * m/v — the caller continues step_no at the interrupted count.
     */
    void load_state(std::istream& in);

    /**
     * @brief Number of transformer blocks
     */
    [[nodiscard]] int num_blocks() const;

    /**
     * @brief Hidden (input/output) dim == number of classes
     */
    [[nodiscard]] int hidden() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    void ensure_scratch(int m);

    int num_blocks_ = 0;
    int hidden_ = 0;
    int heads_ = 0;
    int head_dim_ = 0;
    int intermediate_ = 0;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    std::unique_ptr<LayerNorm> final_ln_;  // classes head (hidden == classes)
    // One AdamW per weight tensor: 11 per block + 2 final-LN gamma/beta.
    std::vector<std::array<std::unique_ptr<optimizers::AdamWOptimizer>, 11>>
        block_opts_;
    std::array<std::unique_ptr<optimizers::AdamWOptimizer>, 2> final_opts_;
    // Device scratch sized to the current batch.
    std::vector<std::unique_ptr<cuda::memory::Buffer<float>>> h_bufs_;  // N block outputs
    std::unique_ptr<cuda::memory::Buffer<float>> d_logits_, d_dlogits_;  // final-LN out + grad
    std::vector<std::unique_ptr<cuda::memory::Buffer<float>>> grad_bufs_;  // N+1 block grad-inputs
    std::vector<std::array<std::unique_ptr<cuda::memory::Buffer<float>>, 11>>
        b_grads_;  // per-block full weight grads
    std::array<std::unique_ptr<cuda::memory::Buffer<float>>, 2> f_grads_;  // final-LN affine grads
    std::unique_ptr<cuda::memory::Buffer<int>> d_targets_;
    // Device scalar for the reported loss (v2.30: read ONE float back instead
    // of the whole logits buffer; also reused for the accuracy count).
    std::unique_ptr<cuda::memory::Buffer<float>> d_loss_;
    loss::CrossEntropyConfig ce_cfg_;
    int scratch_m_ = 0;
};

}  // namespace cuda::neural::training
