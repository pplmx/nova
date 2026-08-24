#pragma once

/**
 * @file training.h
 * @brief End-to-end gradient training driver on the tensor-parallel stack
 *
 * Milestone v2.25 (TASK-027, DEC-011): v2.24 proved the parallel layers take a
 * single exact training step (weight parity vs host fp64), but the stack had
 * never been shown to actually *learn* — no loss-descent, no accuracy, no
 * long-run multi-GPU trajectory check. MicroTrainer turns the verified
 * building blocks (forward -> device cross_entropy_logits_backward -> MLP
 * backward -> per-shard AdamW step, one optimizer per weight tensor) into a
 * reusable driver, and the acceptance is learning: loss descends to a
 * threshold, accuracy rises to the learnable-task ceiling, and a multi-GPU
 * sharded run reproduces the host fp64 full-weight training trajectory.
 *
 * Usage (classification task with hidden == num_classes):
 *   MicroTrainer trainer(ctx, hidden_dim, intermediate_size, opt_cfg);
 *   trainer.set_weight(gate, up, down);
 *   float loss = trainer.train_step(d_X, targets, batch, 1, step_no);
 *
 * The MLP is used as a classifier: its replicated output [batch*seq x hidden]
 * is the logits tensor and labels index into [0, hidden). This keeps the
 * driver self-contained (no separate head/LossFunction object required).
 */

#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/neural/loss/loss_functions.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cstddef>
#include <iosfwd>
#include <memory>

namespace cuda::neural::training {

/**
 * @class MicroTrainer
 * @brief Reusable training loop over a TensorParallelMLP (v2.25)
 *
 * Owns the MLP (with its three weight shards), one AdamW per weight tensor,
 * and the device scratch (logits / dlogits / grad buffers). train_step() runs
 * one full gradient step — forward, device cross-entropy-logits backward
 * (dlogits = (softmax - onehot)/B), MLP backward into the full grad buffers,
 * then per-shard AdamW — and returns the mean CE loss as a host scalar.
 *
 * The concrete learnable task (a fixed synthetic classification problem) lives
 * in the tests; the driver itself is task-agnostic given device activations +
 * host labels.
 */
class MicroTrainer {
public:
    /**
     * @brief Construct the trainer over a tensor-parallel MLP classifier
     * @param ctx NCCL context (rank membership for shards; tp == 1 when it
     *        reports <= 1 device)
     * @param hidden_dim Hidden (input/output) dim == number of classes
     * @param intermediate_size FFN intermediate dim
     * @param opt_cfg AdamW config (one optimizer per weight tensor)
     * @throws std::invalid_argument on non-positive dims (layer ctors reject
     *        non-divisible feature dims when tp > 1)
     */
    MicroTrainer(
        ::cuda::nccl::NcclContext& ctx,
        int hidden_dim,
        int intermediate_size,
        const optimizers::OptimizerConfig& opt_cfg = optimizers::OptimizerConfig{});

    // Non-copyable/non-movable (owns device buffers + the layer stack).
    MicroTrainer(const MicroTrainer&) = delete;
    MicroTrainer& operator=(const MicroTrainer&) = delete;
    MicroTrainer(MicroTrainer&&) = delete;
    MicroTrainer& operator=(MicroTrainer&&) = delete;

    ~MicroTrainer();

    /**
     * @brief Upload the full gate/up/down weights
     */
    void set_weight(const float* gate, const float* up, const float* down);

    /**
     * @brief Run one gradient training step over a batch
     *
     * forward -> device cross_entropy_logits_backward (dlogits) -> MLP
     * backward (full grad-weight buffers) -> per-shard AdamW step.
     *
     * @param input Device activations [batch*seq x hidden_dim] (replicated)
     * @param targets Host labels [batch*seq] in [0, hidden_dim)
     * @param batch Batch size
     * @param seq Sequence length
     * @param step_no Optimizer step counter (AdamW bias correction)
     * @return Mean cross-entropy loss over the batch (host scalar)
     */
    float train_step(const float* input, const int* targets, int batch, int seq,
                     int step_no);

    /**
     * @brief Forward + loss (optionally top-1 accuracy), no weight update
     * @param input Device activations [batch*seq x hidden_dim]
     * @param targets Host labels [batch*seq]
     * @param batch Batch size
     * @param seq Sequence length
     * @param accuracy Optional output for top-1 accuracy in [0,1]
     * @return Mean cross-entropy loss (host scalar)
     */
    float evaluate(const float* input, const int* targets, int batch, int seq,
                   float* accuracy = nullptr);

    /**
     * @brief Copy this rank's gate/up/down weight shards to host memory
     *        (parity verification)
     */
    void copy_weights(float* gate, float* up, float* down) const;

    /**
     * @brief Serialize the full trainer state to a binary stream (v2.32)
     *
     * Deterministic NSCK-v1 format: kind, dims, the three weight tensors, and
     * each optimizer's m/v moment pair. The optimizer step counter is NOT part
     * of the checkpoint — train_step() takes step_no from the caller — so a
     * restored trainer resumes exactly only if stepping continues at the
     * interrupted count (K+1 after a K-step run).
     */
    void save_state(std::ostream& out) const;

    /**
     * @brief Restore trainer state from a save_state() stream (v2.32)
     *
     * Read + validate every record (kind, TP degree, dims, per-tensor sizes,
     * moment sizes) before applying anything: a geometry (incl. TP-degree)
     * mismatch, corrupt magic or truncated stream throws std::runtime_error
     * and leaves this trainer untouched (no partial restore). Only then are
     * weights and moments set (via the shard setters / copy_moments_from), so
     * a successfully restored trainer resumes with byte-identical weights and
     * m/v — the caller continues step_no at the interrupted count. Records are
     * this rank's shards (full at tp == 1); the format stores the TP degree
     * (mismatch rejected) but NOT the rank id, so load this rank's OWN file
     * back — a same-topology file written by another rank matches every size
     * check and is not detected.
     */
    void load_state(std::istream& in);

    /**
     * @brief Hidden (input/output) dim == number of classes
     */
    [[nodiscard]] int hidden_dim() const;

    /**
     * @brief FFN intermediate dim
     */
    [[nodiscard]] int intermediate_size() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    void ensure_scratch(int m);

    int hidden_dim_ = 0;
    int intermediate_size_ = 0;
    std::unique_ptr<TensorParallelMLP> mlp_;
    std::unique_ptr<optimizers::AdamWOptimizer> opt_gate_;
    std::unique_ptr<optimizers::AdamWOptimizer> opt_up_;
    std::unique_ptr<optimizers::AdamWOptimizer> opt_down_;
    loss::CrossEntropyConfig ce_cfg_;
    // Scratch grows to the current batch.
    std::unique_ptr<cuda::memory::Buffer<float>> logits_;
    std::unique_ptr<cuda::memory::Buffer<float>> dlogits_;
    std::unique_ptr<cuda::memory::Buffer<float>> dX_;
    std::unique_ptr<cuda::memory::Buffer<float>> dWg_;
    std::unique_ptr<cuda::memory::Buffer<float>> dWu_;
    std::unique_ptr<cuda::memory::Buffer<float>> dWd_;
    std::unique_ptr<cuda::memory::Buffer<int>> d_targets_;
    // Device scalar for the reported loss (v2.30: read ONE float back instead
    // of the whole logits buffer; also reused for the accuracy count).
    std::unique_ptr<cuda::memory::Buffer<float>> d_loss_;
    int scratch_m_ = 0;
};

}  // namespace cuda::neural::training
