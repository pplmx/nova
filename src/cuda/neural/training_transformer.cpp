/**
 * @file training_transformer.cpp
 * @brief TransformerTrainer — deep multi-block training driver (v2.28)
 *
 * Milestone v2.28 (TASK-041 / DEC-014): MicroTrainer (v2.25) trains a single
 * unnormalized MLP. This driver turns the v2.28 TransformerBlock into a
 * reusable deep learning loop: N pre-LN residual blocks + a final LayerNorm
 * (the hidden == classes head), one AdamW per weight tensor (11N + 2, so each
 * tensor's moments stay private per the v2.24 cpp-review convention).
 *
 * train_step() runs forward (N blocks + final LN) -> device
 * cross_entropy_logits_backward (dlogits = (softmax - onehot)/B) -> per-block
 * backward -> per-tensor AdamW, and returns the mean CE loss as a host scalar
 * (recomputed from the replicated final-LN logits, matching the MicroTrainer
 * seed contract).
 */

#include "cuda/neural/training_transformer.h"

#include <stdexcept>
#include <vector>

namespace cuda::neural::training {

namespace {

// 11 weight-grad tensor sizes per block, in the fixed order the block step /
// copy/backward signatures use: LN1 gamma/beta, Wq/Wk/Wv/Wo, LN2 gamma/beta,
// MLP gate/up/down.
inline int block_grad_size(int idx, int hidden, int qkv, int inter) {
    switch (idx) {
        case 0: return hidden;                    // ln1_gamma
        case 1: return hidden;                    // ln1_beta
        case 2: return hidden * qkv;              // wq
        case 3: return hidden * qkv;              // wk
        case 4: return hidden * qkv;              // wv
        case 5: return qkv * hidden;              // wo
        case 6: return hidden;                    // ln2_gamma
        case 7: return hidden;                    // ln2_beta
        case 8: return hidden * inter;            // wg
        case 9: return hidden * inter;            // wu
        case 10: return inter * hidden;           // wd
        default: return 0;
    }
}

}  // namespace

TransformerTrainer::TransformerTrainer(
    ::cuda::nccl::NcclContext& ctx,
    int num_blocks,
    int hidden,
    int heads,
    int head_dim,
    int intermediate_size,
    const optimizers::OptimizerConfig& opt_cfg)
    : num_blocks_(num_blocks),
      hidden_(hidden),
      heads_(heads),
      head_dim_(head_dim),
      intermediate_(intermediate_size) {
    if (num_blocks_ <= 0 || hidden_ <= 0 || heads_ <= 0 || head_dim_ <= 0 ||
        intermediate_ <= 0) {
        throw std::invalid_argument("TransformerTrainer: dims must be positive");
    }
    ce_cfg_.num_classes = hidden_;
    ce_cfg_.reduction_mean = true;
    blocks_.reserve(num_blocks_);
    block_opts_.reserve(num_blocks_);
    for (int b = 0; b < num_blocks_; ++b) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            ctx, hidden_, heads_, head_dim_, intermediate_));
        std::array<std::unique_ptr<optimizers::AdamWOptimizer>, 11> opts;
        for (auto& o : opts) {
            o = std::make_unique<optimizers::AdamWOptimizer>(opt_cfg);
        }
        block_opts_.push_back(std::move(opts));
    }
    final_ln_ = std::make_unique<LayerNorm>(hidden_);
    for (auto& o : final_opts_) {
        o = std::make_unique<optimizers::AdamWOptimizer>(opt_cfg);
    }
}

TransformerTrainer::~TransformerTrainer() = default;

void TransformerTrainer::ensure_scratch(int m) {
    if (scratch_m_ >= m && b_grads_.size() == static_cast<size_t>(num_blocks_)) {
        return;
    }
    const size_t needed = static_cast<size_t>(m) * hidden_;
    h_bufs_.clear();
    grad_bufs_.clear();
    for (int b = 0; b < num_blocks_; ++b) {
        h_bufs_.push_back(std::make_unique<cuda::memory::Buffer<float>>(needed));
    }
    for (int b = 0; b <= num_blocks_; ++b) {  // N+1: extra tail discards block-0 grad-input
        grad_bufs_.push_back(std::make_unique<cuda::memory::Buffer<float>>(needed));
    }
    d_logits_ = std::make_unique<cuda::memory::Buffer<float>>(needed);
    d_dlogits_ = std::make_unique<cuda::memory::Buffer<float>>(needed);

    // Device scalar for the reported loss (v2.30).
    if (d_loss_ == nullptr) {
        d_loss_ = std::make_unique<cuda::memory::Buffer<float>>(1);
    }

    const int qkv = heads_ * head_dim_;
    b_grads_.clear();
    for (int b = 0; b < num_blocks_; ++b) {
        std::array<std::unique_ptr<cuda::memory::Buffer<float>>, 11> grads;
        for (int i = 0; i < 11; ++i) {
            grads[i] = std::make_unique<cuda::memory::Buffer<float>>(
                block_grad_size(i, hidden_, qkv, intermediate_));
        }
        b_grads_.push_back(std::move(grads));
    }
    f_grads_[0] = std::make_unique<cuda::memory::Buffer<float>>(hidden_);
    f_grads_[1] = std::make_unique<cuda::memory::Buffer<float>>(hidden_);
    d_targets_ = std::make_unique<cuda::memory::Buffer<int>>(m);
    scratch_m_ = m;
}

void TransformerTrainer::set_block_weight(
    int block, const float* ln1_gamma, const float* ln1_beta, const float* wq,
    const float* wk, const float* wv, const float* wo, const float* ln2_gamma,
    const float* ln2_beta, const float* wg, const float* wu, const float* wd) {
    if (block < 0 || block >= num_blocks_) {
        throw std::invalid_argument("TransformerTrainer::set_block_weight: block out of range");
    }
    blocks_[static_cast<size_t>(block)]->set_weight(
        ln1_gamma, ln1_beta, wq, wk, wv, wo, ln2_gamma, ln2_beta, wg, wu, wd);
}

void TransformerTrainer::set_final_ln_weight(const float* gamma,
                                             const float* beta) {
    final_ln_->set_weight(gamma, beta);
}

float TransformerTrainer::train_step(const float* input, const int* targets,
                                     int batch, int seq, int step_no) {
    const int m = batch * seq;
    if (m <= 0) {
        throw std::invalid_argument(
            "TransformerTrainer::train_step requires batch*seq > 0");
    }
    ensure_scratch(m);

    // Forward: block chain -> final LN head (hidden == classes).
    const float* h_prev = input;
    for (int b = 0; b < num_blocks_; ++b) {
        blocks_[static_cast<size_t>(b)]->forward(h_prev, h_bufs_[static_cast<size_t>(b)]->data(),
                                                 batch, seq);
        h_prev = h_bufs_[static_cast<size_t>(b)]->data();
    }
    final_ln_->forward(h_prev, d_logits_->data(), m);
    d_targets_->copy_from(targets, static_cast<size_t>(m));
    loss::cross_entropy_logits_backward(
        d_logits_->data(), d_targets_->data(), d_dlogits_->data(), m, hidden_,
        ce_cfg_, nullptr);

    // Backward: final LN into block N-1's grad-input, then per-block chain.
    final_ln_->backward(h_prev, d_dlogits_->data(),
                        grad_bufs_[static_cast<size_t>(num_blocks_) - 1]->data(),
                        f_grads_[0]->data(), f_grads_[1]->data(), m);
    for (int b = num_blocks_ - 1; b >= 0; --b) {
        const float* b_in =
            (b > 0) ? h_bufs_[static_cast<size_t>(b) - 1]->data() : input;
        // Block b consumes grad_bufs_[b] (its grad-output) and writes its
        // grad-input to grad_bufs_[b-1], which is the next block's grad-output.
        // Block 0's model-input grad is discarded into the tail buffer [N].
        float* b_gi = (b > 0)
                          ? grad_bufs_[static_cast<size_t>(b) - 1]->data()
                          : grad_bufs_[static_cast<size_t>(num_blocks_)]->data();
        std::array<std::unique_ptr<cuda::memory::Buffer<float>>, 11>& g =
            b_grads_[static_cast<size_t>(b)];
        blocks_[static_cast<size_t>(b)]->backward(
            b_in, grad_bufs_[static_cast<size_t>(b)]->data(), b_gi,
            g[0]->data(), g[1]->data(), g[2]->data(), g[3]->data(), g[4]->data(),
            g[5]->data(), g[6]->data(), g[7]->data(), g[8]->data(), g[9]->data(),
            g[10]->data(), batch, seq);
    }

    // One AdamW step per weight tensor.
    final_ln_->step(*final_opts_[0], *final_opts_[1], f_grads_[0]->data(),
                    f_grads_[1]->data(), step_no);
    for (int b = 0; b < num_blocks_; ++b) {
        std::array<std::unique_ptr<optimizers::AdamWOptimizer>, 11>& o =
            block_opts_[static_cast<size_t>(b)];
        std::array<std::unique_ptr<cuda::memory::Buffer<float>>, 11>& g =
            b_grads_[static_cast<size_t>(b)];
        blocks_[static_cast<size_t>(b)]->step(
            *o[0], *o[1], *o[2], *o[3], *o[4], *o[5], *o[6], *o[7], *o[8],
            *o[9], *o[10], g[0]->data(), g[1]->data(), g[2]->data(),
            g[3]->data(), g[4]->data(), g[5]->data(), g[6]->data(), g[7]->data(),
            g[8]->data(), g[9]->data(), g[10]->data(), step_no);
    }

    // Device scalar for the reported loss (v2.30 / issue-v30-ce-loss-host-
    // roundtrip): the loss is reduced on-device and ONE float is read back,
    // instead of the whole m*hidden logits buffer + host loop. Same mean-CE
    // scalar as the host cross_entropy_loss (MicroTrainer seed contract).
    loss::cross_entropy_loss_device(
        d_logits_->data(), d_targets_->data(), d_loss_->data(),
        m, hidden_, ce_cfg_, nullptr);
    float dev_loss = 0.0f;
    d_loss_->copy_to(&dev_loss, 1);
    return dev_loss;
}

float TransformerTrainer::evaluate(const float* input, const int* targets,
                                   int batch, int seq, float* accuracy) {
    const int m = batch * seq;
    if (m <= 0) {
        throw std::invalid_argument(
            "TransformerTrainer::evaluate requires batch*seq > 0");
    }
    ensure_scratch(m);

    const float* h_prev = input;
    for (int b = 0; b < num_blocks_; ++b) {
        blocks_[static_cast<size_t>(b)]->forward(h_prev, h_bufs_[static_cast<size_t>(b)]->data(),
                                                 batch, seq);
        h_prev = h_bufs_[static_cast<size_t>(b)]->data();
    }
    final_ln_->forward(h_prev, d_logits_->data(), m);
    d_targets_->copy_from(targets, static_cast<size_t>(m));

    // Device loss + device top-1 accuracy (v2.30): the whole logits buffer
    // stays on device; only the two scalars come back.
    loss::cross_entropy_loss_device(
        d_logits_->data(), d_targets_->data(), d_loss_->data(),
        m, hidden_, ce_cfg_, nullptr);
    float loss = 0.0f;
    d_loss_->copy_to(&loss, 1);
    if (accuracy != nullptr) {
        loss::accuracy_device(
            d_logits_->data(), d_targets_->data(), d_loss_->data(),
            m, hidden_, nullptr);
        float correct = 0.0f;
        d_loss_->copy_to(&correct, 1);
        *accuracy = correct / static_cast<float>(m);
    }
    return loss;
}

void TransformerTrainer::copy_block_weights(
    int block, float* ln1_gamma, float* ln1_beta, float* wq, float* wk,
    float* wv, float* wo, float* ln2_gamma, float* ln2_beta, float* wg,
    float* wu, float* wd) const {
    if (block < 0 || block >= num_blocks_) {
        throw std::invalid_argument("TransformerTrainer::copy_block_weights: block out of range");
    }
    blocks_[static_cast<size_t>(block)]->copy_weights(
        ln1_gamma, ln1_beta, wq, wk, wv, wo, ln2_gamma, ln2_beta, wg, wu, wd);
}

void TransformerTrainer::copy_final_ln(float* gamma, float* beta) const {
    final_ln_->copy_weights(gamma, beta);
}

// v2.32 P1 stub — replaced in P2 (TASK-057).
void TransformerTrainer::save_state(std::ostream& out) const {
    (void)out;
    throw std::logic_error(
        "TransformerTrainer::save_state not implemented (v2.32 P1 stub)");
}

// v2.32 P1 stub — replaced in P2 (TASK-057).
void TransformerTrainer::load_state(std::istream& in) {
    (void)in;
    throw std::logic_error(
        "TransformerTrainer::load_state not implemented (v2.32 P1 stub)");
}

int TransformerTrainer::num_blocks() const { return num_blocks_; }
int TransformerTrainer::hidden() const { return hidden_; }
int TransformerTrainer::tp_degree() const {
    return blocks_.empty() ? 1 : blocks_[0]->tp_degree();
}

}  // namespace cuda::neural::training
