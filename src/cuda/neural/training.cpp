/**
 * @file training.cpp
 * @brief MicroTrainer — reusable end-to-end training driver (milestone v2.25)
 *
 * Milestone v2.25 (TASK-029 / DEC-011): v2.24 verified the per-layer
 * optimizer step and the device CE-logits backward in isolation. MicroTrainer
 * composes them into a real training loop: forward -> device
 * cross_entropy_logits_backward (dlogits = (softmax - onehot)/B) -> MLP
 * backward (full grad-weight buffers) -> per-shard AdamW step, with one
 * AdamW instance per weight tensor so each tensor's moments stay private.
 *
 * The driver returns the mean CE loss as a host scalar (recomputed from the
 * MLP's replicated logits via the host loss, matching the seed contract) so a
 * caller can observe loss descent without extra reduction kernels.
 */

#include "cuda/neural/training.h"

#include <stdexcept>
#include <vector>

namespace cuda::neural::training {

MicroTrainer::MicroTrainer(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int intermediate_size,
    const optimizers::OptimizerConfig& opt_cfg)
    : hidden_dim_(hidden_dim),
      intermediate_size_(intermediate_size),
      mlp_(std::make_unique<TensorParallelMLP>(ctx, hidden_dim, intermediate_size)),
      opt_gate_(std::make_unique<optimizers::AdamWOptimizer>(opt_cfg)),
      opt_up_(std::make_unique<optimizers::AdamWOptimizer>(opt_cfg)),
      opt_down_(std::make_unique<optimizers::AdamWOptimizer>(opt_cfg)) {
    ce_cfg_.num_classes = hidden_dim;
    ce_cfg_.reduction_mean = true;
}

MicroTrainer::~MicroTrainer() = default;

void MicroTrainer::set_weight(const float* gate, const float* up,
                              const float* down) {
    mlp_->set_weight(gate, up, down);
}

void MicroTrainer::ensure_scratch(int m) {
    const size_t needed = static_cast<size_t>(m) * hidden_dim_;
    if (logits_ != nullptr && static_cast<size_t>(scratch_m_) * hidden_dim_ >= needed) {
        return;
    }
    const size_t h = static_cast<size_t>(hidden_dim_);
    const size_t inter = static_cast<size_t>(intermediate_size_);
    logits_ = std::make_unique<cuda::memory::Buffer<float>>(needed);
    dlogits_ = std::make_unique<cuda::memory::Buffer<float>>(needed);
    dX_ = std::make_unique<cuda::memory::Buffer<float>>(needed);
    dWg_ = std::make_unique<cuda::memory::Buffer<float>>(h * inter);
    dWu_ = std::make_unique<cuda::memory::Buffer<float>>(h * inter);
    dWd_ = std::make_unique<cuda::memory::Buffer<float>>(inter * h);
    // Targets are copied per call (a label batch), sized to m.
    if (d_targets_ == nullptr ||
        static_cast<size_t>(scratch_m_) < static_cast<size_t>(m)) {
        d_targets_ = std::make_unique<cuda::memory::Buffer<int>>(m);
    }
    scratch_m_ = m;
}

float MicroTrainer::train_step(const float* input, const int* targets, int batch,
                               int seq, int step_no) {
    const int m = batch * seq;
    if (m <= 0) {
        throw std::invalid_argument(
            "MicroTrainer::train_step requires batch*seq > 0");
    }
    ensure_scratch(m);

    mlp_->forward(input, logits_->data(), batch, seq);
    d_targets_->copy_from(targets, static_cast<size_t>(m));
    loss::cross_entropy_logits_backward(
        logits_->data(), d_targets_->data(), dlogits_->data(),
        m, hidden_dim_, ce_cfg_, nullptr);
    mlp_->backward(input, dlogits_->data(), dX_->data(),
                   dWg_->data(), dWu_->data(), dWd_->data(), batch, seq);
    mlp_->step(*opt_gate_, *opt_up_, *opt_down_,
               dWg_->data(), dWu_->data(), dWd_->data(), step_no, nullptr);

    // Host scalar for the reported loss (matching the v2.24 seed contract).
    std::vector<float> h_logits(static_cast<size_t>(m) * hidden_dim_);
    logits_->copy_to(h_logits.data(), h_logits.size());
    std::vector<float> loss_out(static_cast<size_t>(m));
    return loss::cross_entropy_loss(
        h_logits.data(), targets, loss_out.data(), m, hidden_dim_,
        ce_cfg_, nullptr);
}

float MicroTrainer::evaluate(const float* input, const int* targets, int batch,
                             int seq, float* accuracy) {
    const int m = batch * seq;
    if (m <= 0) {
        throw std::invalid_argument(
            "MicroTrainer::evaluate requires batch*seq > 0");
    }
    ensure_scratch(m);
    mlp_->forward(input, logits_->data(), batch, seq);

    std::vector<float> h_logits(static_cast<size_t>(m) * hidden_dim_);
    logits_->copy_to(h_logits.data(), h_logits.size());
    std::vector<float> loss_out(static_cast<size_t>(m));
    float loss = loss::cross_entropy_loss(
        h_logits.data(), targets, loss_out.data(), m, hidden_dim_,
        ce_cfg_, nullptr);

    if (accuracy != nullptr) {
        int correct = 0;
        for (int i = 0; i < m; ++i) {
            const float* row = h_logits.data() + static_cast<size_t>(i) * hidden_dim_;
            int argmax = 0;
            for (int c = 1; c < hidden_dim_; ++c) {
                if (row[c] > row[argmax]) {
                    argmax = c;
                }
            }
            if (argmax == targets[i]) {
                ++correct;
            }
        }
        *accuracy = static_cast<float>(correct) / static_cast<float>(m);
    }
    return loss;
}

void MicroTrainer::copy_weights(float* gate, float* up, float* down) const {
    mlp_->copy_weights(gate, up, down);
}

int MicroTrainer::hidden_dim() const { return hidden_dim_; }

int MicroTrainer::intermediate_size() const { return intermediate_size_; }

int MicroTrainer::tp_degree() const { return mlp_->tp_degree(); }

}  // namespace cuda::neural::training
