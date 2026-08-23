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

#include "checkpoint_io.h"

#include <stdexcept>
#include <string>
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
    // Device scalar for the reported loss (v2.30: compute the mean CE on
    // device and read ONE float back instead of the whole logits buffer).
    if (d_loss_ == nullptr) {
        d_loss_ = std::make_unique<cuda::memory::Buffer<float>>(1);
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

    // Device scalar for the reported loss (v2.30 / issue-v30-ce-loss-host-
    // roundtrip): the loss is reduced on-device and ONE float is read back,
    // instead of the whole m*hidden logits buffer + host loop. Same mean-CE
    // scalar as the host cross_entropy_loss (fp64 parity in the P1 tests).
    loss::cross_entropy_loss_device(
        logits_->data(), d_targets_->data(), d_loss_->data(),
        m, hidden_dim_, ce_cfg_, nullptr);
    float dev_loss = 0.0f;
    d_loss_->copy_to(&dev_loss, 1);
    return dev_loss;
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
    d_targets_->copy_from(targets, static_cast<size_t>(m));

    // Device loss + device top-1 accuracy (v2.30): the whole logits buffer
    // stays on device; only the two scalars come back.
    loss::cross_entropy_loss_device(
        logits_->data(), d_targets_->data(), d_loss_->data(),
        m, hidden_dim_, ce_cfg_, nullptr);
    float loss = 0.0f;
    d_loss_->copy_to(&loss, 1);

    if (accuracy != nullptr) {
        loss::accuracy_device(
            logits_->data(), d_targets_->data(), d_loss_->data(),
            m, hidden_dim_, nullptr);
        float correct = 0.0f;
        d_loss_->copy_to(&correct, 1);
        *accuracy = correct / static_cast<float>(m);
    }
    return loss;
}

void MicroTrainer::copy_weights(float* gate, float* up, float* down) const {
    mlp_->copy_weights(gate, up, down);
}

void MicroTrainer::save_state(std::ostream& out) const {
    // The format captures what copy_weights() exposes (the full weights at
    // tp==1, rank shards otherwise). Rank-shard restore needs shard setters on
    // the MLP pieces — the v2.32 follow-up — so refuse to emit a checkpoint
    // that cannot roundtrip rather than write one that silently can't load.
    if (tp_degree() > 1) {
        throw std::runtime_error(
            "MicroTrainer::save_state: tp > 1 rank-shard checkpoints are the "
            "v2.32 follow-up (TASK); verified on the tp == 1 path");
    }
    namespace cp = checkpoint;
    cp::Writer w(out);
    w.u32(cp::kMagic);
    w.u32(cp::kVersion);
    w.u32(cp::kKindMicroTrainer);
    w.u32(static_cast<uint32_t>(hidden_dim_));
    w.u32(static_cast<uint32_t>(intermediate_size_));

    const size_t h = static_cast<size_t>(hidden_dim_);
    const size_t inter = static_cast<size_t>(intermediate_size_);
    const size_t ng = h * inter, nu = h * inter, nd = inter * h;
    std::vector<float> g(ng), u(nu), d(nd);
    copy_weights(g.data(), u.data(), d.data());
    w.tensor(g.data(), ng);
    w.tensor(u.data(), nu);
    w.tensor(d.data(), nd);

    cp::write_adamw_moments(w, *opt_gate_);
    cp::write_adamw_moments(w, *opt_up_);
    cp::write_adamw_moments(w, *opt_down_);
}

void MicroTrainer::load_state(std::istream& in) {
    namespace cp = checkpoint;
    cp::Reader r(in);
    if (r.u32() != cp::kMagic) {
        throw std::runtime_error("Nova checkpoint: bad magic (not a Nova checkpoint)");
    }
    if (r.u32() != cp::kVersion) {
        throw std::runtime_error("Nova checkpoint: unsupported checkpoint version");
    }
    if (r.u32() != cp::kKindMicroTrainer) {
        throw std::runtime_error(
            "Nova checkpoint: kind mismatch (not a MicroTrainer checkpoint)");
    }
    const uint32_t fh = r.u32(), fi = r.u32();
    if (fh != static_cast<uint32_t>(hidden_dim_) ||
        fi != static_cast<uint32_t>(intermediate_size_)) {
        throw std::runtime_error(
            "Nova checkpoint: dimension mismatch (checkpoint " +
            std::to_string(fh) + "x" + std::to_string(fi) + " vs trainer " +
            std::to_string(hidden_dim_) + "x" +
            std::to_string(intermediate_size_) + ")");
    }
    // Symmetric with save_state: a tp > 1 trainer's moment buffers are the
    // rank shard sizes, which don't match a full-weight checkpoint's records.
    if (tp_degree() > 1) {
        throw std::runtime_error(
            "MicroTrainer::load_state: tp > 1 rank-shard checkpoints are the "
            "v2.32 follow-up (TASK); verified on the tp == 1 path");
    }
    // Two-phase load (cpp-reviewer MEDIUM, RIL EV-040): read and validate
    // every tensor and moment record into host memory FIRST, then apply — a
    // corrupt/truncated stream throws with this trainer untouched (no partial
    // restore). Moment records are count-tagged; read_adamw_moments requires
    // count == 0 (fresh) or exactly the companion tensor's size.
    const size_t h = static_cast<size_t>(hidden_dim_);
    const size_t inter = static_cast<size_t>(intermediate_size_);
    const size_t ng = h * inter, nu = h * inter, nd = inter * h;
    std::vector<float> g(ng), u(nu), d(nd);
    r.tensor(g.data(), ng, "gate");
    r.tensor(u.data(), nu, "up");
    r.tensor(d.data(), nd, "down");
    const cp::AdamWMomentRecord mg = cp::read_adamw_moments(r, ng, "gate");
    const cp::AdamWMomentRecord mu = cp::read_adamw_moments(r, nu, "up");
    const cp::AdamWMomentRecord md = cp::read_adamw_moments(r, nd, "down");

    set_weight(g.data(), u.data(), d.data());
    cp::apply_adamw_moments(*opt_gate_, mg);
    cp::apply_adamw_moments(*opt_up_, mu);
    cp::apply_adamw_moments(*opt_down_, md);
}

int MicroTrainer::hidden_dim() const { return hidden_dim_; }

int MicroTrainer::intermediate_size() const { return intermediate_size_; }

int MicroTrainer::tp_degree() const { return mlp_->tp_degree(); }

}  // namespace cuda::neural::training
