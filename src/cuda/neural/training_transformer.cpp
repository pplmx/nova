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

#include "checkpoint_io.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural::training {

namespace {

// A TransformerBlock's weight-tensor layout: LN1 gamma/beta, Wq/Wk/Wv/Wo,
// LN2 gamma/beta, MLP gate/up/down — the fixed order set_block_weight /
// copy_block_weights / backward use. Shared by the 11-tensor grad sizes, the
// checkpoint tensor sizes, and the 11-entry per-block arrays (one source of
// truth for the layout count).
constexpr int kBlockTensorCount = 11;

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

// A block's kBlockTensorCount weight-tensor element counts, in the canonical
// order of set_block_weight / copy_block_weights (matches block_grad_size's
// layout except that the checkpoint records are RANK-SHARD sizes at tp > 1:
// the 7 matmul tensors divide by tp, the 4 replicated LayerNorm gamma/beta
// stay full hidden — v2.33).
void block_tensor_sizes(int hidden, int qkv, int inter, int tp,
                        size_t sizes[]) {
    const size_t h = static_cast<size_t>(hidden);
    const size_t q = static_cast<size_t>(qkv) / tp;
    const size_t m = static_cast<size_t>(inter) / tp;
    sizes[0] = h;             // ln1_gamma (replicated)
    sizes[1] = h;             // ln1_beta
    sizes[2] = h * q;         // wq
    sizes[3] = h * q;         // wk
    sizes[4] = h * q;         // wv
    sizes[5] = q * h;         // wo
    sizes[6] = h;             // ln2_gamma (replicated)
    sizes[7] = h;             // ln2_beta
    sizes[8] = h * m;         // wg
    sizes[9] = h * m;         // wu
    sizes[10] = m * h;        // wd
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

void TransformerTrainer::save_state(std::ostream& out) const {
    namespace cp = checkpoint;
    cp::Writer w(out);
    w.u32(cp::kMagic);
    w.u32(cp::kVersion);
    w.u32(cp::kKindTransformerTrainer);
    w.u32(static_cast<uint32_t>(tp_degree()));
    w.u32(static_cast<uint32_t>(num_blocks_));
    w.u32(static_cast<uint32_t>(hidden_));
    w.u32(static_cast<uint32_t>(heads_));
    w.u32(static_cast<uint32_t>(head_dim_));
    w.u32(static_cast<uint32_t>(intermediate_));

    // Records are this rank's shard sizes (full at tp == 1); copy_block_weights
    // returns exactly these, and each tensor's moment capacity matches.
    const int qkv = heads_ * head_dim_;
    size_t sz[kBlockTensorCount];
    block_tensor_sizes(hidden_, qkv, intermediate_, tp_degree(), sz);
    for (int b = 0; b < num_blocks_; ++b) {
        std::array<std::vector<float>, kBlockTensorCount> tens;
        for (int i = 0; i < kBlockTensorCount; ++i) {
            tens[static_cast<size_t>(i)].resize(sz[i]);
        }
        copy_block_weights(b, tens[0].data(), tens[1].data(), tens[2].data(),
                           tens[3].data(), tens[4].data(), tens[5].data(),
                           tens[6].data(), tens[7].data(), tens[8].data(),
                           tens[9].data(), tens[10].data());
        for (int i = 0; i < kBlockTensorCount; ++i) {
            w.tensor(tens[static_cast<size_t>(i)].data(), sz[i]);
        }
        for (int i = 0; i < kBlockTensorCount; ++i) {
            cp::write_adamw_moments(w, *block_opts_[static_cast<size_t>(b)]
                                             [static_cast<size_t>(i)]);
        }
    }
    std::vector<float> fg(static_cast<size_t>(hidden_));
    std::vector<float> fb(static_cast<size_t>(hidden_));
    copy_final_ln(fg.data(), fb.data());
    w.tensor(fg.data(), fg.size());
    w.tensor(fb.data(), fb.size());
    cp::write_adamw_moments(w, *final_opts_[0]);
    cp::write_adamw_moments(w, *final_opts_[1]);
}

void TransformerTrainer::load_state(std::istream& in) {
    namespace cp = checkpoint;
    cp::Reader r(in);
    if (r.u32() != cp::kMagic) {
        throw std::runtime_error("Nova checkpoint: bad magic (not a Nova checkpoint)");
    }
    if (r.u32() != cp::kVersion) {
        throw std::runtime_error("Nova checkpoint: unsupported checkpoint version");
    }
    if (r.u32() != cp::kKindTransformerTrainer) {
        throw std::runtime_error(
            "Nova checkpoint: kind mismatch (not a TransformerTrainer checkpoint)");
    }
    const uint32_t ftp = r.u32();
    if (ftp != static_cast<uint32_t>(tp_degree())) {
        throw std::runtime_error(
            "Nova checkpoint: TP-degree mismatch (checkpoint " +
            std::to_string(ftp) + " vs trainer " +
            std::to_string(tp_degree()) + ")");
    }
    const uint32_t fnb = r.u32(), fh = r.u32(), fheads = r.u32(),
                   fhd = r.u32(), fi = r.u32();
    if (fnb != static_cast<uint32_t>(num_blocks_) ||
        fh != static_cast<uint32_t>(hidden_) ||
        fheads != static_cast<uint32_t>(heads_) ||
        fhd != static_cast<uint32_t>(head_dim_) ||
        fi != static_cast<uint32_t>(intermediate_)) {
        throw std::runtime_error(
            "Nova checkpoint: dimension mismatch (checkpoint " +
            std::to_string(fnb) + "x" + std::to_string(fh) + "/" +
            std::to_string(fheads) + "/" + std::to_string(fhd) + "/" +
            std::to_string(fi) + " vs trainer " + std::to_string(num_blocks_) +
            "x" + std::to_string(hidden_) + "/" + std::to_string(heads_) + "/" +
            std::to_string(head_dim_) + "/" + std::to_string(intermediate_) + ")");
    }
    // Records validate against THIS trainer's shard sizes (matmul /tp, LN
    // replicated full — v2.33), so only the same topology roundtrips; applied
    // via the block's shard setters (no-slice) rather than the full-weight
    // set_block_weight. Two-phase load (cpp-reviewer MEDIUM, RIL EV-040):
    // read + validate everything into host memory FIRST, then apply — a
    // corrupt/truncated stream throws with this trainer untouched.
    const int qkv = heads_ * head_dim_;
    size_t sz[kBlockTensorCount];
    block_tensor_sizes(hidden_, qkv, intermediate_, tp_degree(), sz);
    std::vector<std::array<std::vector<float>, kBlockTensorCount>> tens(
        static_cast<size_t>(num_blocks_));
    std::vector<std::array<cp::AdamWMomentRecord, kBlockTensorCount>> mots(
        static_cast<size_t>(num_blocks_));
    for (int b = 0; b < num_blocks_; ++b) {
        std::array<std::vector<float>, kBlockTensorCount>& t =
            tens[static_cast<size_t>(b)];
        for (int i = 0; i < kBlockTensorCount; ++i) {
            t[static_cast<size_t>(i)].resize(sz[i]);
        }
        for (int i = 0; i < kBlockTensorCount; ++i) {
            r.tensor(t[static_cast<size_t>(i)].data(), sz[i],
                     (std::string("block ") + std::to_string(b) + " tensor " +
                      std::to_string(i)).c_str());
        }
        for (int i = 0; i < kBlockTensorCount; ++i) {
            mots[static_cast<size_t>(b)][static_cast<size_t>(i)] =
                cp::read_adamw_moments(r, sz[i], "block moment");
        }
    }
    std::vector<float> fg(static_cast<size_t>(hidden_));
    std::vector<float> fb(static_cast<size_t>(hidden_));
    r.tensor(fg.data(), fg.size(), "final gamma");
    r.tensor(fb.data(), fb.size(), "final beta");
    const cp::AdamWMomentRecord mg =
        cp::read_adamw_moments(r, fg.size(), "final gamma");
    const cp::AdamWMomentRecord mb =
        cp::read_adamw_moments(r, fb.size(), "final beta");

    for (int b = 0; b < num_blocks_; ++b) {
        std::array<std::vector<float>, kBlockTensorCount>& t =
            tens[static_cast<size_t>(b)];
        blocks_[static_cast<size_t>(b)]->set_weight_shards(
            t[0].data(), t[1].data(), t[2].data(), t[3].data(), t[4].data(),
            t[5].data(), t[6].data(), t[7].data(), t[8].data(), t[9].data(),
            t[10].data());
        for (int i = 0; i < kBlockTensorCount; ++i) {
            cp::apply_adamw_moments(
                *block_opts_[static_cast<size_t>(b)][static_cast<size_t>(i)],
                mots[static_cast<size_t>(b)][static_cast<size_t>(i)]);
        }
    }
    set_final_ln_weight(fg.data(), fb.data());
    cp::apply_adamw_moments(*final_opts_[0], mg);
    cp::apply_adamw_moments(*final_opts_[1], mb);
}

int TransformerTrainer::num_blocks() const { return num_blocks_; }
int TransformerTrainer::hidden() const { return hidden_; }
int TransformerTrainer::tp_degree() const {
    return blocks_.empty() ? 1 : blocks_[0]->tp_degree();
}

}  // namespace cuda::neural::training
