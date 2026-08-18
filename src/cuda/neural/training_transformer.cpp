#include "cuda/neural/training_transformer.h"

#include <stdexcept>

namespace cuda::neural::training {

// ============================================================================
// v2.28 deep transformer training driver. P1-RED: the constructor establishes
// the parameter ownership (blocks + per-tensor optimizers) and the training
// calls are provisional throw-stubs so the convergence/parity RED tests
// compile and fail here; the real loop lands in P2 (TASK-041).
// ============================================================================

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

void TransformerTrainer::set_block_weight(
    int, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*,
    const float*) {
    throw std::runtime_error("TransformerTrainer::set_block_weight: not implemented (P1 RED stub)");
}

void TransformerTrainer::set_final_ln_weight(const float*, const float*) {
    throw std::runtime_error("TransformerTrainer::set_final_ln_weight: not implemented (P1 RED stub)");
}

float TransformerTrainer::train_step(const float*, const int*, int, int, int) {
    throw std::runtime_error("TransformerTrainer::train_step: not implemented (P1 RED stub)");
}

float TransformerTrainer::evaluate(const float*, const int*, int, int,
                                   float*) {
    throw std::runtime_error("TransformerTrainer::evaluate: not implemented (P1 RED stub)");
}

void TransformerTrainer::copy_block_weights(
    int, float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*, float*) const {
    throw std::runtime_error("TransformerTrainer::copy_block_weights: not implemented (P1 RED stub)");
}

void TransformerTrainer::copy_final_ln(float*, float*) const {
    throw std::runtime_error("TransformerTrainer::copy_final_ln: not implemented (P1 RED stub)");
}

int TransformerTrainer::num_blocks() const { return num_blocks_; }
int TransformerTrainer::hidden() const { return hidden_; }
int TransformerTrainer::tp_degree() const { return 1; }

}  // namespace cuda::neural::training
