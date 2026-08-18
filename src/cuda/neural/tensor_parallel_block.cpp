#include "cuda/neural/tensor_parallel_block.h"

#include <stdexcept>

namespace cuda::neural {

// ============================================================================
// v2.28 pre-LN residual transformer block. P1-RED: provisional throwing stubs
// — the parity/convergence tests pin the forward/backward/step contracts and
// go RED here; the real composition lands in P2 (TASK-041).
// ============================================================================

TransformerBlock::TransformerBlock(
    ::cuda::nccl::NcclContext& ctx,
    int hidden,
    int heads,
    int head_dim,
    int intermediate_size,
    float ln_eps)
    : hidden_(hidden),
      heads_(heads),
      head_dim_(head_dim),
      intermediate_(intermediate_size) {
    if (hidden_ <= 0 || heads_ <= 0 || head_dim_ <= 0 || intermediate_ <= 0) {
        throw std::invalid_argument("TransformerBlock: dims must be positive");
    }
    ln1_ = std::make_unique<LayerNorm>(hidden_, ln_eps);
    attn_ = std::make_unique<TensorParallelMultiHeadAttention>(
        ctx, heads_, head_dim_, hidden_);
    ln2_ = std::make_unique<LayerNorm>(hidden_, ln_eps);
    mlp_ = std::make_unique<TensorParallelMLP>(ctx, hidden_, intermediate_);
}

TransformerBlock::~TransformerBlock() = default;

void TransformerBlock::set_weight(const float*, const float*, const float*,
                                  const float*, const float*, const float*,
                                  const float*, const float*, const float*,
                                  const float*, const float*) {
    throw std::runtime_error("TransformerBlock::set_weight: not implemented (P1 RED stub)");
}

void TransformerBlock::forward(const float*, float*, int, int) {
    throw std::runtime_error("TransformerBlock::forward: not implemented (P1 RED stub)");
}

void TransformerBlock::backward(
    const float*, const float*, float*, float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*, float*, int, int) {
    throw std::runtime_error("TransformerBlock::backward: not implemented (P1 RED stub)");
}

void TransformerBlock::step(
    optimizers::AdamWOptimizer&, optimizers::AdamWOptimizer&,
    optimizers::AdamWOptimizer&, optimizers::AdamWOptimizer&,
    optimizers::AdamWOptimizer&, optimizers::AdamWOptimizer&,
    optimizers::AdamWOptimizer&, optimizers::AdamWOptimizer&,
    optimizers::AdamWOptimizer&, optimizers::AdamWOptimizer&,
    optimizers::AdamWOptimizer&,
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*,
    const float*, int, cudaStream_t) {
    throw std::runtime_error("TransformerBlock::step: not implemented (P1 RED stub)");
}

void TransformerBlock::copy_weights(
    float*, float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*) const {
    throw std::runtime_error("TransformerBlock::copy_weights: not implemented (P1 RED stub)");
}

int TransformerBlock::hidden() const { return hidden_; }
int TransformerBlock::num_heads() const { return heads_; }
int TransformerBlock::intermediate_size() const { return intermediate_; }

}  // namespace cuda::neural
