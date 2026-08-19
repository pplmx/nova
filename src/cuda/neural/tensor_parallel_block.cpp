#include "cuda/neural/tensor_parallel_block.h"

#include "cuda/neural/activations.h"

#include <stdexcept>

namespace cuda::neural {

// ============================================================================
// v2.28 pre-LN residual transformer block (TASK-041 / DEC-014). Composition of
// the verified pieces — LayerNorm (v2.28), TensorParallelMultiHeadAttention
// (v2.26) and TensorParallelMLP (v2.21) — with residual adds (elementwise_add):
//
//     forward: h1 = LN1(x); a = MHA(h1); h2 = x + a; h3 = LN2(h2);
//              out = h2 + MLP(h3)
//     backward: out = h2 + m_out -> dH2_res = dout; MLP gives dL/dLN2-out;
//               dL/dh2 = dout + LN2-path; attention consumes dL/da = dL/dh2;
//               dL/dx = dL/dh2 (residual) + LN1-path.
//
// All activations are replicated [m x hidden]; the only communicator in the
// block is the MHA output projection's AllReduce (v2.26). One AdamW per weight
// tensor (11 per block), stepped by the caller.
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

void TransformerBlock::ensure_scratch(int m) {
    if (scratch_m_ >= m) return;
    const size_t n = static_cast<size_t>(m) * hidden_;
    h1_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    a_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    h2_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    h3_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    m_out_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    d_mlp_gi_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    d_r1ln_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    d_h2_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    d_attn_gi_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    d_xln_ = std::make_unique<cuda::memory::Buffer<float>>(n);
    scratch_m_ = m;
}

void TransformerBlock::set_weight(const float* ln1_gamma, const float* ln1_beta,
                                  const float* wq, const float* wk,
                                  const float* wv, const float* wo,
                                  const float* ln2_gamma,
                                  const float* ln2_beta, const float* wg,
                                  const float* wu, const float* wd) {
    ln1_->set_weight(ln1_gamma, ln1_beta);
    attn_->set_weight(wq, wk, wv, wo);
    ln2_->set_weight(ln2_gamma, ln2_beta);
    mlp_->set_weight(wg, wu, wd);
}

void TransformerBlock::forward(const float* input, float* output, int batch,
                               int seq) {
    ensure_scratch(batch);
    const int total = batch * hidden_;
    ln1_->forward(input, h1_->data(), batch);
    attn_->forward(h1_->data(), a_->data(), batch, seq);
    cuda::neural::elementwise_add(input, a_->data(), h2_->data(), total);
    ln2_->forward(h2_->data(), h3_->data(), batch);
    mlp_->forward(h3_->data(), m_out_->data(), batch, seq);
    cuda::neural::elementwise_add(h2_->data(), m_out_->data(), output, total);
}

void TransformerBlock::backward(
    const float* input, const float* grad_output, float* grad_input,
    float* dln1_gamma, float* dln1_beta, float* dwq, float* dwk, float* dwv,
    float* dwo, float* dln2_gamma, float* dln2_beta, float* dwg, float* dwu,
    float* dwd, int batch, int seq) {
    ensure_scratch(batch);
    const int total = batch * hidden_;
    // out = h2 + m_out with m_out = MLP(h3), h3 = LN2(h2):
    //   MLP backward consumes dout (dL/dm_out), giving dL/dh3 (MLP grad-input)
    //   and the MLP weight grads.
    mlp_->backward(h3_->data(), grad_output, d_mlp_gi_->data(), dwg, dwu, dwd,
                   batch, seq);
    // LN2 backward: input h2, grad dL/dh3 -> dL/dh2 via LN2 (d_r1ln_) + LN2
    // affine grads.
    ln2_->backward(h2_->data(), d_mlp_gi_->data(), d_r1ln_->data(),
                   dln2_gamma, dln2_beta, batch);
    // dL/dh2 = dout (residual path) + d_r1ln_.
    cuda::neural::elementwise_add(grad_output, d_r1ln_->data(), d_h2_->data(),
                                  total);
    // h2 = x + a with a = MHA(h1), h1 = LN1(x):
    //   attention backward consumes dL/da = d_h2_, giving dL/dh1 (attn
    //   grad-input) + the QKV/o weight grads.
    attn_->backward(h1_->data(), d_h2_->data(), d_attn_gi_->data(), dwq, dwk,
                    dwv, dwo, batch, seq);
    // LN1 backward: input x, grad dL/dh1 -> dL/dx via LN1 (d_xln_) + LN1 affine
    // grads.
    ln1_->backward(input, d_attn_gi_->data(), d_xln_->data(), dln1_gamma,
                   dln1_beta, batch);
    // dL/dx = dL/dh2 (residual direct to x) + d_xln_.
    cuda::neural::elementwise_add(d_h2_->data(), d_xln_->data(), grad_input,
                                  total);
}

void TransformerBlock::step(
    optimizers::AdamWOptimizer& o_l1g, optimizers::AdamWOptimizer& o_l1b,
    optimizers::AdamWOptimizer& o_q, optimizers::AdamWOptimizer& o_k,
    optimizers::AdamWOptimizer& o_v, optimizers::AdamWOptimizer& o_o,
    optimizers::AdamWOptimizer& o_l2g, optimizers::AdamWOptimizer& o_l2b,
    optimizers::AdamWOptimizer& o_g, optimizers::AdamWOptimizer& o_u,
    optimizers::AdamWOptimizer& o_d,
    const float* dln1_gamma, const float* dln1_beta, const float* dwq,
    const float* dwk, const float* dwv, const float* dwo,
    const float* dln2_gamma, const float* dln2_beta, const float* dwg,
    const float* dwu, const float* dwd, int step_no, cudaStream_t stream) {
    ln1_->step(o_l1g, o_l1b, dln1_gamma, dln1_beta, step_no, stream);
    attn_->step(o_q, o_k, o_v, o_o, dwq, dwk, dwv, dwo, step_no, stream);
    ln2_->step(o_l2g, o_l2b, dln2_gamma, dln2_beta, step_no, stream);
    mlp_->step(o_g, o_u, o_d, dwg, dwu, dwd, step_no, stream);
}

void TransformerBlock::copy_weights(
    float* ln1_gamma, float* ln1_beta, float* wq, float* wk, float* wv,
    float* wo, float* ln2_gamma, float* ln2_beta, float* wg, float* wu,
    float* wd) const {
    ln1_->copy_weights(ln1_gamma, ln1_beta);
    attn_->copy_weights(wq, wk, wv, wo);
    ln2_->copy_weights(ln2_gamma, ln2_beta);
    mlp_->copy_weights(wg, wu, wd);
}

int TransformerBlock::hidden() const { return hidden_; }
int TransformerBlock::num_heads() const { return heads_; }
int TransformerBlock::intermediate_size() const { return intermediate_; }
int TransformerBlock::tp_degree() const { return attn_->tp_degree(); }

}  // namespace cuda::neural
