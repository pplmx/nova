/**
 * @file tensor_parallel_attention.cpp
 * @brief Tensor-parallel Multi-Head Attention block (milestone v2.26)
 *
 * Builds attention on the verified v2.21-25 layer conventions: Q/K/V as
 * ColumnParallelLayer (each rank owns contiguous head columns -> the rank's
 * heads are complete locally, so the per-head SDPA needs NO cross-rank
 * communication), output as RowParallelLayer (one block-wise AllReduce ->
 * replicated output). Backward follows the v2.23 pattern: row-layer backward
 * yields the local head-grads, SDPA backward gives dQ/dK/dV, and each QKV
 * column layer scatters its grad-weight slice and AllReduces the replicated-
 * input grad-input, summed to the full block grad-input.
 *
 * The multi-head SDPA itself (forward + analytic backward) is computed on the
 * host via D2H/H2D copies, mirroring the existing host-side AdamW idiom in
 * optimizers.cpp: correctness-first, verified reference-parity (a later
 * performance pass can move it to device kernels). There is deliberately no
 * dead device kernel here — the incomplete single-GPU MultiHeadAttention
 * (issue-v24-mha-incomplete) is the anti-pattern.
 */

#include "cuda/neural/tensor_parallel_attention.h"

#include "cuda/device/error.h"
#include "cuda/neural/activations.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural {

namespace {

[[noreturn]] void throw_no_weight(const char* what) {
    throw std::runtime_error(
        std::string(what) +
        " called before set_weight: the block owns no weight shard yet "
        "(upload one via set_weight).");
}

// Per-head softmax(Q K^T / sqrt(d)) V over m keys (self-attention, no mask).
// Host layout: x[(b*local_heads + h)*head_dim + d]; keys/values share the
// same (b,h) indexing; key row j is k[(j*local_heads + h)*head_dim .. ].
void sdpa_forward_cpu(const float* q, const float* k, const float* v,
                      float* out, int m, int local_heads, int head_dim) {
    const float inv = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t hd = static_cast<size_t>(head_dim);
    const size_t localStr = static_cast<size_t>(local_heads) * hd;
    for (int b = 0; b < m; ++b) {
        for (int h = 0; h < local_heads; ++h) {
            const float* q_row = q + (static_cast<size_t>(b) * local_heads + h) * hd;
            const float* k_base = k + static_cast<size_t>(h) * hd;
            const float* v_base = v + static_cast<size_t>(h) * hd;
            float* out_row = out + (static_cast<size_t>(b) * local_heads + h) * hd;

            std::vector<float> scores(static_cast<size_t>(m));
            float max_s = -1e30f;
            for (int j = 0; j < m; ++j) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    s += q_row[d] * k_base[static_cast<size_t>(j) * localStr + d];
                }
                scores[static_cast<size_t>(j)] = s * inv;
                max_s = std::max(max_s, scores[static_cast<size_t>(j)]);
            }
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                scores[static_cast<size_t>(j)] =
                    std::exp(scores[static_cast<size_t>(j)] - max_s);
                sum += scores[static_cast<size_t>(j)];
            }
            for (int j = 0; j < m; ++j) {
                scores[static_cast<size_t>(j)] /= sum;
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < m; ++j) {
                    acc += scores[static_cast<size_t>(j)] *
                           v_base[static_cast<size_t>(j) * localStr + d];
                }
                out_row[d] = acc;
            }
        }
    }
}

// Analytic backward of out = softmax(Q K^T / sqrt(d)) V.
// Per head: P[i,j] = softmax_j(q_i . k_j / sqrt(d)); dV[j,:] = sum_i P[i,j]
// dout[i,:]; dP[i,j] = dout[i,:] . v[j,:]; dscore[i,j] = P[i,j] * (dP[i,j] -
// sum_k P[i,k] dP[i,k]) (softmax Jacobian, then scaled by 1/sqrt(d)); dQ[i,:]
// = sum_j dscore[i,j] k[j,:]; dK[j,:] = sum_i dscore[i,j] q[i,:].
void sdpa_backward_cpu(const float* q, const float* k, const float* v,
                       const float* dout, float* dq, float* dk, float* dv,
                       int m, int local_heads, int head_dim) {
    const float inv = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t hd = static_cast<size_t>(head_dim);
    const size_t localStr = static_cast<size_t>(local_heads) * hd;
    for (int b = 0; b < m; ++b) {
        for (int h = 0; h < local_heads; ++h) {
            const size_t row_off = (static_cast<size_t>(b) * local_heads + h) * hd;
            const float* q_row = q + row_off;
            const float* k_base = k + static_cast<size_t>(h) * hd;
            const float* v_base = v + static_cast<size_t>(h) * hd;
            const float* dout_row = dout + row_off;
            float* dq_row = dq + row_off;

            // Forward probs P[j] for this query row.
            std::vector<float> scores(static_cast<size_t>(m));
            std::vector<float> probs(static_cast<size_t>(m));
            float max_s = -1e30f;
            for (int j = 0; j < m; ++j) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    s += q_row[d] * k_base[static_cast<size_t>(j) * localStr + d];
                }
                scores[static_cast<size_t>(j)] = s * inv;
                max_s = std::max(max_s, scores[static_cast<size_t>(j)]);
            }
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                probs[static_cast<size_t>(j)] =
                    std::exp(scores[static_cast<size_t>(j)] - max_s);
                sum += probs[static_cast<size_t>(j)];
            }
            for (int j = 0; j < m; ++j) {
                probs[static_cast<size_t>(j)] /= sum;
            }

            // dV[j,:] = P[j] * dout[:,:].
            for (int j = 0; j < m; ++j) {
                const float pj = probs[static_cast<size_t>(j)];
                float* dv_row =
                    dv + (static_cast<size_t>(j) * local_heads + h) * hd;
                for (int d = 0; d < head_dim; ++d) {
                    dv_row[d] += pj * dout_row[d];
                }
            }

            // dscore[j] = P[j] * (dP[j] - sum_k P[k] dP[k]), dP[j] =
            // dout[:,:] . v[j,:]; scale by inv (score = raw*inv).
            std::vector<float> dP(static_cast<size_t>(m));
            for (int j = 0; j < m; ++j) {
                float dp = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dp += dout_row[d] * v_base[static_cast<size_t>(j) * localStr + d];
                }
                dP[static_cast<size_t>(j)] = dp;
            }
            float weighted = 0.0f;
            for (int j = 0; j < m; ++j) {
                weighted += probs[static_cast<size_t>(j)] *
                            dP[static_cast<size_t>(j)];
            }
            std::vector<float> dscore(static_cast<size_t>(m));
            for (int j = 0; j < m; ++j) {
                dscore[static_cast<size_t>(j)] =
                    probs[static_cast<size_t>(j)] *
                    (dP[static_cast<size_t>(j)] - weighted) * inv;
            }

            // dQ[i,:] = sum_j dscore[j] k[j,:]; dK[j,:] = dscore[j] q[i,:].
            for (int d = 0; d < head_dim; ++d) {
                float acc_q = 0.0f;
                for (int j = 0; j < m; ++j) {
                    acc_q += dscore[static_cast<size_t>(j)] *
                             k_base[static_cast<size_t>(j) * localStr + d];
                }
                dq_row[d] += acc_q;
            }
            for (int j = 0; j < m; ++j) {
                const float ds = dscore[static_cast<size_t>(j)];
                float* dk_row =
                    dk + (static_cast<size_t>(j) * local_heads + h) * hd;
                for (int d = 0; d < head_dim; ++d) {
                    dk_row[d] += ds * q_row[d];
                }
            }
        }
    }
}

}  // namespace

void sdpa_forward(const float* q, const float* k, const float* v, float* out,
                  int m, int local_heads, int head_dim, cudaStream_t stream) {
    if (m <= 0 || local_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument(
            "sdpa_forward requires positive m, local_heads, head_dim");
    }
    const size_t n = static_cast<size_t>(m) * local_heads * head_dim;
    std::vector<float> h_q(n), h_k(n), h_v(n), h_out(n);
    CUDA_CHECK(cudaMemcpy(h_q.data(), q, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k.data(), k, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), v, n * sizeof(float), cudaMemcpyDeviceToHost));
    sdpa_forward_cpu(h_q.data(), h_k.data(), h_v.data(), h_out.data(), m,
                     local_heads, head_dim);
    CUDA_CHECK(cudaMemcpy(out, h_out.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    (void)stream;
}

void sdpa_backward(const float* q, const float* k, const float* v,
                   const float* out, const float* dout, float* dq, float* dk,
                   float* dv, int m, int local_heads, int head_dim,
                   cudaStream_t stream) {
    if (m <= 0 || local_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument(
            "sdpa_backward requires positive m, local_heads, head_dim");
    }
    const size_t n = static_cast<size_t>(m) * local_heads * head_dim;
    std::vector<float> h_q(n), h_k(n), h_v(n), h_dout(n);
    std::vector<float> h_dq(n, 0.0f), h_dk(n, 0.0f), h_dv(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_q.data(), q, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k.data(), k, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), v, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dout.data(), dout, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    sdpa_backward_cpu(h_q.data(), h_k.data(), h_v.data(), h_dout.data(),
                      h_dq.data(), h_dk.data(), h_dv.data(), m, local_heads,
                      head_dim);
    CUDA_CHECK(cudaMemcpy(dq, h_dq.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, h_dk.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, h_dv.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    (void)out;
    (void)stream;
}

// ============================================================================
// TensorParallelMultiHeadAttention
// ============================================================================

TensorParallelMultiHeadAttention::TensorParallelMultiHeadAttention(
    ::cuda::nccl::NcclContext& ctx, int num_heads, int head_dim, int hidden_dim)
    : num_heads_(num_heads),
      head_dim_(head_dim),
      hidden_dim_(hidden_dim),
      q_proj_(std::make_unique<ColumnParallelLayer>(ctx, hidden_dim,
                                                    num_heads * head_dim)),
      k_proj_(std::make_unique<ColumnParallelLayer>(ctx, hidden_dim,
                                                    num_heads * head_dim)),
      v_proj_(std::make_unique<ColumnParallelLayer>(ctx, hidden_dim,
                                                    num_heads * head_dim)),
      out_proj_(std::make_unique<RowParallelLayer>(ctx, num_heads * head_dim,
                                                   hidden_dim)) {
    if (num_heads_ <= 0 || head_dim_ <= 0 || hidden_dim_ <= 0) {
        throw std::invalid_argument(
            "TensorParallelMultiHeadAttention requires positive dims");
    }
}

TensorParallelMultiHeadAttention::~TensorParallelMultiHeadAttention() = default;

void TensorParallelMultiHeadAttention::ensure_scratch(int m) {
    const size_t need =
        static_cast<size_t>(m) *
        (static_cast<size_t>(num_heads_) / static_cast<size_t>(tp_degree())) *
        static_cast<size_t>(head_dim_);
    if (q_buf_ != nullptr &&
        static_cast<size_t>(scratch_m_) >= static_cast<size_t>(m)) {
        return;
    }
    q_buf_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    k_buf_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    v_buf_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    qkv_out_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    scratch_m_ = m;
}

void TensorParallelMultiHeadAttention::set_weight(
    const float* wq, const float* wk, const float* wv, const float* wo) {
    q_proj_->set_weight(wq);
    k_proj_->set_weight(wk);
    v_proj_->set_weight(wv);
    out_proj_->set_weight(wo);
}

void TensorParallelMultiHeadAttention::forward(
    const float* input, float* output, int batch, int seq) {
    const int m = batch * seq;
    if (m <= 0) {
        throw std::invalid_argument(
            "TensorParallelMultiHeadAttention::forward requires batch*seq > 0");
    }
    ensure_scratch(m);
    q_proj_->forward(input, q_buf_->data(), batch, seq);
    k_proj_->forward(input, k_buf_->data(), batch, seq);
    v_proj_->forward(input, v_buf_->data(), batch, seq);
    const int local_heads = num_heads_ / tp_degree();
    sdpa_forward(q_buf_->data(), k_buf_->data(), v_buf_->data(),
                 qkv_out_->data(), m, local_heads, head_dim_, nullptr);
    out_proj_->forward(qkv_out_->data(), output, batch, seq);
}

void TensorParallelMultiHeadAttention::backward(
    const float* input, const float* grad_output, float* grad_input,
    float* grad_wq, float* grad_wk, float* grad_wv, float* grad_wo,
    int batch, int seq) {
    const int m = batch * seq;
    if (m <= 0) {
        throw std::invalid_argument(
            "TensorParallelMultiHeadAttention::backward requires batch*seq > 0");
    }
    ensure_scratch(m);
    const int local_heads = num_heads_ / tp_degree();
    const size_t head_block =
        static_cast<size_t>(m) * local_heads * head_dim_;

    cuda::memory::Buffer<float> d_q(head_block), d_k(head_block),
        d_v(head_block), d_qkv(head_block);

    // Output row-layer backward: local head-grads (sharded), grad-wo slice.
    out_proj_->backward(qkv_out_->data(), grad_output, d_qkv.data(),
                        grad_wo, batch, seq);
    // SDPA backward -> dQ/dK/dV (analytic scaled-dot-product chain).
    sdpa_backward(q_buf_->data(), k_buf_->data(), v_buf_->data(),
                  qkv_out_->data(), d_qkv.data(), d_q.data(), d_k.data(),
                  d_v.data(), m, local_heads, head_dim_, nullptr);
    // QKV column-layer backward: each writes its rank column-slice grad-weight
    // and AllReduces the full replicated grad-input; a temporary full-width
    // buffer holds each layer's [m x hidden] result, then they are summed.
    const size_t gi_elems = static_cast<size_t>(m) * hidden_dim_;
    cuda::memory::Buffer<float> gi_q(gi_elems), gi_k(gi_elems),
        gi_v(gi_elems);
    q_proj_->backward(input, d_q.data(), gi_q.data(), grad_wq, batch, seq);
    k_proj_->backward(input, d_k.data(), gi_k.data(), grad_wk, batch, seq);
    v_proj_->backward(input, d_v.data(), gi_v.data(), grad_wv, batch, seq);
    cuda::neural::elementwise_add(gi_q.data(), gi_k.data(), grad_input,
                                  static_cast<int>(gi_elems));
    cuda::neural::elementwise_add(grad_input, gi_v.data(), grad_input,
                                  static_cast<int>(gi_elems));
}

void TensorParallelMultiHeadAttention::step(
    optimizers::AdamWOptimizer& opt_q, optimizers::AdamWOptimizer& opt_k,
    optimizers::AdamWOptimizer& opt_v, optimizers::AdamWOptimizer& opt_o,
    const float* grad_wq, const float* grad_wk, const float* grad_wv,
    const float* grad_wo, int step_no, cudaStream_t stream) {
    // One AdamW per weight tensor — the per-tensor-moment convention.
    q_proj_->step(opt_q, grad_wq, step_no, stream);
    k_proj_->step(opt_k, grad_wk, step_no, stream);
    v_proj_->step(opt_v, grad_wv, step_no, stream);
    out_proj_->step(opt_o, grad_wo, step_no, stream);
}

void TensorParallelMultiHeadAttention::copy_weights(
    float* wq, float* wk, float* wv, float* wo) const {
    q_proj_->copy_weight_shard(wq);
    k_proj_->copy_weight_shard(wk);
    v_proj_->copy_weight_shard(wv);
    out_proj_->copy_weight_shard(wo);
}

int TensorParallelMultiHeadAttention::num_heads() const { return num_heads_; }

int TensorParallelMultiHeadAttention::head_dim() const { return head_dim_; }

int TensorParallelMultiHeadAttention::hidden_dim() const { return hidden_dim_; }

int TensorParallelMultiHeadAttention::tp_degree() const {
    return std::max(1, q_proj_ ? q_proj_->tp_degree() : 1);
}

}  // namespace cuda::neural
