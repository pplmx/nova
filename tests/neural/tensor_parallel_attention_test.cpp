/**
 * @file tensor_parallel_attention_test.cpp
 * @brief Single-GPU (tp == 1) functional tests for TensorParallelMultiHeadAttention
 *
 * Milestone v2.26 (TASK-032 / DEC-012): no attention exists on the parallel
 * path — the single-GPU MultiHeadAttention is an incomplete shell
 * (issue-v24-mha-incomplete: it scales a buffer and returns). This block
 * builds attention on the verified weight-sharded layer conventions. These
 * tests pin the tp == 1 path (uninitialized NcclContext -> effective tp 1):
 * forward and backward must match a host fp64 multi-head self-attention
 * reference (QKV projections, per-head softmax(QK^T/sqrt(d))V, output
 * projection; analytic dV/dP/dscore/dQ/dK/dV gradients). The multi-GPU shard
 * parity lives in tensor_parallel_attention_multigpu_test.cpp.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/tensor_parallel_attention.h"
#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/neural/loss/loss_functions.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <utility>

using namespace cuda::neural;
using namespace cuda::neural::optimizers;
using namespace cuda::neural::loss;

namespace {

cuda::nccl::NcclContext& uninitialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

void fill_random(float* data, size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

void host_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) {
                acc += static_cast<double>(A[i * k + t]) *
                       static_cast<double>(B[t * n + j]);
            }
            C[i * n + j] = static_cast<float>(acc);
        }
    }
}

// Host fp64 multi-head self-attention forward over the FULL weight set:
// qkv_q = X Wq, per-head softmax(qkv_k^T/sqrt(d)) qkv_v, out = concat @ Wo.
void ref_attention_forward(const float* X, const float* Wq, const float* Wk,
                           const float* Wv, const float* Wo, float* out,
                           int m, int heads, int hd, int hidden) {
    const int qkv = heads * hd;
    const int stride = qkv;  // position-major stride for k/v
    std::vector<float> q(m * qkv), k(m * qkv), v(m * qkv), att(m * qkv);
    host_matmul(X, Wq, q.data(), m, qkv, hidden);
    host_matmul(X, Wk, k.data(), m, qkv, hidden);
    host_matmul(X, Wv, v.data(), m, qkv, hidden);
    const float inv = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int b = 0; b < m; ++b) {
        for (int h = 0; h < heads; ++h) {
            const float* q_row = q.data() + (b * heads + h) * hd;
            const float* k_base = k.data() + h * hd;
            const float* v_base = v.data() + h * hd;
            float* att_row = att.data() + (b * heads + h) * hd;
            std::vector<float> scores(m);
            float max_s = -1e30f;
            for (int j = 0; j < m; ++j) {
                double s = 0.0;
                for (int d = 0; d < hd; ++d) {
                    s += static_cast<double>(q_row[d]) * k_base[j * stride + d];
                }
                scores[j] = static_cast<float>(s) * inv;
                max_s = std::max(max_s, scores[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                scores[j] = std::exp(scores[j] - max_s);
                sum += scores[j];
            }
            for (int j = 0; j < m; ++j) scores[j] /= sum;
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < m; ++j) acc += scores[j] * v_base[j * stride + d];
                att_row[d] = static_cast<float>(acc);
            }
        }
    }
    host_matmul(att.data(), Wo, out, m, hidden, qkv);
}

// Host fp64 multi-head self-attention backward over the FULL weights:
// dqkv (dQKV) -> dWq = X^T dQ, dWk = X^T dK, dWv = X^T dV, dWo = att^T dout,
// dX = dQ Wq^T + dK Wk^T + dV Wv^T.
void ref_linear_backward(const float* X, const float* W, const float* dY,
                         float* dX, float* dW, int m, int k, int n) {
    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < n; ++b) {
            double acc = 0.0;
            for (int i = 0; i < m; ++i) {
                acc += static_cast<double>(X[i * k + a]) * dY[i * n + b];
            }
            dW[a * n + b] = static_cast<float>(acc);
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int a = 0; a < k; ++a) {
            double acc = 0.0;
            for (int b = 0; b < n; ++b) {
                acc += static_cast<double>(dY[i * n + b]) * W[a * n + b];
            }
            dX[i * k + a] = static_cast<float>(acc);
        }
    }
}

void ref_attention_backward(const float* X, const float* Wq, const float* Wk,
                            const float* Wv, const float* Wo, const float* dout,
                            float* dX, float* dWq, float* dWk, float* dWv,
                            float* dWo, int m, int heads, int hd, int hidden) {
    const int qkv = heads * hd;
    const int stride = qkv;  // position-major stride for k/v
    // Forward activations (needed for att and dP).
    std::vector<float> q(m * qkv), k(m * qkv), v(m * qkv), att(m * qkv);
    host_matmul(X, Wq, q.data(), m, qkv, hidden);
    host_matmul(X, Wk, k.data(), m, qkv, hidden);
    host_matmul(X, Wv, v.data(), m, qkv, hidden);
    const float inv = 1.0f / std::sqrt(static_cast<float>(hd));
    auto probs_for = [&](int b, int h) {
        std::vector<float> p(m);
        const float* q_row = q.data() + (b * heads + h) * hd;
        const float* k_base = k.data() + h * hd;
        float mx = -1e30f;
        for (int j = 0; j < m; ++j) {
            float s = 0.0f;
            for (int d = 0; d < hd; ++d) s += q_row[d] * k_base[j * stride + d];
            p[j] = s * inv;
            mx = std::max(mx, p[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < m; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
        for (int j = 0; j < m; ++j) p[j] /= sum;
        return p;
    };
    for (int b = 0; b < m; ++b) {
        for (int h = 0; h < heads; ++h) {
            const float* q_row = q.data() + (b * heads + h) * hd;
            const float* k_base = k.data() + h * hd;
            const float* v_base = v.data() + h * hd;
            float* att_row = att.data() + (b * heads + h) * hd;
            auto p = probs_for(b, h);
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < m; ++j) acc += p[j] * v_base[j * stride + d];
                att_row[d] = static_cast<float>(acc);
            }
        }
    }
    // dWqkv (dQ/dK/dV full) via the row-major layout used by the projections.
    std::vector<float> dQ(m * qkv, 0.0f), dK(m * qkv, 0.0f), dV(m * qkv, 0.0f);
    // First the output projection backward: datt = dout Wo^T, dWo = att^T dout.
    std::vector<float> datt(m * qkv);
    ref_linear_backward(att.data(), Wo, dout, datt.data(), dWo, m, qkv, hidden);
    // Then per-head SDPA backward.
    std::vector<float> dP(m), dscore(m);
    for (int b = 0; b < m; ++b) {
        for (int h = 0; h < heads; ++h) {
            const float* q_row = q.data() + (b * heads + h) * hd;
            const float* k_base = k.data() + h * hd;
            const float* v_base = v.data() + h * hd;
            const float* datt_row = datt.data() + (b * heads + h) * hd;
            float* dq_row = dQ.data() + (b * heads + h) * hd;
            auto p = probs_for(b, h);
            // dV[j,:] += P[j] * datt[b,:]
            for (int j = 0; j < m; ++j) {
                const float pj = p[j];
                float* dV_row = dV.data() + (j * heads + h) * hd;
                for (int d = 0; d < hd; ++d) dV_row[d] += pj * datt_row[d];
            }
            // dP[j] = datt . v[j,:]; dscore[j] = P[j]*(dP[j]-sum_k P[k]dP[k])
            for (int j = 0; j < m; ++j) {
                double dp = 0.0;
                for (int d = 0; d < hd; ++d) {
                    dp += static_cast<double>(datt_row[d]) * v_base[j * stride + d];
                }
                dP[j] = static_cast<float>(dp);
            }
            double weighted = 0.0;
            for (int j = 0; j < m; ++j) weighted += p[j] * dP[j];
            for (int j = 0; j < m; ++j) {
                dscore[j] = p[j] * (dP[j] - static_cast<float>(weighted)) * inv;
            }
            // dQ[b,:] += sum_j dscore[j] k[j,:]; dK[j,:] += dscore[j] q[b,:]
            for (int d = 0; d < hd; ++d) {
                double acc_q = 0.0;
                for (int j = 0; j < m; ++j)
                    acc_q += static_cast<double>(dscore[j]) * k_base[j * stride + d];
                dq_row[d] += static_cast<float>(acc_q);
            }
            for (int j = 0; j < m; ++j) {
                const float ds = dscore[j];
                float* dK_row = dK.data() + (j * heads + h) * hd;
                for (int d = 0; d < hd; ++d) dK_row[d] += ds * q_row[d];
            }
        }
    }
    // Weight grads: dWq = X^T dQ, dWk = X^T dK, dWv = X^T dV.
    const int qkv_sz = qkv * hidden;  // dW buffer is k*n
    std::vector<float> tmp(qkv_sz);
    ref_linear_backward(X, Wq, dQ.data(), tmp.data(), dWq, m, hidden, qkv);
    ref_linear_backward(X, Wk, dK.data(), tmp.data(), dWk, m, hidden, qkv);
    ref_linear_backward(X, Wv, dV.data(), tmp.data(), dWv, m, hidden, qkv);
    // Input grad: dX = dQ Wq^T + dK Wk^T + dV Wv^T.
    std::vector<float> dXq(m * hidden), dXk(m * hidden), dXv(m * hidden);
    ref_linear_backward(X, Wq, dQ.data(), dXq.data(), tmp.data(), m, hidden, qkv);
    ref_linear_backward(X, Wk, dK.data(), dXk.data(), tmp.data(), m, hidden, qkv);
    ref_linear_backward(X, Wv, dV.data(), dXv.data(), tmp.data(), m, hidden, qkv);
    for (int i = 0; i < m * hidden; ++i) {
        dX[i] = dXq[i] + dXk[i] + dXv[i];
    }
}

}  // namespace

class TensorParallelAttentionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { cudaFree(nullptr); }

    static testing::AssertionResult arrays_near(const float* expected,
                                                const float* actual,
                                                size_t count,
                                                float tolerance) {
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(expected[i] - actual[i]) > tolerance) {
                return testing::AssertionFailure()
                       << "mismatch at " << i << ": expected " << expected[i]
                       << " actual " << actual[i] << " (|diff| "
                       << std::abs(expected[i] - actual[i]) << " > " << tolerance
                       << ")";
            }
        }
        return testing::AssertionSuccess();
    }
};

// tp==1: the attention block must equal a host fp64 multi-head self-attention
// forward on the full weights.
TEST_F(TensorParallelAttentionTest, ForwardMatchesReference) {
    const int m = 4, heads = 2, hd = 3, hidden = 6;  // qkv = 6 = hidden
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<float> Wq(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> Wk(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> Wv(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> Wo(static_cast<size_t>(heads * hd) * hidden);
    fill_random(X.data(), X.size(), 1);
    fill_random(Wq.data(), Wq.size(), 2);
    fill_random(Wk.data(), Wk.size(), 3);
    fill_random(Wv.data(), Wv.size(), 4);
    fill_random(Wo.data(), Wo.size(), 5);

    std::vector<float> ref(static_cast<size_t>(m) * hidden);
    ref_attention_forward(X.data(), Wq.data(), Wk.data(), Wv.data(), Wo.data(),
                          ref.data(), m, heads, hd, hidden);

    TensorParallelMultiHeadAttention attn(uninitialized_ctx(), heads, hd, hidden);
    attn.set_weight(Wq.data(), Wk.data(), Wv.data(), Wo.data());
    cuda::memory::Buffer<float> d_X(m * hidden), d_out(m * hidden);
    d_X.copy_from(X.data(), X.size());
    attn.forward(d_X.data(), d_out.data(), m, 1);
    std::vector<float> out(static_cast<size_t>(m) * hidden);
    d_out.copy_to(out.data(), out.size());
    EXPECT_TRUE(arrays_near(ref.data(), out.data(), out.size(), 1e-3f));
    EXPECT_EQ(attn.num_heads(), heads);
    EXPECT_EQ(attn.head_dim(), hd);
    EXPECT_EQ(attn.hidden_dim(), hidden);
    EXPECT_EQ(attn.tp_degree(), 1);
}

// Direct SDPA grad check: sdpa_forward/backward must match a host reference.
TEST_F(TensorParallelAttentionTest, SDPAForwardBackwardMatchHost) {
    const int m = 3, local_heads = 2, hd = 3;
    const size_t n = (size_t)m * local_heads * hd;
    std::vector<float> q(n), k(n), v(n), dout(n);
    fill_random(q.data(), n, 51);
    fill_random(k.data(), n, 52);
    fill_random(v.data(), n, 53);
    fill_random(dout.data(), n, 54);
    cuda::memory::Buffer<float> dq_b(n), dk_b(n), dv_b(n), dout_b(n), out_b(n);
    cuda::memory::Buffer<float> dqo(n), dko(n), dvo(n);
    dq_b.copy_from(q.data(), n); dk_b.copy_from(k.data(), n);
    dv_b.copy_from(v.data(), n); dout_b.copy_from(dout.data(), n);
    sdpa_forward(dq_b.data(), dk_b.data(), dv_b.data(), out_b.data(), m, local_heads, hd);
    sdpa_backward(dq_b.data(), dk_b.data(), dv_b.data(), out_b.data(),
                  dout_b.data(), dqo.data(), dko.data(), dvo.data(), m, local_heads, hd);

    const float inv = 1.0f / std::sqrt((float)hd);
    std::vector<float> ref_out(n), ref_dq(n, 0), ref_dk(n, 0), ref_dv(n, 0);
    // Compute softmax probs fresh per (b,h) to avoid cross-head contamination.
    auto probs_for = [&](int b, int h) {
        std::vector<float> p(m);
        const float* qr = q.data() + (b*local_heads + h)*hd;
        float mx = -1e30f;
        for (int j = 0; j < m; ++j) {
            float s = 0;
            for (int d = 0; d < hd; ++d) s += qr[d] * k[(j*local_heads + h)*hd + d];
            p[j] = s * inv; mx = std::max(mx, p[j]);
        }
        float sum = 0;
        for (int j = 0; j < m; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
        for (int j = 0; j < m; ++j) p[j] /= sum;
        return p;
    };
    for (int b = 0; b < m; ++b) for (int h = 0; h < local_heads; ++h) {
        const float* qr = q.data() + (b*local_heads + h)*hd;
        const float* dor = dout.data() + (b*local_heads + h)*hd;
        auto p = probs_for(b, h);
        for (int d = 0; d < hd; ++d) {
            float a = 0;
            for (int j = 0; j < m; ++j) a += p[j] * v[(j*local_heads + h)*hd + d];
            ref_out[(b*local_heads + h)*hd + d] = a;
        }
        for (int j = 0; j < m; ++j) {
            float dp = 0;
            for (int d = 0; d < hd; ++d) dp += dor[d] * v[(j*local_heads + h)*hd + d];
            float w = 0;
            for (int jj = 0; jj < m; ++jj) {
                float dpp = 0;
                for (int d = 0; d < hd; ++d) dpp += dor[d] * v[(jj*local_heads + h)*hd + d];
                w += p[jj] * dpp;
            }
            float ds = p[j] * (dp - w) * inv;
            for (int d = 0; d < hd; ++d) {
                ref_dq[(b*local_heads + h)*hd + d] += ds * k[(j*local_heads + h)*hd + d];
                ref_dk[(j*local_heads + h)*hd + d] += ds * qr[d];
            }
            for (int d = 0; d < hd; ++d) ref_dv[(j*local_heads + h)*hd + d] += p[j] * dor[d];
        }
    }
    std::vector<float> o_out(n), o_dq(n), o_dk(n), o_dv(n);
    out_b.copy_to(o_out.data(), n); dqo.copy_to(o_dq.data(), n);
    dko.copy_to(o_dk.data(), n); dvo.copy_to(o_dv.data(), n);
    EXPECT_TRUE(arrays_near(ref_out.data(), o_out.data(), n, 1e-4f)) << "SDPA out";
    EXPECT_TRUE(arrays_near(ref_dq.data(), o_dq.data(), n, 1e-4f)) << "SDPA dq";
    EXPECT_TRUE(arrays_near(ref_dk.data(), o_dk.data(), n, 1e-4f)) << "SDPA dk";
    EXPECT_TRUE(arrays_near(ref_dv.data(), o_dv.data(), n, 1e-4f)) << "SDPA dv";
}

// tp==1 backward: grads (input + all four weights) must match the host fp64
// analytic reference.
TEST_F(TensorParallelAttentionTest, BackwardMatchesReference) {
    const int m = 4, heads = 2, hd = 3, hidden = 6;
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<float> Wq(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> Wk(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> Wv(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> Wo(static_cast<size_t>(heads * hd) * hidden);
    std::vector<float> dout(static_cast<size_t>(m) * hidden);
    fill_random(X.data(), X.size(), 11);
    fill_random(Wq.data(), Wq.size(), 12);
    fill_random(Wk.data(), Wk.size(), 13);
    fill_random(Wv.data(), Wv.size(), 14);
    fill_random(Wo.data(), Wo.size(), 15);
    fill_random(dout.data(), dout.size(), 16);

    std::vector<float> ref_dX(static_cast<size_t>(m) * hidden);
    std::vector<float> ref_dWq(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> ref_dWk(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> ref_dWv(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> ref_dWo(static_cast<size_t>(heads * hd) * hidden);
    ref_attention_backward(X.data(), Wq.data(), Wk.data(), Wv.data(), Wo.data(),
                           dout.data(), ref_dX.data(), ref_dWq.data(),
                           ref_dWk.data(), ref_dWv.data(), ref_dWo.data(),
                           m, heads, hd, hidden);

    TensorParallelMultiHeadAttention attn(uninitialized_ctx(), heads, hd, hidden);
    attn.set_weight(Wq.data(), Wk.data(), Wv.data(), Wo.data());
    cuda::memory::Buffer<float> d_X(m * hidden), d_dout(m * hidden),
        d_dX(m * hidden);
    cuda::memory::Buffer<float> d_dWq(static_cast<size_t>(hidden) * heads * hd);
    cuda::memory::Buffer<float> d_dWk(static_cast<size_t>(hidden) * heads * hd);
    cuda::memory::Buffer<float> d_dWv(static_cast<size_t>(hidden) * heads * hd);
    cuda::memory::Buffer<float> d_dWo(static_cast<size_t>(heads * hd) * hidden);
    d_X.copy_from(X.data(), X.size());
    d_dout.copy_from(dout.data(), dout.size());
    cuda::memory::Buffer<float> d_out(m * hidden);
    attn.forward(d_X.data(), d_out.data(), m, 1);  // scratch qkv (separate out)
    attn.backward(d_X.data(), d_dout.data(), d_dX.data(), d_dWq.data(),
                  d_dWk.data(), d_dWv.data(), d_dWo.data(), m, 1);

    std::vector<float> dX(static_cast<size_t>(m) * hidden);
    std::vector<float> dWq(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> dWk(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> dWv(static_cast<size_t>(hidden) * heads * hd);
    std::vector<float> dWo(static_cast<size_t>(heads * hd) * hidden);
    d_dX.copy_to(dX.data(), dX.size());
    d_dWq.copy_to(dWq.data(), dWq.size());
    d_dWk.copy_to(dWk.data(), dWk.size());
    d_dWv.copy_to(dWv.data(), dWv.size());
    d_dWo.copy_to(dWo.data(), dWo.size());

    EXPECT_TRUE(arrays_near(ref_dWq.data(), dWq.data(), dWq.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWk.data(), dWk.data(), dWk.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWv.data(), dWv.data(), dWv.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWo.data(), dWo.data(), dWo.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dX.data(), dX.data(), dX.size(), 2e-3f))
        << "attention grad-input differs from the analytic reference";
}

// ============================================================================
// v2.26 capstone (TASK-034): a mini-transformer — one attention block feeding
// a TensorParallelMLP, logits = MLP(attn(X)) — must train on the tp<=1 path:
// loss descends over K AdamW steps and top-1 accuracy rises above chance.
// The block/MLP forward+backward are separately host-validated, so the NEW
// claim this pins is "the composed transformer trains on the parallel stack".
// ============================================================================
TEST_F(TensorParallelAttentionTest, MiniTransformerTrainingConverges) {
    const int m = 16, hidden = 8, heads = 4, hd = 2, inter = 16, K = 40;
    const int qkv = heads * hd;  // == hidden

    // Learnable task: gaussian inputs, fixed random "teacher" labels.
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<int> targets(m);
    {
        std::mt19937 rng(20260820);
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::vector<float> T(static_cast<size_t>(hidden) * hidden);
        for (auto& t : T) t = norm(rng);
        for (int i = 0; i < m; ++i) {
            for (int d = 0; d < hidden; ++d) X[i * hidden + d] = norm(rng);
            double best = -1e30;
            int argmax = 0;
            for (int c = 0; c < hidden; ++c) {
                double acc = 0.0;
                for (int d = 0; d < hidden; ++d)
                    acc += static_cast<double>(X[i * hidden + d]) * T[c * hidden + d];
                if (acc > best) { best = acc; argmax = c; }
            }
            targets[i] = argmax;
        }
    }
    std::vector<float> Wq(static_cast<size_t>(hidden) * qkv),
        Wk(static_cast<size_t>(hidden) * qkv), Wv(static_cast<size_t>(hidden) * qkv),
        Wo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> Wg(static_cast<size_t>(hidden) * inter),
        Wu(static_cast<size_t>(hidden) * inter), Wd(static_cast<size_t>(inter) * hidden);
    fill_random(Wq.data(), Wq.size(), 61);
    fill_random(Wk.data(), Wk.size(), 62);
    fill_random(Wv.data(), Wv.size(), 63);
    fill_random(Wo.data(), Wo.size(), 64);
    fill_random(Wg.data(), Wg.size(), 65);
    fill_random(Wu.data(), Wu.size(), 66);
    fill_random(Wd.data(), Wd.size(), 67);

    TensorParallelMultiHeadAttention attn(uninitialized_ctx(), heads, hd, hidden);
    attn.set_weight(Wq.data(), Wk.data(), Wv.data(), Wo.data());
    TensorParallelMLP mlp(uninitialized_ctx(), hidden, inter);
    mlp.set_weight(Wg.data(), Wu.data(), Wd.data());

    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
    AdamWOptimizer oq(cfg), ok(cfg), ov(cfg), oo(cfg), og(cfg), ou(cfg), od(cfg);
    CrossEntropyConfig ce;
    ce.num_classes = hidden;
    ce.reduction_mean = true;

    cuda::memory::Buffer<float> d_X(m * hidden), d_att(m * hidden),
        d_logits(m * hidden), d_dlogits(m * hidden);
    cuda::memory::Buffer<float> d_dA(m * hidden);          // attn grad-input
    cuda::memory::Buffer<float> d_dWq((size_t)hidden * qkv), d_dWk((size_t)hidden * qkv),
        d_dWv((size_t)hidden * qkv), d_dWo((size_t)qkv * hidden);
    cuda::memory::Buffer<float> d_dWg((size_t)hidden * inter),
        d_dWu((size_t)hidden * inter), d_dWd((size_t)inter * hidden);
    cuda::memory::Buffer<int> d_t(m);
    d_X.copy_from(X.data(), X.size());
    d_t.copy_from(targets.data(), m);

    auto train_step = [&](int s) {
        // forward: att -> mlp -> logits
        attn.forward(d_X.data(), d_att.data(), m, 1);
        mlp.forward(d_att.data(), d_logits.data(), m, 1);
        // seed
        cross_entropy_logits_backward(d_logits.data(), d_t.data(),
                                            d_dlogits.data(), m, hidden, ce);
        // MLP backward (input = attn output)
        mlp.backward(d_att.data(), d_dlogits.data(), d_dA.data(),
                     d_dWg.data(), d_dWu.data(), d_dWd.data(), m, 1);
        // attention backward (input = X)
        attn.backward(d_X.data(), d_dA.data(), d_att.data(),
                      d_dWq.data(), d_dWk.data(), d_dWv.data(), d_dWo.data(), m, 1);
        // step all seven weight tensors
        mlp.step(og, ou, od, d_dWg.data(), d_dWu.data(), d_dWd.data(), s);
        attn.step(oq, ok, ov, oo, d_dWq.data(), d_dWk.data(), d_dWv.data(),
                  d_dWo.data(), s);
    };

    auto evaluate = [&]() {
        attn.forward(d_X.data(), d_att.data(), m, 1);
        mlp.forward(d_att.data(), d_logits.data(), m, 1);
        std::vector<float> h_logits(static_cast<size_t>(m) * hidden);
        d_logits.copy_to(h_logits.data(), h_logits.size());
        std::vector<float> lo(static_cast<size_t>(m));
        float loss = cross_entropy_loss(h_logits.data(), targets.data(),
                                              lo.data(), m, hidden, ce);
        int correct = 0;
        for (int i = 0; i < m; ++i) {
            const float* row = h_logits.data() + static_cast<size_t>(i) * hidden;
            int argmax = 0;
            for (int c = 1; c < hidden; ++c)
                if (row[c] > row[argmax]) argmax = c;
            if (argmax == targets[i]) ++correct;
        }
        return std::pair<float, float>(loss, static_cast<float>(correct) / m);
    };

    auto [loss0, acc0] = evaluate();
    for (int s = 1; s <= K; ++s) train_step(s);
    auto [lossK, accK] = evaluate();

    EXPECT_LT(lossK, loss0) << "mini-transformer loss must descend";
    EXPECT_GT(accK, acc0 + 0.1f) << "attention+MLP accuracy must rise materially";
    EXPECT_GT(accK, 0.4f);
}
