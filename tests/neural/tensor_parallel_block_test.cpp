/**
 * @file tensor_parallel_block_test.cpp
 * @brief Single-GPU (tp == 1) tests for the v2.28 pre-LN residual
 *        TransformerBlock and the deep TransformerTrainer
 *
 * Milestone v2.28 (TASK-040 / DEC-014): the v2.26 mini-transformer was a
 * single unnormalized block — attention -> MLP, no LayerNorm, no residual.
 * This file pins the NEW contracts on the tp <= 1 path (uninitialized
 * NcclContext -> effective tp 1):
 *
 *   1. TransformerBlock forward:  y = x + MLP(LN2(x + MHA(LN1(x)))) equals a
 *      host fp64 whole-block reference (independent LN + attention + MLP),
 *   2. TransformerBlock backward: the block grad-input and all eleven weight
 *      grads equal the host fp64 analytic reference (this is what pins the
 *      residual backprop wiring — the composition bug class),
 *   3. TransformerTrainer (N blocks + final LN) drives loss descent and
 *      accuracy rise on the learnable synthetic classification task.
 *
 * The multi-GPU shard==single-GPU parity lives in
 * tensor_parallel_block_multigpu_test.cpp.
 *
 * P1-RED: TransformerBlock/TransformerTrainer are provisional throw-stubs, so
 * these tests fail here and turn GREEN with the P2 composition (TASK-041).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/tensor_parallel_block.h"
#include "cuda/neural/training_transformer.h"
#include "cuda/neural/layer_norm.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::neural;
using namespace cuda::neural::optimizers;
using namespace cuda::neural::training;

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

bool arrays_near(const float* expected, const float* actual, size_t n,
                 float tol = 2e-3f) {
    for (size_t i = 0; i < n; ++i) {
        const float ref = expected[i];
        const float diff = std::fabs(actual[i] - ref);
        if (diff > tol && diff > tol * std::fabs(ref)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Host fp64 primitives (independent of the device stack)
// ============================================================================

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

void host_linear_backward(const float* X, const float* W, const float* dY,
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

// ---------------------------------------------------------------------------
// Host fp64 multi-head self-attention forward/backward (VERBATIM from the
// v2.26 tensor_parallel_attention_test.cpp references, which the green v2.26
// suite validates).
// ---------------------------------------------------------------------------
void host_attention_forward(const float* X, const float* Wq, const float* Wk,
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
                for (int j = 0; j < m; ++j)
                    acc += static_cast<double>(scores[j]) * v_base[j * stride + d];
                att_row[d] = static_cast<float>(acc);
            }
        }
    }
    host_matmul(att.data(), Wo, out, m, hidden, qkv);
}

void host_attention_backward(const float* X, const float* Wq, const float* Wk,
                             const float* Wv, const float* Wo, const float* dout,
                             float* dX, float* dWq, float* dWk, float* dWv,
                             float* dWo, int m, int heads, int hd, int hidden) {
    const int qkv = heads * hd;
    const int stride = qkv;
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
    std::vector<float> dQ(m * qkv, 0.0f), dK(m * qkv, 0.0f), dV(m * qkv, 0.0f);
    std::vector<float> datt(m * qkv);
    host_linear_backward(att.data(), Wo, dout, datt.data(), dWo, m, qkv, hidden);
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
    std::vector<float> tmp(qkv * hidden);
    host_linear_backward(X, Wq, dQ.data(), dX, dWq, m, hidden, qkv);
    std::vector<float> dXk(m * hidden), dXv(m * hidden);
    host_linear_backward(X, Wk, dK.data(), dXk.data(), dWk, m, hidden, qkv);
    host_linear_backward(X, Wv, dV.data(), dXv.data(), dWv, m, hidden, qkv);
    for (int i = 0; i < m * hidden; ++i) {
        dX[i] = dX[i] + dXk[i] + dXv[i];
    }
}

// ---------------------------------------------------------------------------
// Host fp64 gated-MLP (silu(gate) .* up -> down), matching the v2.21 layers.
// ---------------------------------------------------------------------------
void host_mlp_forward(const float* X, const float* Wg, const float* Wu,
                      const float* Wd, float* out, int m, int h, int inter) {
    std::vector<float> gate(m * inter), up(m * inter), sub(m * inter);
    host_matmul(X, Wg, gate.data(), m, inter, h);
    host_matmul(X, Wu, up.data(), m, inter, h);
    for (size_t i = 0; i < sub.size(); ++i) {
        const double g = gate[i];
        sub[i] = static_cast<float>((g / (1.0 + std::exp(-g))) * up[i]);
    }
    host_matmul(sub.data(), Wd, out, m, h, inter);
}

void host_mlp_backward(const float* X, const float* Wg, const float* Wu,
                       const float* Wd, const float* dout, float* dX,
                       float* dWg, float* dWu, float* dWd, int m, int h,
                       int inter) {
    std::vector<float> gate(m * inter), up(m * inter), sub(m * inter);
    host_matmul(X, Wg, gate.data(), m, inter, h);
    host_matmul(X, Wu, up.data(), m, inter, h);
    for (size_t i = 0; i < sub.size(); ++i) {
        const double g = gate[i];
        sub[i] = static_cast<float>((g / (1.0 + std::exp(-g))) * up[i]);
    }
    // dsub = dout Wd^T ; dWd = sub^T dout
    std::vector<float> dsub(m * inter);
    host_linear_backward(sub.data(), Wd, dout, dsub.data(), dWd, m, inter, h);
    std::vector<float> dgate(m * inter), dup(m * inter);
    for (size_t i = 0; i < dsub.size(); ++i) {
        const double g = gate[i];
        const double s = 1.0 / (1.0 + std::exp(-g));
        const double sp = s * (1.0 - s);
        dgate[i] = static_cast<float>(dsub[i] * up[i] * (s + g * sp));
        dup[i] = static_cast<float>(dsub[i] * (g / (1.0 + std::exp(-g))));
    }
    host_linear_backward(X, Wg, dgate.data(), dX, dWg, m, h, inter);
    std::vector<float> dXu(m * h);
    host_linear_backward(X, Wu, dup.data(), dXu.data(), dWu, m, h, inter);
    for (int i = 0; i < m * h; ++i) dX[i] += dXu[i];
}

// ---------------------------------------------------------------------------
// Host fp64 per-row LayerNorm forward/backward.
// ---------------------------------------------------------------------------
void host_ln_forward(const float* x, const float* gamma, const float* beta,
                     float* y, int m, int h, float eps) {
    for (int i = 0; i < m; ++i) {
        const float* xr = x + i * h;
        double sum = 0.0;
        for (int j = 0; j < h; ++j) sum += xr[j];
        const double mean = sum / h;
        double var = 0.0;
        for (int j = 0; j < h; ++j) {
            const double d = static_cast<double>(xr[j]) - mean;
            var += d * d;
        }
        var /= h;
        const double inv = 1.0 / std::sqrt(var + eps);
        float* yr = y + i * h;
        for (int j = 0; j < h; ++j) {
            const double xhat = (static_cast<double>(xr[j]) - mean) * inv;
            yr[j] = static_cast<float>(static_cast<double>(gamma[j]) * xhat +
                                       static_cast<double>(beta[j]));
        }
    }
}

void host_ln_backward(const float* x, const float* gamma, const float* dy,
                      float* dx, float* dgamma, float* dbeta, int m, int h,
                      float eps) {
    std::vector<double> mean(m), inv(m);
    for (int i = 0; i < m; ++i) {
        const float* xr = x + i * h;
        double s = 0.0;
        for (int j = 0; j < h; ++j) s += static_cast<double>(xr[j]);
        mean[i] = s / h;
        double v = 0.0;
        for (int j = 0; j < h; ++j) {
            const double d = static_cast<double>(xr[j]) - mean[i];
            v += d * d;
        }
        inv[i] = 1.0 / std::sqrt(v / h + eps);
    }
    for (int j = 0; j < h; ++j) {
        dgamma[j] = 0.0f;
        dbeta[j] = 0.0f;
        for (int i = 0; i < m; ++i) {
            const float* xr = x + i * h;
            const double xhat = (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            dgamma[j] += static_cast<float>(static_cast<double>(dy[i * h + j]) * xhat);
            dbeta[j] += dy[i * h + j];
        }
    }
    for (int i = 0; i < m; ++i) {
        const float* xr = x + i * h;
        const float* dyr = dy + i * h;
        float* dxr = dx + i * h;
        double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
        std::vector<double> dxhat(h);
        for (int j = 0; j < h; ++j) {
            const double xhat = (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            dxhat[j] = static_cast<double>(dyr[j]) * static_cast<double>(gamma[j]);
            sum_dxhat += dxhat[j];
            sum_dxhat_xhat += dxhat[j] * xhat;
        }
        for (int j = 0; j < h; ++j) {
            const double xhat = (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            const double v = inv[i] *
                (dxhat[j] - sum_dxhat / h - xhat * (sum_dxhat_xhat / h));
            dxr[j] = static_cast<float>(v);
        }
    }
}

// ============================================================================
// Host fp64 whole-block reference:
//   out = r1 + MLP(LN2(r1)) where r1 = x + MHA(LN1(x))
// ============================================================================
void host_block_forward(const float* X, const float* l1g, const float* l1b,
                        const float* Wq, const float* Wk, const float* Wv,
                        const float* Wo, const float* l2g, const float* l2b,
                        const float* Wg, const float* Wu, const float* Wd,
                        float* out, int m, int heads, int hd, int hidden,
                        int inter, float eps) {
    std::vector<float> h1(static_cast<size_t>(m) * hidden),
        a(static_cast<size_t>(m) * hidden), r1(static_cast<size_t>(m) * hidden),
        h2(static_cast<size_t>(m) * hidden), mo(static_cast<size_t>(m) * hidden);
    host_ln_forward(X, l1g, l1b, h1.data(), m, hidden, eps);
    host_attention_forward(h1.data(), Wq, Wk, Wv, Wo, a.data(), m, heads, hd,
                           hidden);
    for (int i = 0; i < m * hidden; ++i) r1[i] = X[i] + a[i];
    host_ln_forward(r1.data(), l2g, l2b, h2.data(), m, hidden, eps);
    host_mlp_forward(h2.data(), Wg, Wu, Wd, mo.data(), m, hidden, inter);
    for (int i = 0; i < m * hidden; ++i) out[i] = r1[i] + mo[i];
}

void host_block_backward(
    const float* X, const float* l1g, const float* l1b, const float* Wq,
    const float* Wk, const float* Wv, const float* Wo, const float* l2g,
    const float* l2b, const float* Wg, const float* Wu, const float* Wd,
    const float* dout, float* dX, float* dl1g, float* dl1b, float* dWq,
    float* dWk, float* dWv, float* dWo, float* dl2g, float* dl2b, float* dWg,
    float* dWu, float* dWd, int m, int heads, int hd, int hidden, int inter,
    float eps) {
    const int total = m * hidden;
    std::vector<float> h1(total), a(total), r1(total), h2(total), mo(total);
    host_ln_forward(X, l1g, l1b, h1.data(), m, hidden, eps);
    host_attention_forward(h1.data(), Wq, Wk, Wv, Wo, a.data(), m, heads, hd,
                           hidden);
    for (int i = 0; i < total; ++i) r1[i] = X[i] + a[i];
    host_ln_forward(r1.data(), l2g, l2b, h2.data(), m, hidden, eps);
    host_mlp_forward(h2.data(), Wg, Wu, Wd, mo.data(), m, hidden, inter);

    // y = r1 + mo: MLP backward feeds dL/dLN2-out = dout (dL/dmo).
    std::vector<float> dH2(total), dR1_ln(total), dR1(total), dA_attn(total),
        dX_ln(total);
    host_mlp_backward(h2.data(), Wg, Wu, Wd, dout, dH2.data(), dWg, dWu, dWd,
                      m, hidden, inter);
    host_ln_backward(r1.data(), l2g, dH2.data(), dR1_ln.data(), dl2g, dl2b,
                     m, hidden, eps);
    for (int i = 0; i < total; ++i) dR1[i] = dout[i] + dR1_ln[i];
    // r1 = x + a: attention backward consumes dR1 (dL/da); LN1 paths into x.
    host_attention_backward(h1.data(), Wq, Wk, Wv, Wo, dR1.data(),
                            dA_attn.data(), dWq, dWk, dWv, dWo, m, heads, hd,
                            hidden);
    host_ln_backward(X, l1g, dA_attn.data(), dX_ln.data(), dl1g, dl1b, m,
                     hidden, eps);
    for (int i = 0; i < total; ++i) dX[i] = dR1[i] + dX_ln[i];
}

}  // namespace

class TransformerBlockTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { cudaFree(nullptr); }
};

// Device block forward == host fp64 whole-block reference.
TEST_F(TransformerBlockTest, BlockForwardMatchesHostReference) {
    const int m = 8, hidden = 8, heads = 4, hd = 2, inter = 16, qkv = hidden;
    const float eps = 1e-5f;
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<float> l1g(hidden), l1b(hidden), l2g(hidden), l2b(hidden);
    std::vector<float> Wq(static_cast<size_t>(hidden) * qkv),
        Wk(static_cast<size_t>(hidden) * qkv), Wv(static_cast<size_t>(hidden) * qkv),
        Wo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> Wg(static_cast<size_t>(hidden) * inter),
        Wu(static_cast<size_t>(hidden) * inter),
        Wd(static_cast<size_t>(inter) * hidden);
    fill_random(X.data(), X.size(), 51);
    fill_random(l1g.data(), l1g.size(), 52);
    fill_random(l1b.data(), l1b.size(), 53);
    fill_random(l2g.data(), l2g.size(), 54);
    fill_random(l2b.data(), l2b.size(), 55);
    fill_random(Wq.data(), Wq.size(), 56);
    fill_random(Wk.data(), Wk.size(), 57);
    fill_random(Wv.data(), Wv.size(), 58);
    fill_random(Wo.data(), Wo.size(), 59);
    fill_random(Wg.data(), Wg.size(), 60);
    fill_random(Wu.data(), Wu.size(), 61);
    fill_random(Wd.data(), Wd.size(), 62);

    std::vector<float> ref(static_cast<size_t>(m) * hidden);
    host_block_forward(X.data(), l1g.data(), l1b.data(), Wq.data(), Wk.data(),
                       Wv.data(), Wo.data(), l2g.data(), l2b.data(), Wg.data(),
                       Wu.data(), Wd.data(), ref.data(), m, heads, hd, hidden,
                       inter, eps);

    TransformerBlock block(uninitialized_ctx(), hidden, heads, hd, inter, eps);
    block.set_weight(l1g.data(), l1b.data(), Wq.data(), Wk.data(), Wv.data(),
                     Wo.data(), l2g.data(), l2b.data(), Wg.data(), Wu.data(),
                     Wd.data());
    cuda::memory::Buffer<float> d_x(m * hidden), d_y(m * hidden);
    d_x.copy_from(X.data(), X.size());
    block.forward(d_x.data(), d_y.data(), m, 1);
    std::vector<float> out(static_cast<size_t>(m) * hidden);
    d_y.copy_to(out.data(), out.size());

    EXPECT_TRUE(arrays_near(ref.data(), out.data(), out.size(), 2e-3f))
        << "block forward differs from the host fp64 whole-block reference";
}

// Device block backward == host fp64 block reference (grad-input + 11 grads).
TEST_F(TransformerBlockTest, BlockBackwardMatchesHostReference) {
    const int m = 8, hidden = 8, heads = 4, hd = 2, inter = 16, qkv = hidden;
    const float eps = 1e-5f;
    std::vector<float> X(static_cast<size_t>(m) * hidden),
        dout(static_cast<size_t>(m) * hidden);
    std::vector<float> l1g(hidden), l1b(hidden), l2g(hidden), l2b(hidden);
    std::vector<float> Wq(static_cast<size_t>(hidden) * qkv),
        Wk(static_cast<size_t>(hidden) * qkv), Wv(static_cast<size_t>(hidden) * qkv),
        Wo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> Wg(static_cast<size_t>(hidden) * inter),
        Wu(static_cast<size_t>(hidden) * inter),
        Wd(static_cast<size_t>(inter) * hidden);
    fill_random(X.data(), X.size(), 71);
    fill_random(dout.data(), dout.size(), 72);
    fill_random(l1g.data(), l1g.size(), 73);
    fill_random(l1b.data(), l1b.size(), 74);
    fill_random(l2g.data(), l2g.size(), 75);
    fill_random(l2b.data(), l2b.size(), 76);
    fill_random(Wq.data(), Wq.size(), 77);
    fill_random(Wk.data(), Wk.size(), 78);
    fill_random(Wv.data(), Wv.size(), 79);
    fill_random(Wo.data(), Wo.size(), 80);
    fill_random(Wg.data(), Wg.size(), 81);
    fill_random(Wu.data(), Wu.size(), 82);
    fill_random(Wd.data(), Wd.size(), 83);

    std::vector<float> ref_dX(static_cast<size_t>(m) * hidden),
        ref_dl1g(hidden), ref_dl1b(hidden), ref_dl2g(hidden), ref_dl2b(hidden),
        ref_dWq(static_cast<size_t>(hidden) * qkv),
        ref_dWk(static_cast<size_t>(hidden) * qkv),
        ref_dWv(static_cast<size_t>(hidden) * qkv),
        ref_dWo(static_cast<size_t>(qkv) * hidden),
        ref_dWg(static_cast<size_t>(hidden) * inter),
        ref_dWu(static_cast<size_t>(hidden) * inter),
        ref_dWd(static_cast<size_t>(inter) * hidden);
    host_block_backward(
        X.data(), l1g.data(), l1b.data(), Wq.data(), Wk.data(), Wv.data(),
        Wo.data(), l2g.data(), l2b.data(), Wg.data(), Wu.data(), Wd.data(),
        dout.data(), ref_dX.data(), ref_dl1g.data(), ref_dl1b.data(),
        ref_dWq.data(), ref_dWk.data(), ref_dWv.data(), ref_dWo.data(),
        ref_dl2g.data(), ref_dl2b.data(), ref_dWg.data(), ref_dWu.data(),
        ref_dWd.data(), m, heads, hd, hidden, inter, eps);

    TransformerBlock block(uninitialized_ctx(), hidden, heads, hd, inter, eps);
    block.set_weight(l1g.data(), l1b.data(), Wq.data(), Wk.data(), Wv.data(),
                     Wo.data(), l2g.data(), l2b.data(), Wg.data(), Wu.data(),
                     Wd.data());
    cuda::memory::Buffer<float> d_x(m * hidden), d_dout(m * hidden),
        d_dx(m * hidden), d_out(m * hidden);
    cuda::memory::Buffer<float> d_dl1g(hidden), d_dl1b(hidden),
        d_dl2g(hidden), d_dl2b(hidden);
    cuda::memory::Buffer<float> d_dWq((size_t)hidden * qkv),
        d_dWk((size_t)hidden * qkv), d_dWv((size_t)hidden * qkv),
        d_dWo((size_t)qkv * hidden);
    cuda::memory::Buffer<float> d_dWg((size_t)hidden * inter),
        d_dWu((size_t)hidden * inter), d_dWd((size_t)inter * hidden);
    d_x.copy_from(X.data(), X.size());
    d_dout.copy_from(dout.data(), dout.size());
    block.forward(d_x.data(), d_out.data(), m, 1);  // scratch
    block.backward(d_x.data(), d_dout.data(), d_dx.data(), d_dl1g.data(),
                   d_dl1b.data(), d_dWq.data(), d_dWk.data(), d_dWv.data(),
                   d_dWo.data(), d_dl2g.data(), d_dl2b.data(), d_dWg.data(),
                   d_dWu.data(), d_dWd.data(), m, 1);

    std::vector<float> dX(static_cast<size_t>(m) * hidden), dl1g(hidden),
        dl1b(hidden), dl2g(hidden), dl2b(hidden);
    std::vector<float> dWq(static_cast<size_t>(hidden) * qkv),
        dWk(static_cast<size_t>(hidden) * qkv), dWv(static_cast<size_t>(hidden) * qkv),
        dWo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> dWg(static_cast<size_t>(hidden) * inter),
        dWu(static_cast<size_t>(hidden) * inter),
        dWd(static_cast<size_t>(inter) * hidden);
    d_dx.copy_to(dX.data(), dX.size());
    d_dl1g.copy_to(dl1g.data(), dl1g.size());
    d_dl1b.copy_to(dl1b.data(), dl1b.size());
    d_dl2g.copy_to(dl2g.data(), dl2g.size());
    d_dl2b.copy_to(dl2b.data(), dl2b.size());
    d_dWq.copy_to(dWq.data(), dWq.size());
    d_dWk.copy_to(dWk.data(), dWk.size());
    d_dWv.copy_to(dWv.data(), dWv.size());
    d_dWo.copy_to(dWo.data(), dWo.size());
    d_dWg.copy_to(dWg.data(), dWg.size());
    d_dWu.copy_to(dWu.data(), dWu.size());
    d_dWd.copy_to(dWd.data(), dWd.size());

    EXPECT_TRUE(arrays_near(ref_dWq.data(), dWq.data(), dWq.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWk.data(), dWk.data(), dWk.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWv.data(), dWv.data(), dWv.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWo.data(), dWo.data(), dWo.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWg.data(), dWg.data(), dWg.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWu.data(), dWu.data(), dWu.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWd.data(), dWd.data(), dWd.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl1g.data(), dl1g.data(), dl1g.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl1b.data(), dl1b.data(), dl1b.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl2g.data(), dl2g.data(), dl2g.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl2b.data(), dl2b.data(), dl2b.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dX.data(), dX.data(), dX.size(), 2e-3f))
        << "block grad-input differs from the host fp64 reference";
}

// The block must also be correct when seq > 1 (batch*seq rows): LayerNorm and
// the residual adds normalize/sum over every [batch*seq x hidden] row, and the
// MHA/MLP consume all m = batch*seq rows. Regression for the reviewer C1 fix
// (TransformerBlock had used batch as the row count -> device OOB at seq > 1).
TEST_F(TransformerBlockTest, BlockForwardMatchesHostReferenceSeq2) {
    const int batch = 2, seq = 4, m = batch * seq;
    const int hidden = 8, heads = 4, hd = 2, inter = 16, qkv = hidden;
    const float eps = 1e-5f;
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<float> l1g(hidden), l1b(hidden), l2g(hidden), l2b(hidden);
    std::vector<float> Wq(static_cast<size_t>(hidden) * qkv),
        Wk(static_cast<size_t>(hidden) * qkv), Wv(static_cast<size_t>(hidden) * qkv),
        Wo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> Wg(static_cast<size_t>(hidden) * inter),
        Wu(static_cast<size_t>(hidden) * inter),
        Wd(static_cast<size_t>(inter) * hidden);
    fill_random(X.data(), X.size(), 501);
    fill_random(l1g.data(), l1g.size(), 502);
    fill_random(l1b.data(), l1b.size(), 503);
    fill_random(l2g.data(), l2g.size(), 504);
    fill_random(l2b.data(), l2b.size(), 505);
    fill_random(Wq.data(), Wq.size(), 506);
    fill_random(Wk.data(), Wk.size(), 507);
    fill_random(Wv.data(), Wv.size(), 508);
    fill_random(Wo.data(), Wo.size(), 509);
    fill_random(Wg.data(), Wg.size(), 510);
    fill_random(Wu.data(), Wu.size(), 511);
    fill_random(Wd.data(), Wd.size(), 512);

    std::vector<float> ref(static_cast<size_t>(m) * hidden);
    host_block_forward(X.data(), l1g.data(), l1b.data(), Wq.data(), Wk.data(),
                       Wv.data(), Wo.data(), l2g.data(), l2b.data(), Wg.data(),
                       Wu.data(), Wd.data(), ref.data(), m, heads, hd, hidden,
                       inter, eps);

    TransformerBlock block(uninitialized_ctx(), hidden, heads, hd, inter, eps);
    block.set_weight(l1g.data(), l1b.data(), Wq.data(), Wk.data(), Wv.data(),
                     Wo.data(), l2g.data(), l2b.data(), Wg.data(), Wu.data(),
                     Wd.data());
    cuda::memory::Buffer<float> d_x(m * hidden), d_y(m * hidden);
    d_x.copy_from(X.data(), X.size());
    block.forward(d_x.data(), d_y.data(), batch, seq);
    std::vector<float> out(static_cast<size_t>(m) * hidden);
    d_y.copy_to(out.data(), out.size());

    EXPECT_TRUE(arrays_near(ref.data(), out.data(), out.size(), 2e-3f))
        << "block forward (seq=2, batch*seq rows) differs from the host fp64 reference";
}

TEST_F(TransformerBlockTest, BlockBackwardMatchesHostReferenceSeq2) {
    const int batch = 2, seq = 4, m = batch * seq;
    const int hidden = 8, heads = 4, hd = 2, inter = 16, qkv = hidden;
    const float eps = 1e-5f;
    std::vector<float> X(static_cast<size_t>(m) * hidden),
        dout(static_cast<size_t>(m) * hidden);
    std::vector<float> l1g(hidden), l1b(hidden), l2g(hidden), l2b(hidden);
    std::vector<float> Wq(static_cast<size_t>(hidden) * qkv),
        Wk(static_cast<size_t>(hidden) * qkv), Wv(static_cast<size_t>(hidden) * qkv),
        Wo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> Wg(static_cast<size_t>(hidden) * inter),
        Wu(static_cast<size_t>(hidden) * inter),
        Wd(static_cast<size_t>(inter) * hidden);
    fill_random(X.data(), X.size(), 601);
    fill_random(dout.data(), dout.size(), 602);
    fill_random(l1g.data(), l1g.size(), 603);
    fill_random(l1b.data(), l1b.size(), 604);
    fill_random(l2g.data(), l2g.size(), 605);
    fill_random(l2b.data(), l2b.size(), 606);
    fill_random(Wq.data(), Wq.size(), 607);
    fill_random(Wk.data(), Wk.size(), 608);
    fill_random(Wv.data(), Wv.size(), 609);
    fill_random(Wo.data(), Wo.size(), 610);
    fill_random(Wg.data(), Wg.size(), 611);
    fill_random(Wu.data(), Wu.size(), 612);
    fill_random(Wd.data(), Wd.size(), 613);

    std::vector<float> ref_dX(static_cast<size_t>(m) * hidden),
        ref_dl1g(hidden), ref_dl1b(hidden), ref_dl2g(hidden), ref_dl2b(hidden),
        ref_dWq(static_cast<size_t>(hidden) * qkv),
        ref_dWk(static_cast<size_t>(hidden) * qkv),
        ref_dWv(static_cast<size_t>(hidden) * qkv),
        ref_dWo(static_cast<size_t>(qkv) * hidden),
        ref_dWg(static_cast<size_t>(hidden) * inter),
        ref_dWu(static_cast<size_t>(hidden) * inter),
        ref_dWd(static_cast<size_t>(inter) * hidden);
    host_block_backward(
        X.data(), l1g.data(), l1b.data(), Wq.data(), Wk.data(), Wv.data(),
        Wo.data(), l2g.data(), l2b.data(), Wg.data(), Wu.data(), Wd.data(),
        dout.data(), ref_dX.data(), ref_dl1g.data(), ref_dl1b.data(),
        ref_dWq.data(), ref_dWk.data(), ref_dWv.data(), ref_dWo.data(),
        ref_dl2g.data(), ref_dl2b.data(), ref_dWg.data(), ref_dWu.data(),
        ref_dWd.data(), m, heads, hd, hidden, inter, eps);

    TransformerBlock block(uninitialized_ctx(), hidden, heads, hd, inter, eps);
    block.set_weight(l1g.data(), l1b.data(), Wq.data(), Wk.data(), Wv.data(),
                     Wo.data(), l2g.data(), l2b.data(), Wg.data(), Wu.data(),
                     Wd.data());
    cuda::memory::Buffer<float> d_x(m * hidden), d_dout(m * hidden),
        d_dx(m * hidden), d_out(m * hidden);
    cuda::memory::Buffer<float> d_dl1g(hidden), d_dl1b(hidden),
        d_dl2g(hidden), d_dl2b(hidden);
    cuda::memory::Buffer<float> d_dWq((size_t)hidden * qkv),
        d_dWk((size_t)hidden * qkv), d_dWv((size_t)hidden * qkv),
        d_dWo((size_t)qkv * hidden);
    cuda::memory::Buffer<float> d_dWg((size_t)hidden * inter),
        d_dWu((size_t)hidden * inter), d_dWd((size_t)inter * hidden);
    d_x.copy_from(X.data(), X.size());
    d_dout.copy_from(dout.data(), dout.size());
    block.forward(d_x.data(), d_out.data(), batch, seq);  // scratch
    block.backward(d_x.data(), d_dout.data(), d_dx.data(), d_dl1g.data(),
                   d_dl1b.data(), d_dWq.data(), d_dWk.data(), d_dWv.data(),
                   d_dWo.data(), d_dl2g.data(), d_dl2b.data(), d_dWg.data(),
                   d_dWu.data(), d_dWd.data(), batch, seq);

    std::vector<float> dX(static_cast<size_t>(m) * hidden), dl1g(hidden),
        dl1b(hidden), dl2g(hidden), dl2b(hidden);
    std::vector<float> dWq(static_cast<size_t>(hidden) * qkv),
        dWk(static_cast<size_t>(hidden) * qkv), dWv(static_cast<size_t>(hidden) * qkv),
        dWo(static_cast<size_t>(qkv) * hidden);
    std::vector<float> dWg(static_cast<size_t>(hidden) * inter),
        dWu(static_cast<size_t>(hidden) * inter),
        dWd(static_cast<size_t>(inter) * hidden);
    d_dx.copy_to(dX.data(), dX.size());
    d_dl1g.copy_to(dl1g.data(), dl1g.size());
    d_dl1b.copy_to(dl1b.data(), dl1b.size());
    d_dl2g.copy_to(dl2g.data(), dl2g.size());
    d_dl2b.copy_to(dl2b.data(), dl2b.size());
    d_dWq.copy_to(dWq.data(), dWq.size());
    d_dWk.copy_to(dWk.data(), dWk.size());
    d_dWv.copy_to(dWv.data(), dWv.size());
    d_dWo.copy_to(dWo.data(), dWo.size());
    d_dWg.copy_to(dWg.data(), dWg.size());
    d_dWu.copy_to(dWu.data(), dWu.size());
    d_dWd.copy_to(dWd.data(), dWd.size());

    EXPECT_TRUE(arrays_near(ref_dWq.data(), dWq.data(), dWq.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWk.data(), dWk.data(), dWk.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWv.data(), dWv.data(), dWv.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWo.data(), dWo.data(), dWo.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWg.data(), dWg.data(), dWg.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWu.data(), dWu.data(), dWu.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dWd.data(), dWd.data(), dWd.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl1g.data(), dl1g.data(), dl1g.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl1b.data(), dl1b.data(), dl1b.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl2g.data(), dl2g.data(), dl2g.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dl2b.data(), dl2b.data(), dl2b.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dX.data(), dX.data(), dX.size(), 2e-3f))
        << "block grad-input (seq=2) differs from the host fp64 reference";
}

// ============================================================================
// v2.28 capstone single-GPU (TASK-040): a DEEP (N-block, pre-LN, residual)
// mini-transformer — final LN head, logits == the [m x hidden] head output —
// must train on the tp <= 1 path: loss descends over K AdamW steps and top-1
// accuracy rises above chance. This is what v2.26's single unnormalized block
// could not do.
// ============================================================================
TEST_F(TransformerBlockTest, DeepTransformerTrainingConverges) {
    const int m = 16, hidden = 8, heads = 4, hd = 2, inter = 16, K = 60;
    const int blocks = 3;

    // Learnable task: gaussian inputs, fixed random "teacher" labels.
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<int> targets(m);
    {
        std::mt19937 rng(20260819);
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

    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;
    TransformerTrainer trainer(uninitialized_ctx(), blocks, hidden, heads, hd,
                               inter, cfg);

    // Random init all weights (deterministic seeds).
    const int qkv = hidden;
    std::vector<std::vector<float>> blg(blocks, std::vector<float>(hidden)),
        blb(blocks, std::vector<float>(hidden)),
        b2g(blocks, std::vector<float>(hidden)),
        b2b(blocks, std::vector<float>(hidden));
    std::vector<std::vector<float>> bWq(blocks, std::vector<float>(hidden * qkv)),
        bWk(blocks, std::vector<float>(hidden * qkv)),
        bWv(blocks, std::vector<float>(hidden * qkv)),
        bWo(blocks, std::vector<float>(qkv * hidden));
    std::vector<std::vector<float>> bWg(blocks, std::vector<float>(hidden * inter)),
        bWu(blocks, std::vector<float>(hidden * inter)),
        bWd(blocks, std::vector<float>(inter * hidden));
    std::vector<float> fg(hidden, 1.0f), fb(hidden, 0.0f);
    unsigned seed = 91;
    for (int b = 0; b < blocks; ++b) {
        fill_random(blg[b].data(), blg[b].size(), seed++);
        fill_random(blb[b].data(), blb[b].size(), seed++);
        fill_random(b2g[b].data(), b2g[b].size(), seed++);
        fill_random(b2b[b].data(), b2b[b].size(), seed++);
        fill_random(bWq[b].data(), bWq[b].size(), seed++);
        fill_random(bWk[b].data(), bWk[b].size(), seed++);
        fill_random(bWv[b].data(), bWv[b].size(), seed++);
        fill_random(bWo[b].data(), bWo[b].size(), seed++);
        fill_random(bWg[b].data(), bWg[b].size(), seed++);
        fill_random(bWu[b].data(), bWu[b].size(), seed++);
        fill_random(bWd[b].data(), bWd[b].size(), seed++);
        trainer.set_block_weight(b, blg[b].data(), blb[b].data(), bWq[b].data(),
                                 bWk[b].data(), bWv[b].data(), bWo[b].data(),
                                 b2g[b].data(), b2b[b].data(), bWg[b].data(),
                                 bWu[b].data(), bWd[b].data());
    }
    trainer.set_final_ln_weight(fg.data(), fb.data());

    cuda::memory::Buffer<float> d_X(m * hidden);
    d_X.copy_from(X.data(), X.size());

    float loss_0 = 0.0f, acc_0 = 0.0f, loss_k = 0.0f, acc_k = 0.0f;
    loss_0 = trainer.evaluate(d_X.data(), targets.data(), m, 1, &acc_0);
    for (int s = 1; s <= K; ++s) {
        trainer.train_step(d_X.data(), targets.data(), m, 1, s);
    }
    loss_k = trainer.evaluate(d_X.data(), targets.data(), m, 1, &acc_k);

    // Chance accuracy for 8 classes is 1/8 = 0.125.
    EXPECT_LT(loss_k, loss_0) << "deep transformer loss must descend over training";
    EXPECT_GT(acc_k, 0.5f) << "deep transformer accuracy must rise well above chance (1/8)";
    EXPECT_GT(acc_k, acc_0 + 0.1f);
}

// seq > 1 must work through the whole trainer (regression for the reviewer C1
// fix: blocks previously sized/normalized on batch rows only, an OOB at
// batch*seq rows). A few seq=2 train_steps must return finite losses that
// actually move (the teacher task is learnable at fixed seq).
TEST_F(TransformerBlockTest, TrainerSupportsSeqGreaterThanOne) {
    const int batch = 4, seq = 2, m = batch * seq;
    const int hidden = 8, heads = 4, hd = 2, inter = 16;
    const int blocks = 2;
    std::vector<float> X;
    std::vector<int> targets;
    {
        std::mt19937 rng(20260822);
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::vector<float> T(static_cast<size_t>(hidden) * hidden);
        for (auto& t : T) t = norm(rng);
        X.assign(static_cast<size_t>(m) * hidden, 0.0f);
        targets.assign(m, 0);
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
    OptimizerConfig cfg;
    cfg.learning_rate = 0.05f;
    TransformerTrainer trainer(uninitialized_ctx(), blocks, hidden, heads, hd,
                               inter, cfg);
    const int qkv = hidden;
    unsigned seed = 700;
    for (int b = 0; b < blocks; ++b) {
        std::vector<float> l1g(hidden), l1b(hidden), l2g(hidden), l2b(hidden);
        std::vector<float> Wq(hidden * qkv), Wk(hidden * qkv), Wv(hidden * qkv),
            Wo(qkv * hidden), Wg(hidden * inter), Wu(hidden * inter),
            Wd(inter * hidden);
        fill_random(l1g.data(), l1g.size(), seed++);
        fill_random(l1b.data(), l1b.size(), seed++);
        fill_random(l2g.data(), l2g.size(), seed++);
        fill_random(l2b.data(), l2b.size(), seed++);
        fill_random(Wq.data(), Wq.size(), seed++);
        fill_random(Wk.data(), Wk.size(), seed++);
        fill_random(Wv.data(), Wv.size(), seed++);
        fill_random(Wo.data(), Wo.size(), seed++);
        fill_random(Wg.data(), Wg.size(), seed++);
        fill_random(Wu.data(), Wu.size(), seed++);
        fill_random(Wd.data(), Wd.size(), seed++);
        trainer.set_block_weight(b, l1g.data(), l1b.data(), Wq.data(),
                                 Wk.data(), Wv.data(), Wo.data(), l2g.data(),
                                 l2b.data(), Wg.data(), Wu.data(), Wd.data());
    }
    std::vector<float> fg(hidden, 1.0f), fb(hidden, 0.0f);
    trainer.set_final_ln_weight(fg.data(), fb.data());

    cuda::memory::Buffer<float> d_X(m * hidden);
    d_X.copy_from(X.data(), X.size());

    float l0 = trainer.evaluate(d_X.data(), targets.data(), batch, seq, nullptr);
    EXPECT_TRUE(std::isfinite(l0));
    float l1 = trainer.train_step(d_X.data(), targets.data(), batch, seq, 1);
    float lk = l1;
    for (int s = 2; s <= 8; ++s) {
        lk = trainer.train_step(d_X.data(), targets.data(), batch, seq, s);
        EXPECT_TRUE(std::isfinite(lk));
    }
    EXPECT_LT(lk, l0) << "seq=2 deep transformer must learn (loss descend)";
}
