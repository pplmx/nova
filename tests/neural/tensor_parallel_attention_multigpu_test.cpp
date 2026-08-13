/**
 * @file tensor_parallel_attention_multigpu_test.cpp
 * @brief Real thread-per-rank multi-GPU tests for TensorParallelMultiHeadAttention
 *
 * Milestone v2.26 (TASK-032 / DEC-012): the new TP attention block must match
 * a single-GPU full-weight reference on real multi-GPU, exactly like the
 * v2.21-25 layers. Each rank owns fully-local head shards (QKV column-
 * parallel), so the per-head SDPA has no cross-rank communication; only the
 * output row-projection AllReduces. Assembled forward outputs and backward
 * grad-weight slices must equal a host fp64 full-weight reference.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/tensor_parallel_attention.h"
#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/neural/loss/loss_functions.h"
#include "cuda/neural/matmul.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include "distributed_test_common.h"

using namespace cuda::neural;
using namespace cuda::neural::optimizers;
using namespace cuda::neural::loss;
using namespace cuda::mesh;

using cuda::distributed::test::run_per_rank;

namespace {

// A fresh uninitialized NcclContext reports 0 devices -> effective tp 1. Used
// for the single-GPU full-weight reference run inside this file.
cuda::nccl::NcclContext& uninitialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

bool multigpu_nccl_ready() {
    auto& ctx = cuda::nccl::NcclContext::instance();
    try {
        ctx.initialize();
    } catch (const std::exception&) {
        return false;
    }
    return cuda::distributed::test::heal_and_ready(ctx);
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

// Host fp64 multi-head self-attention forward (full weights).
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

// Host fp64 attention backward for weight grads only (assembled slices).
void ref_attention_backward_weights(const float* X, const float* Wq,
                                    const float* Wk, const float* Wv,
                                    const float* Wo, const float* dout,
                                    float* dWq, float* dWk, float* dWv,
                                    float* dWo, int m, int heads, int hd,
                                    int hidden) {
    const int qkv = heads * hd;
    const int stride = qkv;  // position-major stride for k/v
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
    std::vector<float> tmp(m * hidden);
    // dWo = att^T dout directly.
    for (int a = 0; a < qkv; ++a) {
        for (int b = 0; b < hidden; ++b) {
            double acc = 0.0;
            for (int i = 0; i < m; ++i) {
                acc += static_cast<double>(att[i * qkv + a]) * dout[i * hidden + b];
            }
            dWo[a * hidden + b] = static_cast<float>(acc);
        }
    }
    // datt = dout Wo^T
    for (int i = 0; i < m; ++i) {
        for (int a = 0; a < qkv; ++a) {
            double acc = 0.0;
            for (int b = 0; b < hidden; ++b) {
                acc += static_cast<double>(dout[i * hidden + b]) * Wo[a * hidden + b];
            }
            datt[i * qkv + a] = static_cast<float>(acc);
        }
    }
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
            std::vector<float> dP(m);
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
                const float ds = p[j] * (dP[j] - static_cast<float>(weighted)) * inv;
                float* dK_row = dK.data() + (j * heads + h) * hd;
                for (int d = 0; d < hd; ++d) {
                    dq_row[d] += ds * k_base[j * stride + d];
                    dK_row[d] += ds * q_row[d];
                }
            }
        }
    }
    for (int a = 0; a < hidden; ++a) {
        for (int c = 0; c < qkv; ++c) {
            double aq = 0.0, ak = 0.0, av = 0.0;
            for (int i = 0; i < m; ++i) {
                aq += static_cast<double>(X[i * hidden + a]) * dQ[i * qkv + c];
                ak += static_cast<double>(X[i * hidden + a]) * dK[i * qkv + c];
                av += static_cast<double>(X[i * hidden + a]) * dV[i * qkv + c];
            }
            dWq[a * qkv + c] = static_cast<float>(aq);
            dWk[a * qkv + c] = static_cast<float>(ak);
            dWv[a * qkv + c] = static_cast<float>(av);
        }
    }
}

}  // namespace

class TensorParallelAttentionMultiGpuTest : public ::testing::Test {
protected:
    void SetUp() override { DeviceMesh::instance().initialize(); }

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

// Assembled multi-GPU attention forward must equal the single-GPU full-weight
// reference (per-rank heads are local; only the output row-layer AllReduces).
TEST_F(TensorParallelAttentionMultiGpuTest, MultiGpu_AttentionForwardMatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU attention test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU attention requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 4, heads = 4, hd = 2, hidden = 8;  // heads divisible by tp
    std::vector<float> h_X(m * hidden);
    std::vector<float> h_Wq(hidden * heads * hd), h_Wk(hidden * heads * hd),
        h_Wv(hidden * heads * hd), h_Wo(heads * hd * hidden);
    fill_random(h_X.data(), h_X.size(), 31);
    fill_random(h_Wq.data(), h_Wq.size(), 32);
    fill_random(h_Wk.data(), h_Wk.size(), 33);
    fill_random(h_Wv.data(), h_Wv.size(), 34);
    fill_random(h_Wo.data(), h_Wo.size(), 35);

    std::vector<float> h_ref(m * hidden);
    ref_attention_forward(h_X.data(), h_Wq.data(), h_Wk.data(), h_Wv.data(),
                          h_Wo.data(), h_ref.data(), m, heads, hd, hidden);

    std::vector<std::vector<float>> outs(device_count,
                                         std::vector<float>(m * hidden));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_X(m * hidden), d_out(m * hidden);
        d_X.copy_from(h_X.data(), h_X.size());
        TensorParallelMultiHeadAttention attn(ctx, heads, hd, hidden);
        attn.set_weight(h_Wq.data(), h_Wk.data(), h_Wv.data(), h_Wo.data());
        attn.forward(d_X.data(), d_out.data(), m, 1);
        d_out.copy_to(outs[d].data(), m * hidden);
    });

    for (int r = 0; r < device_count; ++r) {
        EXPECT_TRUE(arrays_near(h_ref.data(), outs[r].data(), m * hidden, 1e-2f))
            << "rank " << r << " attention output differs from the reference";
    }
}

// Assembled multi-GPU attention backward weight-grad shards must equal the
// single-GPU full-weight reference (each rank contributes its head-slice).
TEST_F(TensorParallelAttentionMultiGpuTest, MultiGpu_AttentionBackwardMatchesReference) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU attention test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU attention requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 4, heads = 4, hd = 2, hidden = 8;
    const int qkv = heads * hd;
    const int stride = qkv;  // position-major stride for k/v
    std::vector<float> h_X(m * hidden);
    std::vector<float> h_Wq(hidden * qkv), h_Wk(hidden * qkv), h_Wv(hidden * qkv),
        h_Wo(qkv * hidden);
    std::vector<float> h_dout(m * hidden);
    fill_random(h_X.data(), h_X.size(), 41);
    fill_random(h_Wq.data(), h_Wq.size(), 42);
    fill_random(h_Wk.data(), h_Wk.size(), 43);
    fill_random(h_Wv.data(), h_Wv.size(), 44);
    fill_random(h_Wo.data(), h_Wo.size(), 45);
    fill_random(h_dout.data(), h_dout.size(), 46);

    std::vector<float> ref_dWq(hidden * qkv), ref_dWk(hidden * qkv),
        ref_dWv(hidden * qkv), ref_dWo(qkv * hidden);
    ref_attention_backward_weights(h_X.data(), h_Wq.data(), h_Wk.data(),
                                   h_Wv.data(), h_Wo.data(), h_dout.data(),
                                   ref_dWq.data(), ref_dWk.data(), ref_dWv.data(),
                                   ref_dWo.data(), m, heads, hd, hidden);

    const int local_qkv = qkv / device_count;
    std::vector<std::vector<float>> dWq_s(device_count,
                                          std::vector<float>(hidden * local_qkv));
    std::vector<std::vector<float>> dWk_s(device_count,
                                          std::vector<float>(hidden * local_qkv));
    std::vector<std::vector<float>> dWv_s(device_count,
                                          std::vector<float>(hidden * local_qkv));
    std::vector<std::vector<float>> dWo_s(device_count,
                                          std::vector<float>(local_qkv * hidden));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_X(m * hidden), d_dout(m * hidden),
            d_dX(m * hidden), d_out(m * hidden);
        cuda::memory::Buffer<float> d_dWq(hidden * qkv), d_dWk(hidden * qkv),
            d_dWv(hidden * qkv), d_dWo(qkv * hidden);
        d_X.copy_from(h_X.data(), h_X.size());
        d_dout.copy_from(h_dout.data(), h_dout.size());
        TensorParallelMultiHeadAttention attn(ctx, heads, hd, hidden);
        attn.set_weight(h_Wq.data(), h_Wk.data(), h_Wv.data(), h_Wo.data());
        cuda::memory::Buffer<float> d_fwd(m * hidden);
        attn.forward(d_X.data(), d_fwd.data(), m, 1);  // scratch qkv (separate)
        attn.backward(d_X.data(), d_dout.data(), d_dX.data(), d_dWq.data(),
                      d_dWk.data(), d_dWv.data(), d_dWo.data(), m, 1);
        // The full buffers hold the rank slices; copy them back and slice on
        // host. QKV are column-parallel: rank r owns columns
        // [r*local_qkv, +local_qkv) of each [hidden x qkv] grad.
        std::vector<float> full_dWq(hidden * qkv), full_dWk(hidden * qkv),
            full_dWv(hidden * qkv), full_dWo(qkv * hidden);
        d_dWq.copy_to(full_dWq.data(), full_dWq.size());
        d_dWk.copy_to(full_dWk.data(), full_dWk.size());
        d_dWv.copy_to(full_dWv.data(), full_dWv.size());
        d_dWo.copy_to(full_dWo.data(), full_dWo.size());
        const int col_off = d * local_qkv;
        for (int a = 0; a < hidden; ++a) {
            for (int c = 0; c < local_qkv; ++c) {
                dWq_s[d][a * local_qkv + c] = full_dWq[a * qkv + col_off + c];
                dWk_s[d][a * local_qkv + c] = full_dWk[a * qkv + col_off + c];
                dWv_s[d][a * local_qkv + c] = full_dWv[a * qkv + col_off + c];
            }
        }
        // Rank r owns rows [r*local_qkv, +local_qkv) of the [qkv x hidden]
        // grad-wo matrix; the row-major byte offset is r*local_qkv*hidden.
        const int row_off = d * local_qkv * hidden;
        std::copy(full_dWo.begin() + row_off, full_dWo.begin() + row_off + local_qkv * hidden,
                  dWo_s[d].begin());
    });

    // Assemble and compare.
    std::vector<float> aWq(hidden * qkv), aWk(hidden * qkv), aWv(hidden * qkv),
        aWo(qkv * hidden);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < hidden; ++a) {
            for (int c = 0; c < local_qkv; ++c) {
                aWq[a * qkv + r * local_qkv + c] = dWq_s[r][a * local_qkv + c];
                aWk[a * qkv + r * local_qkv + c] = dWk_s[r][a * local_qkv + c];
                aWv[a * qkv + r * local_qkv + c] = dWv_s[r][a * local_qkv + c];
            }
        }
        for (int c = 0; c < local_qkv * hidden; ++c) {
            aWo[r * local_qkv * hidden + c] = dWo_s[r][c];
        }
    }
    EXPECT_TRUE(arrays_near(ref_dWq.data(), aWq.data(), hidden * qkv, 2e-2f));
    EXPECT_TRUE(arrays_near(ref_dWk.data(), aWk.data(), hidden * qkv, 2e-2f));
    EXPECT_TRUE(arrays_near(ref_dWv.data(), aWv.data(), hidden * qkv, 2e-2f));
    EXPECT_TRUE(arrays_near(ref_dWo.data(), aWo.data(), qkv * hidden, 2e-2f))
        << "assembled output-projection grad differs from the reference";
}

// ============================================================================
// v2.26 capstone multi-GPU (TASK-034): a mini-transformer — attention feeding
// the verified MLP, logits = MLP(attn(X)) — trains on real multi-GPU. Each
// rank runs K identical AdamW steps over its shards; the assembled per-rank
// weight shards after the run must match a single-GPU full-weight reference
// run (tp==1, uninitialized context), and the loss must descend. This pins
// "the trainable transformer trains the same sharded as full".
// ============================================================================
TEST_F(TensorParallelAttentionMultiGpuTest, MultiGpu_MiniTransformer_MatchesSingleGpu) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for mini-transformer training test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Mini-transformer training requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 12, hidden = 8, heads = 4, hd = 2, inter = 16, K = 12;
    const int qkv = heads * hd;  // == hidden

    // Learnable task (same generator as the single-GPU convergence test).
    std::vector<float> X(static_cast<size_t>(m) * hidden);
    std::vector<int> targets(m);
    {
        std::mt19937 rng(20260821);
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
    std::vector<float> Wq((size_t)hidden * qkv), Wk((size_t)hidden * qkv),
        Wv((size_t)hidden * qkv), Wo((size_t)qkv * hidden);
    std::vector<float> Wg((size_t)hidden * inter), Wu((size_t)hidden * inter),
        Wd((size_t)inter * hidden);
    fill_random(Wq.data(), Wq.size(), 71);
    fill_random(Wk.data(), Wk.size(), 72);
    fill_random(Wv.data(), Wv.size(), 73);
    fill_random(Wo.data(), Wo.size(), 74);
    fill_random(Wg.data(), Wg.size(), 75);
    fill_random(Wu.data(), Wu.size(), 76);
    fill_random(Wd.data(), Wd.size(), 77);

    OptimizerConfig cfg;
    cfg.learning_rate = 0.02f;

    // Single-GPU full-weight reference run (tp==1, no NCCL) on device 0.
    std::vector<float> rWq = Wq, rWk = Wk, rWv = Wv, rWo = Wo;
    std::vector<float> rWg = Wg, rWu = Wu, rWd = Wd;
    TensorParallelMultiHeadAttention ref_attn(uninitialized_ctx(), heads, hd, hidden);
    ref_attn.set_weight(rWq.data(), rWk.data(), rWv.data(), rWo.data());
    TensorParallelMLP ref_mlp(uninitialized_ctx(), hidden, inter);
    ref_mlp.set_weight(rWg.data(), rWu.data(), rWd.data());
    {
        AdamWOptimizer oq(cfg), ok(cfg), ov(cfg), oo(cfg), og(cfg), ou(cfg), od(cfg);
        cuda::memory::Buffer<float> dX(m * hidden), dA(m * hidden), dL(m * hidden),
            dD(m * hidden), dDA(m * hidden);
        cuda::memory::Buffer<float> dWq(hidden * qkv), dWk(hidden * qkv),
            dWv(hidden * qkv), dWo(qkv * hidden);
        cuda::memory::Buffer<float> dWg(hidden * inter), dWu(hidden * inter),
            dWd(inter * hidden);
        cuda::memory::Buffer<int> dt(m);
        dX.copy_from(X.data(), X.size());
        dt.copy_from(targets.data(), m);
        CrossEntropyConfig ce; ce.num_classes = hidden; ce.reduction_mean = true;
        for (int s = 1; s <= K; ++s) {
            ref_attn.forward(dX.data(), dA.data(), m, 1);
            ref_mlp.forward(dA.data(), dL.data(), m, 1);
            cross_entropy_logits_backward(dL.data(), dt.data(), dD.data(),
                                          m, hidden, ce);
            ref_mlp.backward(dA.data(), dD.data(), dDA.data(),
                             dWg.data(), dWu.data(), dWd.data(), m, 1);
            ref_attn.backward(dX.data(), dDA.data(), dA.data(),
                              dWq.data(), dWk.data(), dWv.data(), dWo.data(), m, 1);
            ref_mlp.step(og, ou, od, dWg.data(), dWu.data(), dWd.data(), s);
            ref_attn.step(oq, ok, ov, oo, dWq.data(), dWk.data(), dWv.data(),
                          dWo.data(), s);
        }
        ref_attn.copy_weights(rWq.data(), rWk.data(), rWv.data(), rWo.data());
        ref_mlp.copy_weights(rWg.data(), rWu.data(), rWd.data());
    }

    // Multi-GPU: per-rank attention+MLP, per-tensor AdamW, K identical steps.
    const int local_qkv = qkv / device_count;
    std::vector<std::vector<float>> Wq_s(device_count, std::vector<float>(hidden * local_qkv));
    std::vector<std::vector<float>> Wk_s(device_count, std::vector<float>(hidden * local_qkv));
    std::vector<std::vector<float>> Wv_s(device_count, std::vector<float>(hidden * local_qkv));
    std::vector<std::vector<float>> Wo_s(device_count, std::vector<float>(local_qkv * hidden));
    std::vector<std::vector<float>> Wg_s(device_count, std::vector<float>(hidden * (inter / device_count)));
    std::vector<std::vector<float>> Wu_s(device_count, std::vector<float>(hidden * (inter / device_count)));
    std::vector<std::vector<float>> Wd_s(device_count, std::vector<float>((inter / device_count) * hidden));
    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> dX(m * hidden), dA(m * hidden), dL(m * hidden),
            dD(m * hidden), dDA(m * hidden);
        cuda::memory::Buffer<float> dWq(hidden * qkv), dWk(hidden * qkv),
            dWv(hidden * qkv), dWo(qkv * hidden);
        cuda::memory::Buffer<float> dWg(hidden * inter), dWu(hidden * inter),
            dWd(inter * hidden);
        cuda::memory::Buffer<int> dt(m);
        dX.copy_from(X.data(), X.size());
        dt.copy_from(targets.data(), m);

        TensorParallelMultiHeadAttention attn(ctx, heads, hd, hidden);
        attn.set_weight(Wq.data(), Wk.data(), Wv.data(), Wo.data());
        TensorParallelMLP mlp(ctx, hidden, inter);
        mlp.set_weight(Wg.data(), Wu.data(), Wd.data());
        AdamWOptimizer oq(cfg), ok(cfg), ov(cfg), oo(cfg), og(cfg), ou(cfg), od(cfg);
        CrossEntropyConfig ce; ce.num_classes = hidden; ce.reduction_mean = true;
        for (int s = 1; s <= K; ++s) {
            attn.forward(dX.data(), dA.data(), m, 1);
            mlp.forward(dA.data(), dL.data(), m, 1);
            cross_entropy_logits_backward(dL.data(), dt.data(), dD.data(),
                                          m, hidden, ce);
            mlp.backward(dA.data(), dD.data(), dDA.data(),
                         dWg.data(), dWu.data(), dWd.data(), m, 1);
            attn.backward(dX.data(), dDA.data(), dA.data(),
                          dWq.data(), dWk.data(), dWv.data(), dWo.data(), m, 1);
            mlp.step(og, ou, od, dWg.data(), dWu.data(), dWd.data(), s);
            attn.step(oq, ok, ov, oo, dWq.data(), dWk.data(), dWv.data(),
                      dWo.data(), s);
        }
        // copy_weights reads back the rank's COMPACT shard: QKV as the
        // column shard [hidden x local_qkv], Wo as the row block
        // [local_qkv x hidden], gate/up as column shards [hidden x ilocal],
        // down as the row block [ilocal x hidden]. Store each directly.
        attn.copy_weights(Wq_s[d].data(), Wk_s[d].data(), Wv_s[d].data(),
                          Wo_s[d].data());
        mlp.copy_weights(Wg_s[d].data(), Wu_s[d].data(), Wd_s[d].data());
    });

    // Assemble the per-rank shards and compare against the single-GPU ref.
    std::vector<float> aWq(hidden * qkv), aWk(hidden * qkv), aWv(hidden * qkv),
        aWo(qkv * hidden), aWg(hidden * inter), aWu(hidden * inter),
        aWd(inter * hidden);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < hidden; ++a) {
            for (int c = 0; c < local_qkv; ++c) {
                aWq[a * qkv + r * local_qkv + c] = Wq_s[r][a * local_qkv + c];
                aWk[a * qkv + r * local_qkv + c] = Wk_s[r][a * local_qkv + c];
                aWv[a * qkv + r * local_qkv + c] = Wv_s[r][a * local_qkv + c];
            }
        }
        for (int c = 0; c < local_qkv * hidden; ++c) aWo[r * local_qkv * hidden + c] = Wo_s[r][c];
        const int ilocal = inter / device_count;
        for (int a = 0; a < hidden; ++a)
            for (int c = 0; c < ilocal; ++c) {
                aWg[a * inter + r * ilocal + c] = Wg_s[r][a * ilocal + c];
                aWu[a * inter + r * ilocal + c] = Wu_s[r][a * ilocal + c];
            }
        for (int c = 0; c < ilocal * hidden; ++c) aWd[r * ilocal * hidden + c] = Wd_s[r][c];
    }
    EXPECT_TRUE(arrays_near(rWq.data(), aWq.data(), hidden * qkv, 2e-2f));
    EXPECT_TRUE(arrays_near(rWk.data(), aWk.data(), hidden * qkv, 2e-2f));
    EXPECT_TRUE(arrays_near(rWv.data(), aWv.data(), hidden * qkv, 2e-2f));
    EXPECT_TRUE(arrays_near(rWo.data(), aWo.data(), qkv * hidden, 2e-2f));
    EXPECT_TRUE(arrays_near(rWg.data(), aWg.data(), hidden * inter, 2e-2f));
    EXPECT_TRUE(arrays_near(rWu.data(), aWu.data(), hidden * inter, 2e-2f));
    EXPECT_TRUE(arrays_near(rWd.data(), aWd.data(), inter * hidden, 2e-2f))
        << "assembled mini-transformer shards differ from the single-GPU ref";
}
