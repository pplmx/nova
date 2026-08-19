/**
 * @file attention_kernels_test.cpp
 * @brief Single-GPU kernel-level contracts for the v2.29 device SDPA (TASK-044)
 *
 * Milestone v2.29 (DEC-015): the public sdpa_forward/sdpa_backward in
 * tensor_parallel_attention.cpp compute multi-head scaled-dot-product
 * attention ON THE HOST (D2H -> host loops -> H2D), once per block per
 * train_step — the binding deep-transformer training-latency constraint
 * (Round-28 LEARN M3; the v2.26 header deferred the device move). This file
 * pins the P1 RED contract surface: detail::sdpa_forward_device /
 * detail::sdpa_backward_device (attention_kernels.cu) must equal an
 * independent host-fp64 SDPA reference (forward out; backward dq/dk/dv) at
 * seq>1, heads>1, head_dim>1.
 *
 * P1-RED: the detail:: entries are provisional throw-stubs, so these tests
 * fail (throw) here and turn GREEN with the P2 device kernels (TASK-045).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/tensor_parallel_attention.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace {

// Per-element near check (abs, or rel when the reference is large) — the
// established neural-parity convention.
bool arrays_near(const float* expected, const float* actual, size_t n,
                 float tol = 1e-4f) {
    for (size_t i = 0; i < n; ++i) {
        const float ref = expected[i];
        const float diff = std::fabs(actual[i] - ref);
        if (diff > tol && diff > tol * std::fabs(ref)) {
            return false;
        }
    }
    return true;
}

void fill_random(float* data, size_t count, unsigned seed, float lo = -1.0f,
                 float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

// Host-fp64 SDPA forward: out[b,h,:] = softmax_j( q[b,h,.] . k[j,h,.] / sqrt(d) )
// v[j,h,:], over m positions/keys (self-attention). Buffer layout matches the
// public API: q/k/v/out are [m x heads x head_dim], position-major then head.
void ref_sdpa_forward_fp64(const float* q, const float* k, const float* v,
                           float* out, int m, int heads, int hd) {
    const double inv = 1.0 / std::sqrt(static_cast<double>(hd));
    const size_t head_stride = static_cast<size_t>(hd);
    const size_t pos_stride = static_cast<size_t>(heads) * hd;
    std::vector<double> score(m), prob(m);
    for (int b = 0; b < m; ++b) {
        const size_t qoff = static_cast<size_t>(b) * pos_stride;
        for (int h = 0; h < heads; ++h) {
            const size_t hoff = static_cast<size_t>(h) * head_stride;
            double max_s = -1e300;
            for (int j = 0; j < m; ++j) {
                double s = 0.0;
                const size_t koff = static_cast<size_t>(j) * pos_stride + hoff;
                for (int d = 0; d < hd; ++d) {
                    s += static_cast<double>(q[qoff + hoff + d]) *
                         static_cast<double>(k[koff + d]);
                }
                score[static_cast<size_t>(j)] = s * inv;
                max_s = std::max(max_s, score[static_cast<size_t>(j)]);
            }
            double sum = 0.0;
            for (int j = 0; j < m; ++j) {
                prob[static_cast<size_t>(j)] =
                    std::exp(score[static_cast<size_t>(j)] - max_s);
                sum += prob[static_cast<size_t>(j)];
            }
            for (int j = 0; j < m; ++j) {
                prob[static_cast<size_t>(j)] /= sum;
            }
            const size_t oi = qoff + hoff;
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < m; ++j) {
                    const size_t voff = static_cast<size_t>(j) * pos_stride + hoff;
                    acc += prob[static_cast<size_t>(j)] *
                           static_cast<double>(v[voff + d]);
                }
                out[oi + d] = static_cast<float>(acc);
            }
        }
    }
}

// Host-fp64 analytic SDPA backward (mirrors the v2.26 CPU chain, in double):
// dQ[b,:] = sum_j dscore[b,j] k[j,:]; dK[j,:] = dscore[b,j] q[b,:];
// dV[j,:] = P[b,j] dout[b,:], with dscore[b,j] = P[b,j]*(dP[b,j] -
// sum_k P[b,k] dP[b,k])/sqrt(d), dP[b,j] = dout[b,:] . v[j,:].
void ref_sdpa_backward_fp64(const float* q, const float* k, const float* v,
                            const float* dout, float* dq, float* dk, float* dv,
                            int m, int heads, int hd) {
    const double inv = 1.0 / std::sqrt(static_cast<double>(hd));
    const size_t head_stride = static_cast<size_t>(hd);
    const size_t pos_stride = static_cast<size_t>(heads) * hd;
    std::vector<double> score(m), prob(m), dp(m), ds(m);
    for (int b = 0; b < m; ++b) {
        const size_t qoff = static_cast<size_t>(b) * pos_stride;
        for (int h = 0; h < heads; ++h) {
            const size_t hoff = static_cast<size_t>(h) * head_stride;
            double max_s = -1e300;
            for (int j = 0; j < m; ++j) {
                double s = 0.0;
                const size_t koff = static_cast<size_t>(j) * pos_stride + hoff;
                for (int d = 0; d < hd; ++d) {
                    s += static_cast<double>(q[qoff + hoff + d]) *
                         static_cast<double>(k[koff + d]);
                }
                score[static_cast<size_t>(j)] = s * inv;
                max_s = std::max(max_s, score[static_cast<size_t>(j)]);
            }
            double sum = 0.0;
            for (int j = 0; j < m; ++j) {
                prob[static_cast<size_t>(j)] =
                    std::exp(score[static_cast<size_t>(j)] - max_s);
                sum += prob[static_cast<size_t>(j)];
            }
            for (int j = 0; j < m; ++j) {
                prob[static_cast<size_t>(j)] /= sum;
            }

            // dV[j,:] += P[j] * dout[b,:].
            for (int j = 0; j < m; ++j) {
                const double pj = prob[static_cast<size_t>(j)];
                const size_t dvoff = static_cast<size_t>(j) * pos_stride + hoff;
                for (int d = 0; d < hd; ++d) {
                    dv[dvoff + d] += static_cast<float>(
                        pj * static_cast<double>(dout[qoff + hoff + d]));
                }
            }
            // dP[j] = dout[b,:] . v[j,:]; weighted = sum_k P[k] dP[k].
            double weighted = 0.0;
            for (int j = 0; j < m; ++j) {
                double acc = 0.0;
                const size_t voff = static_cast<size_t>(j) * pos_stride + hoff;
                for (int d = 0; d < hd; ++d) {
                    acc += static_cast<double>(dout[qoff + hoff + d]) *
                           static_cast<double>(v[voff + d]);
                }
                dp[static_cast<size_t>(j)] = acc;
                weighted += prob[static_cast<size_t>(j)] * acc;
            }
            for (int j = 0; j < m; ++j) {
                ds[static_cast<size_t>(j)] =
                    prob[static_cast<size_t>(j)] *
                    (dp[static_cast<size_t>(j)] - weighted) * inv;
            }
            // dQ[b,:] = sum_j ds[j] k[j,:].
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < m; ++j) {
                    const size_t koff = static_cast<size_t>(j) * pos_stride + hoff;
                    acc += ds[static_cast<size_t>(j)] *
                           static_cast<double>(k[koff + d]);
                }
                dq[qoff + hoff + d] += static_cast<float>(acc);
            }
            // dK[j,:] += ds[j] q[b,:].
            for (int j = 0; j < m; ++j) {
                const double dsj = ds[static_cast<size_t>(j)];
                const size_t dkoff = static_cast<size_t>(j) * pos_stride + hoff;
                for (int d = 0; d < hd; ++d) {
                    dk[dkoff + d] += static_cast<float>(
                        dsj * static_cast<double>(q[qoff + hoff + d]));
                }
            }
        }
    }
}

}  // namespace

namespace cuda::neural {

TEST(AttentionKernelsTest, ForwardDeviceMatchesFp64Reference) {
    const int batch = 2, seq = 3, heads = 3, hd = 8;  // m = 6 rows, seq>1
    const int m = batch * seq;
    const size_t n = static_cast<size_t>(m) * heads * hd;
    std::vector<float> q(n), k(n), v(n), ref(n), out(n);
    fill_random(q.data(), n, 11);
    fill_random(k.data(), n, 22);
    fill_random(v.data(), n, 33);
    ref_sdpa_forward_fp64(q.data(), k.data(), v.data(), ref.data(), m, heads,
                          hd);

    memory::Buffer<float> d_q(n), d_k(n), d_v(n), d_out(n);
    d_q.copy_from(q.data(), n);
    d_k.copy_from(k.data(), n);
    d_v.copy_from(v.data(), n);
    detail::sdpa_forward_device(d_q.data(), d_k.data(), d_v.data(),
                                d_out.data(), m, heads, hd, nullptr);
    d_out.copy_to(out.data(), n);

    EXPECT_TRUE(arrays_near(ref.data(), out.data(), n, 1e-4f)) << "SDPA out";
}

TEST(AttentionKernelsTest, BackwardDeviceMatchesFp64Reference) {
    const int batch = 2, seq = 3, heads = 3, hd = 8;
    const int m = batch * seq;
    const size_t n = static_cast<size_t>(m) * heads * hd;
    std::vector<float> q(n), k(n), v(n), dout(n);
    std::vector<float> ref_dq(n, 0.0f), ref_dk(n, 0.0f), ref_dv(n, 0.0f);
    std::vector<float> dq(n), dk(n), dv(n);
    fill_random(q.data(), n, 101);
    fill_random(k.data(), n, 202);
    fill_random(v.data(), n, 303);
    fill_random(dout.data(), n, 404);
    ref_sdpa_backward_fp64(q.data(), k.data(), v.data(), dout.data(),
                           ref_dq.data(), ref_dk.data(), ref_dv.data(), m,
                           heads, hd);

    memory::Buffer<float> d_q(n), d_k(n), d_v(n), d_dout(n);
    memory::Buffer<float> d_dq(n), d_dk(n), d_dv(n);
    d_q.copy_from(q.data(), n);
    d_k.copy_from(k.data(), n);
    d_v.copy_from(v.data(), n);
    d_dout.copy_from(dout.data(), n);
    detail::sdpa_backward_device(d_q.data(), d_k.data(), d_v.data(),
                                 d_dout.data(), d_dq.data(), d_dk.data(),
                                 d_dv.data(), m, heads, hd, nullptr);
    d_dq.copy_to(dq.data(), n);
    d_dk.copy_to(dk.data(), n);
    d_dv.copy_to(dv.data(), n);

    EXPECT_TRUE(arrays_near(ref_dq.data(), dq.data(), n, 1e-4f)) << "SDPA dq";
    EXPECT_TRUE(arrays_near(ref_dk.data(), dk.data(), n, 1e-4f)) << "SDPA dk";
    EXPECT_TRUE(arrays_near(ref_dv.data(), dv.data(), n, 1e-4f)) << "SDPA dv";
}

TEST(AttentionKernelsTest, ForwardDeviceMatchesFp64ReferenceLargeSeq) {
    // m = 128 keys (blockDim would not cover the whole key range in one pass)
    // with head_dim=16 — exercises the chunked-key/block-reduction path.
    const int batch = 16, seq = 8, heads = 2, hd = 16;
    const int m = batch * seq;
    const size_t n = static_cast<size_t>(m) * heads * hd;
    std::vector<float> q(n), k(n), v(n), ref(n), out(n);
    fill_random(q.data(), n, 501);
    fill_random(k.data(), n, 502);
    fill_random(v.data(), n, 503);
    ref_sdpa_forward_fp64(q.data(), k.data(), v.data(), ref.data(), m, heads,
                          hd);

    memory::Buffer<float> d_q(n), d_k(n), d_v(n), d_out(n);
    d_q.copy_from(q.data(), n);
    d_k.copy_from(k.data(), n);
    d_v.copy_from(v.data(), n);
    detail::sdpa_forward_device(d_q.data(), d_k.data(), d_v.data(),
                                d_out.data(), m, heads, hd, nullptr);
    d_out.copy_to(out.data(), n);

    EXPECT_TRUE(arrays_near(ref.data(), out.data(), n, 2e-4f)) << "SDPA out";
}

TEST(AttentionKernelsTest, BackwardDeviceMatchesFp64ReferenceLargeSeq) {
    const int batch = 16, seq = 8, heads = 2, hd = 16;
    const int m = batch * seq;
    const size_t n = static_cast<size_t>(m) * heads * hd;
    std::vector<float> q(n), k(n), v(n), dout(n);
    std::vector<float> ref_dq(n, 0.0f), ref_dk(n, 0.0f), ref_dv(n, 0.0f);
    std::vector<float> dq(n), dk(n), dv(n);
    fill_random(q.data(), n, 601);
    fill_random(k.data(), n, 602);
    fill_random(v.data(), n, 603);
    fill_random(dout.data(), n, 604);
    ref_sdpa_backward_fp64(q.data(), k.data(), v.data(), dout.data(),
                           ref_dq.data(), ref_dk.data(), ref_dv.data(), m,
                           heads, hd);

    memory::Buffer<float> d_q(n), d_k(n), d_v(n), d_dout(n);
    memory::Buffer<float> d_dq(n), d_dk(n), d_dv(n);
    d_q.copy_from(q.data(), n);
    d_k.copy_from(k.data(), n);
    d_v.copy_from(v.data(), n);
    d_dout.copy_from(dout.data(), n);
    detail::sdpa_backward_device(d_q.data(), d_k.data(), d_v.data(),
                                 d_dout.data(), d_dq.data(), d_dk.data(),
                                 d_dv.data(), m, heads, hd, nullptr);
    d_dq.copy_to(dq.data(), n);
    d_dk.copy_to(dk.data(), n);
    d_dv.copy_to(dv.data(), n);

    EXPECT_TRUE(arrays_near(ref_dq.data(), dq.data(), n, 2e-4f)) << "SDPA dq";
    EXPECT_TRUE(arrays_near(ref_dk.data(), dk.data(), n, 2e-4f)) << "SDPA dk";
    EXPECT_TRUE(arrays_near(ref_dv.data(), dv.data(), n, 2e-4f)) << "SDPA dv";
}

}  // namespace cuda::neural
