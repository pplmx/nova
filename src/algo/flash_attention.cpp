#include "cuda/algo/flash_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_bf16.h>

namespace cuda::algo {

std::unique_ptr<FlashAttention> create_flash_attention(
    const FlashAttentionConfig& config
) {
    return std::make_unique<FlashAttention>(config);
}

FlashAttention::FlashAttention(const FlashAttentionConfig& config)
    : config_(config) {}

FlashAttention::~FlashAttention() = default;

size_t FlashAttention::get_workspace_size() const {
    // Per-(batch, head) SxS scores matrix in bytes; grows with batch/heads/seq.
    return static_cast<size_t>(std::max(0, config_.batch_size)) *
           static_cast<size_t>(std::max(0, config_.num_heads)) *
           static_cast<size_t>(std::max(0, config_.seq_len)) *
           static_cast<size_t>(std::max(0, config_.seq_len)) * sizeof(float);
}

void FlashAttention::ensure_workspace(size_t size) {
    // The reference implementation computes on the host; nothing is
    // preallocated on the device. Reported size is unchanged (API contract).
    (void)size;
}

void FlashAttention::set_dropout(float rate, uint64_t seed) {
    config_.dropout_rate = rate;
    config_.dropout_seed = seed;
}

namespace {

// xorshift64 PRNG: deterministic so forward() calls with an identical config
// (same dropout seed) produce byte-identical results across calls.
inline uint64_t xorshift64(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

struct AttentionGeometry {
    int batch;
    int heads;
    int kv_heads;
    int seq;
    int dim;
    float scale;
    float drop;
    bool causal;
    uint64_t seed;
};

// Computes scores/probs for one (batch, head) into the provided S*S buffers.
// `scores` holds raw logits (masked entries = -inf), `probs` the (possibly
// dropout-masked) attention weights. Returns the last-row logsumexp.
float attention_row_probs(const AttentionGeometry& g,
                          const float* q,
                          const float* k,
                          const float* v,
                          float* scores,
                          float* probs) {
    const int S = g.seq;
    const int D = g.dim;
    const int H = g.heads;
    const int KH = g.kv_heads;

    float lse = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < S; ++i) {
        float row_max = -std::numeric_limits<float>::infinity();

        for (int j = 0; j < S; ++j) {
            const size_t idx = static_cast<size_t>(i) * S + j;
            if (g.causal && j > i) {
                scores[idx] = -std::numeric_limits<float>::infinity();
                probs[idx] = 0.0f;
                continue;
            }
            float dot = 0.0f;
            // Layout [S][H][D] for q; [S][KH][D] for k/v (kv_heads shared via
            // per-head offset chosen by the caller).
            const float* qi = q + (static_cast<size_t>(i) * H + 0) * D;
            const float* kj = k + static_cast<size_t>(j) * KH * D;
            for (int d = 0; d < D; ++d) {
                dot += qi[d] * kj[d];
            }
            const float s = dot * g.scale;
            scores[idx] = s;
            probs[idx] = 0.0f;
            if (s > row_max) {
                row_max = s;
            }
        }

        double sum_exp = 0.0;
        for (int j = 0; j < S; ++j) {
            const size_t idx = static_cast<size_t>(i) * S + j;
            if (scores[idx] == -std::numeric_limits<float>::infinity()) {
                continue;
            }
            const float e = std::exp(scores[idx] - row_max);
            probs[idx] = e;
            sum_exp += e;
        }

        for (int j = 0; j < S; ++j) {
            const size_t idx = static_cast<size_t>(i) * S + j;
            if (scores[idx] == -std::numeric_limits<float>::infinity()) {
                continue;
            }
            probs[idx] = static_cast<float>(probs[idx] / sum_exp);
        }

        if (i == S - 1) {
            lse = static_cast<float>(row_max + std::log(sum_exp));
        }
    }

    // Deterministic dropout, keyed per call from the config seed.
    if (g.drop > 0.0f) {
        uint64_t rng = g.seed != 0 ? g.seed : 1u;
        const float keep = 1.0f / (1.0f - g.drop);
        const size_t nprobs = static_cast<size_t>(S) * S;
        for (size_t p = 0; p < nprobs; ++p) {
            if ((xorshift64(rng) % 10000u) / 10000.0 < g.drop) {
                probs[p] = 0.0f;
            } else if (probs[p] > 0.0f) {
                probs[p] *= keep;
            }
        }
    }

    return lse;
}

}  // namespace

void FlashAttention::forward(
    memory::Buffer<float>& output,
    memory::Buffer<float>& softmax_lse,
    const memory::Buffer<float>& query,
    const memory::Buffer<float>& key,
    const memory::Buffer<float>& value,
    const stream::Stream& stream
) {
    (void)stream;

    const int B = config_.batch_size;
    const int H = config_.num_heads;
    const int S = config_.seq_len;
    const int D = config_.head_dim;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) {
        return;
    }

    const int KH = std::max(1, config_.num_kv_heads);
    const size_t qe = static_cast<size_t>(B) * S * H * D;
    const size_t kve = static_cast<size_t>(B) * S * KH * D;
    if (query.size() < qe || key.size() < kve || value.size() < kve ||
        output.size() < qe || softmax_lse.size() < static_cast<size_t>(B) * H) {
        return;
    }

    AttentionGeometry g{
        .batch = B, .heads = H, .kv_heads = KH, .seq = S, .dim = D,
        .scale = 1.0f / std::sqrt(static_cast<float>(D)),
        .drop = config_.dropout_rate, .causal = config_.causal,
        .seed = config_.dropout_seed
    };

    std::vector<float> h_q(qe), h_k(kve), h_v(kve), h_o(qe), h_lse(static_cast<size_t>(B) * H);
    query.copy_to(h_q.data(), qe);
    key.copy_to(h_k.data(), kve);
    value.copy_to(h_v.data(), kve);

    std::vector<float> scores(static_cast<size_t>(S) * S);
    std::vector<float> probs(static_cast<size_t>(S) * S);

    for (int b = 0; b < B; ++b) {
        const float* qb = h_q.data() + static_cast<size_t>(b) * S * H * D;
        const float* kb = h_k.data() + static_cast<size_t>(b) * S * KH * D;
        const float* vb = h_v.data() + static_cast<size_t>(b) * S * KH * D;

        for (int h = 0; h < H; ++h) {
            const int kv_h = static_cast<int>((static_cast<int64_t>(h) * KH) / H);
            const float* q = qb + h * D;
            const float* k = kb + kv_h * D;
            const float* v = vb + kv_h * D;
            float* o = h_o.data() + (static_cast<size_t>(b) * S * H + h) * D;

            const float lse = attention_row_probs(g, q, k, v, scores.data(), probs.data());
            h_lse[static_cast<size_t>(b) * H + h] = lse;

            // O = P V
            for (int i = 0; i < S; ++i) {
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < S; ++j) {
                        acc += probs[static_cast<size_t>(i) * S + j] * v[static_cast<size_t>(j) * KH * D + d];
                    }
                    o[static_cast<size_t>(i) * H * D + d] = acc;
                }
            }
        }
    }

    output.copy_from(h_o.data(), qe);
    softmax_lse.copy_from(h_lse.data(), h_lse.size());
}

void FlashAttention::forward_bf16(
    memory::Buffer<void>& output,
    memory::Buffer<float>& softmax_lse,
    const memory::Buffer<void>& query,
    const memory::Buffer<void>& key,
    const memory::Buffer<void>& value,
    const stream::Stream& stream
) {
    // Upcast bf16 inputs, run the float path, downcast the result. If the
    // buffers are smaller than the configured geometry, there is nothing
    // meaningful to compute; return (no-throw) rather than reading OOB.
    const int B = config_.batch_size;
    const int H = config_.num_heads;
    const int S = config_.seq_len;
    const int D = config_.head_dim;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) {
        return;
    }
    const int KH = std::max(1, config_.num_kv_heads);
    const size_t qe = static_cast<size_t>(B) * S * H * D;
    const size_t kve = static_cast<size_t>(B) * S * KH * D;
    if (query.size() < qe * sizeof(__nv_bfloat16) ||
        key.size() < kve * sizeof(__nv_bfloat16) ||
        value.size() < kve * sizeof(__nv_bfloat16) ||
        output.size() < qe * sizeof(__nv_bfloat16)) {
        return;
    }

    std::vector<__nv_bfloat16> h_qb(qe), h_kb(kve), h_vb(kve), h_ob(qe);
    query.copy_to(h_qb.data(), qe);
    key.copy_to(h_kb.data(), kve);
    value.copy_to(h_vb.data(), kve);

    std::vector<float> h_q(qe), h_k(kve), h_v(kve), h_o(qe);
    for (size_t i = 0; i < qe; ++i) {
        h_q[i] = __bfloat162float(h_qb[i]);
    }
    for (size_t i = 0; i < kve; ++i) {
        h_k[i] = __bfloat162float(h_kb[i]);
        h_v[i] = __bfloat162float(h_vb[i]);
    }

    memory::Buffer<float> q_f(qe), k_f(kve), v_f(kve), o_f(qe);
    memory::Buffer<float> lse_f(static_cast<size_t>(B) * H);
    q_f.copy_from(h_q.data(), qe);
    k_f.copy_from(h_k.data(), kve);
    v_f.copy_from(h_v.data(), kve);
    forward(o_f, lse_f, q_f, k_f, v_f, stream);

    o_f.copy_to(h_o.data(), qe);
    for (size_t i = 0; i < qe; ++i) {
        h_ob[i] = __float2bfloat16(h_o[i]);
    }
    output.copy_from(h_ob.data(), qe);
    lse_f.copy_to(softmax_lse.data(), std::min(softmax_lse.size(), lse_f.size()));
}

void FlashAttention::backward(
    memory::Buffer<float>& dq,
    memory::Buffer<float>& dk,
    memory::Buffer<float>& dv,
    const memory::Buffer<float>& output,
    const memory::Buffer<float>& dout,
    const memory::Buffer<float>& query,
    const memory::Buffer<float>& key,
    const memory::Buffer<float>& value,
    const memory::Buffer<float>& softmax_lse,
    const stream::Stream& stream
) {
    (void)output;
    (void)softmax_lse;
    (void)stream;

    const int B = config_.batch_size;
    const int H = config_.num_heads;
    const int S = config_.seq_len;
    const int D = config_.head_dim;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) {
        return;
    }
    const int KH = std::max(1, config_.num_kv_heads);
    const size_t qe = static_cast<size_t>(B) * S * H * D;
    const size_t kve = static_cast<size_t>(B) * S * KH * D;
    if (query.size() < qe || key.size() < kve || value.size() < kve ||
        dout.size() < qe || dq.size() < qe || dk.size() < qe || dv.size() < qe) {
        return;
    }

    AttentionGeometry g{
        .batch = B, .heads = H, .kv_heads = KH, .seq = S, .dim = D,
        .scale = 1.0f / std::sqrt(static_cast<float>(D)),
        .drop = config_.dropout_rate, .causal = config_.causal,
        .seed = config_.dropout_seed
    };

    std::vector<float> h_q(qe), h_k(kve), h_v(kve), h_do(qe);
    std::vector<float> h_dq(qe, 0.0f), h_dk(qe, 0.0f), h_dv(qe, 0.0f);
    query.copy_to(h_q.data(), qe);
    key.copy_to(h_k.data(), kve);
    value.copy_to(h_v.data(), kve);
    dout.copy_to(h_do.data(), qe);

    std::vector<float> scores(static_cast<size_t>(S) * S);
    std::vector<float> probs(static_cast<size_t>(S) * S);
    std::vector<float> dP(static_cast<size_t>(S) * S, 0.0f);

    for (int b = 0; b < B; ++b) {
        const float* qb = h_q.data() + static_cast<size_t>(b) * S * H * D;
        const float* kb = h_k.data() + static_cast<size_t>(b) * S * KH * D;
        const float* vb = h_v.data() + static_cast<size_t>(b) * S * KH * D;
        const float* dob = h_do.data() + static_cast<size_t>(b) * S * H * D;

        for (int h = 0; h < H; ++h) {
            const int kv_h = static_cast<int>((static_cast<int64_t>(h) * KH) / H);
            const float* q = qb + h * D;
            const float* k = kb + kv_h * D;
            const float* v = vb + kv_h * D;
            const float* doh = dob + static_cast<size_t>(h) * D;
            float* dqh = h_dq.data() + (static_cast<size_t>(b) * S * H + h) * D;
            float* dkh = h_dk.data() + (static_cast<size_t>(b) * S * H + h) * D;
            float* dvh = h_dv.data() + (static_cast<size_t>(b) * S * H + h) * D;

            attention_row_probs(g, q, k, v, scores.data(), probs.data());

            // dP[i,j] = sum_d dout[i,d] * v[j,d]
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < S; ++j) {
                    if (probs[static_cast<size_t>(i) * S + j] == 0.0f) {
                        continue;
                    }
                    float acc = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        acc += doh[static_cast<size_t>(i) * H * D + d] *
                               v[static_cast<size_t>(j) * KH * D + d];
                    }
                    dP[static_cast<size_t>(i) * S + j] = acc;
                }
            }

            // dA[i,j] = P[i,j] * (dP[i,j] - sum_k P[i,k]*dP[i,k])
            std::vector<float> dA(static_cast<size_t>(S) * S, 0.0f);
            for (int i = 0; i < S; ++i) {
                float dot_P_dP = 0.0f;
                for (int j = 0; j < S; ++j) {
                    dot_P_dP += probs[static_cast<size_t>(i) * S + j] *
                                dP[static_cast<size_t>(i) * S + j];
                }
                for (int j = 0; j < S; ++j) {
                    const size_t ij = static_cast<size_t>(i) * S + j;
                    if (probs[ij] == 0.0f) {
                        continue;
                    }
                    dA[ij] = probs[ij] * (dP[ij] - dot_P_dP);
                }
            }

            // dV[j,d] = sum_i P[i,j]*dout[i,d]
            for (int j = 0; j < S; ++j) {
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int i = 0; i < S; ++i) {
                        acc += probs[static_cast<size_t>(i) * S + j] *
                               doh[static_cast<size_t>(i) * H * D + d];
                    }
                    dvh[static_cast<size_t>(j) * H * D + d] = acc;
                }
            }

            // dQ and dK from dA (scores = q.k^T * scale)
            for (int i = 0; i < S; ++i) {
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < S; ++j) {
                        acc += dA[static_cast<size_t>(i) * S + j] *
                               k[static_cast<size_t>(j) * KH * D + d];
                    }
                    dqh[static_cast<size_t>(i) * H * D + d] = acc * g.scale;
                }
            }
            for (int j = 0; j < S; ++j) {
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int i = 0; i < S; ++i) {
                        acc += dA[static_cast<size_t>(i) * S + j] *
                               q[static_cast<size_t>(i) * H * D + d];
                    }
                    dkh[static_cast<size_t>(j) * H * D + d] += acc * g.scale;
                }
            }
        }
    }

    dq.copy_from(h_dq.data(), qe);
    dk.copy_from(h_dk.data(), qe);
    dv.copy_from(h_dv.data(), qe);
}

}  // namespace cuda::algo
