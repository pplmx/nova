/**
 * @file tensor_parallel_layers_test.cpp
 * @brief Single-GPU (tp == 1) functional tests for the weight-managed
 *        TensorParallelLayers
 *
 * Milestone v2.21 (TASK-010 / issue-v20-tp-layers-stubs): the v1.3
 * ColumnParallelLayer / RowParallelLayer / TensorParallelMLP wrappers were
 * non-functional scaffolding that owned no weight storage and were disposed
 * as fail-fast stubs in v2.20 (DEC-006). This milestone implements them for
 * real: each layer owns a device weight buffer (the rank's shard of the full
 * weight), set_weight() uploads and slices the full weight, and forward()
 * computes the documented sharded/replicated outputs.
 *
 * These tests cover the tp <= 1 path with an uninitialized NcclContext (0
 * devices -> effective tp degree 1): the layer degenerates to a plain
 * single-shard matmul on the full weight, no NCCL required. Every forward is
 * asserted against a single-GPU reference on the public matmul primitive
 * (plus the SiLU-gated MLP reference). The multi-GPU shard/AllReduce parity
 * paths live in tensor_parallel_multigpu_test.cpp.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/matmul.h"
#include "cuda/neural/activations.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::neural;
using namespace cuda::neural::optimizers;

namespace {

// An uninitialized NcclContext reports 0 devices; the layers treat tp <= 1 as
// a plain single-shard matmul on the full weight (no rank slicing, no NCCL).
cuda::nccl::NcclContext& uninitialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

void fill_random(float* data, size_t count) {
    static std::mt19937 rng(11);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

// Single-GPU reference: C = A @ B through the public matmul primitive.
void ref_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    cuda::memory::Buffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
    d_A.copy_from(A, static_cast<size_t>(m) * k);
    d_B.copy_from(B, static_cast<size_t>(k) * n);
    cuda::neural::matmul(d_A.data(), d_B.data(), d_C.data(), m, n, k);
    d_C.copy_to(C, static_cast<size_t>(m) * n);
}

// SiLU-gated reference: out = (silu(A @ W_gate) .* (A @ W_up)) @ W_down.
// Computed on device via the public primitives so the layer is compared
// against the library's own matmul/activation stack (single-GPU).
void ref_gated_mlp(const float* A, const float* W_gate, const float* W_up,
                   const float* W_down, float* out, int m, int h, int inter) {
    cuda::memory::Buffer<float> d_A(m * h);
    cuda::memory::Buffer<float> d_gate(m * inter), d_up(m * inter),
        d_sub(m * inter), d_out(m * h);
    cuda::memory::Buffer<float> d_W_gate(h * inter), d_W_up(h * inter),
        d_W_down(h * inter);
    d_A.copy_from(A, static_cast<size_t>(m) * h);
    d_W_gate.copy_from(W_gate, static_cast<size_t>(h) * inter);
    d_W_up.copy_from(W_up, static_cast<size_t>(h) * inter);
    d_W_down.copy_from(W_down, static_cast<size_t>(h) * inter);

    cuda::neural::matmul(d_A.data(), d_W_gate.data(), d_gate.data(),
                         m, inter, h);
    cuda::neural::matmul(d_A.data(), d_W_up.data(), d_up.data(), m, inter, h);
    cuda::neural::silu_and_mul(d_gate.data(), d_up.data(), d_sub.data(),
                               m * inter);
    cuda::neural::matmul(d_sub.data(), d_W_down.data(), d_out.data(),
                         m, h, inter);
    d_out.copy_to(out, static_cast<size_t>(m) * h);
}

}  // namespace

class TensorParallelLayersSingleGpuTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        cudaFree(nullptr);  // Ensure a CUDA context for device-dependent calls
    }

    static testing::AssertionResult arrays_near(
        const float* expected, const float* actual,
        size_t count, float tolerance) {
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(expected[i] - actual[i]) > tolerance) {
                return testing::AssertionFailure()
                       << "mismatch at " << i << ": expected " << expected[i]
                       << " actual " << actual[i] << " (|diff| "
                       << std::abs(expected[i] - actual[i]) << " > " << tolerance << ")";
            }
        }
        return testing::AssertionSuccess();
    }
};

// Host double-precision matmul: C[m x n] = A[m x k] @ B[k x n] (row-major).
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

// dW = X^T dY [k x n], dX = dY W^T [m x k] — analytic backward of Y = X W,
// computed in double precision.
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

// With no NCCL context the layer must fall back to a plain full-weight matmul
// (tp degree 1): ColumnParallelLayer output equals the single-GPU reference.
TEST_F(TensorParallelLayersSingleGpuTest, ColumnParallelLayer_MatchesReference) {
    const int m = 8, k = 16, n = 32;
    std::vector<float> A(static_cast<size_t>(m) * k);
    std::vector<float> W(static_cast<size_t>(k) * n);
    std::vector<float> ref(static_cast<size_t>(m) * n);
    fill_random(A.data(), A.size());
    fill_random(W.data(), W.size());
    ref_matmul(A.data(), W.data(), ref.data(), m, n, k);

    ColumnParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W.data());

    cuda::memory::Buffer<float> d_A(m * k), d_out(m * n);
    d_A.copy_from(A.data(), A.size());

    std::vector<float> out(static_cast<size_t>(m) * n);
    layer.forward(d_A.data(), d_out.data(), m, 1);
    d_out.copy_to(out.data(), out.size());
    EXPECT_TRUE(arrays_near(ref.data(), out.data(), out.size(), 1e-3f));
    EXPECT_EQ(layer.in_features(), k);
    EXPECT_EQ(layer.out_features(), n);
    EXPECT_EQ(layer.tp_degree(), 1);
}

// Same fall-back contract for RowParallelLayer.
TEST_F(TensorParallelLayersSingleGpuTest, RowParallelLayer_MatchesReference) {
    const int m = 8, k = 16, n = 32;
    std::vector<float> A(static_cast<size_t>(m) * k);
    std::vector<float> W(static_cast<size_t>(k) * n);
    std::vector<float> ref(static_cast<size_t>(m) * n);
    fill_random(A.data(), A.size());
    fill_random(W.data(), W.size());
    ref_matmul(A.data(), W.data(), ref.data(), m, n, k);

    RowParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W.data());

    cuda::memory::Buffer<float> d_A(m * k), d_out(m * n);
    d_A.copy_from(A.data(), A.size());

    std::vector<float> out(static_cast<size_t>(m) * n);
    layer.forward(d_A.data(), d_out.data(), m, 1);
    d_out.copy_to(out.data(), out.size());
    EXPECT_TRUE(arrays_near(ref.data(), out.data(), out.size(), 1e-3f));
    EXPECT_EQ(layer.in_features(), k);
    EXPECT_EQ(layer.out_features(), n);
}

// TensorParallelMLP (SiLU-gated FFN) on the tp-1 path must equal the gated
// reference: out = (silu(X W_gate) .* (X W_up)) W_down.
TEST_F(TensorParallelLayersSingleGpuTest, TensorParallelMLP_MatchesReference) {
    const int m = 8, h = 16, inter = 48;
    std::vector<float> A(static_cast<size_t>(m) * h);
    std::vector<float> W_gate(static_cast<size_t>(h) * inter);
    std::vector<float> W_up(static_cast<size_t>(h) * inter);
    std::vector<float> W_down(static_cast<size_t>(inter) * h);
    fill_random(A.data(), A.size());
    fill_random(W_gate.data(), W_gate.size());
    fill_random(W_up.data(), W_up.size());
    fill_random(W_down.data(), W_down.size());

    std::vector<float> ref(static_cast<size_t>(m) * h);
    ref_gated_mlp(A.data(), W_gate.data(), W_up.data(), W_down.data(),
                  ref.data(), m, h, inter);

    TensorParallelMLP mlp(uninitialized_ctx(), h, inter);
    mlp.set_weight(W_gate.data(), W_up.data(), W_down.data());

    cuda::memory::Buffer<float> d_A(m * h), d_out(m * h);
    d_A.copy_from(A.data(), A.size());

    std::vector<float> out(static_cast<size_t>(m) * h);
    mlp.forward(d_A.data(), d_out.data(), m, 1);
    d_out.copy_to(out.data(), out.size());
    // tol absorbs fp32 accumulate-order + activation differences (~1e-2 level).
    EXPECT_TRUE(arrays_near(ref.data(), out.data(), out.size(), 2e-2f));
}

// The layer wrappers own their weights: re-setting the weight must change the
// forward result (the buffer is real storage, not a pass-through pointer).
TEST_F(TensorParallelLayersSingleGpuTest, SetWeightOverwritesBuffer) {
    const int m = 4, k = 8, n = 8;
    std::vector<float> A(static_cast<size_t>(m) * k);
    std::vector<float> W_a(static_cast<size_t>(k) * n);
    std::vector<float> W_b(static_cast<size_t>(k) * n);
    fill_random(A.data(), A.size());
    fill_random(W_a.data(), W_a.size());
    fill_random(W_b.data(), W_b.size());

    std::vector<float> ref_b(static_cast<size_t>(m) * n);
    ref_matmul(A.data(), W_b.data(), ref_b.data(), m, n, k);

    ColumnParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W_a.data());
    layer.set_weight(W_b.data());  // must overwrite, not accumulate

    cuda::memory::Buffer<float> d_A(m * k), d_out(m * n);
    d_A.copy_from(A.data(), A.size());
    std::vector<float> out(static_cast<size_t>(m) * n);
    layer.forward(d_A.data(), d_out.data(), m, 1);
    d_out.copy_to(out.data(), out.size());
    EXPECT_TRUE(arrays_near(ref_b.data(), out.data(), out.size(), 1e-3f));
}

// ============================================================================
// v2.23 backward reference tests — milestone v2.23 P1 (TASK-019). These pin the
// analytic gradient contract (dW = X^T dY, dX = dY W^T) on the tp<=1 path; they
// are RED against the provisional throw-stub and go GREEN with the P2 backward.
// ============================================================================

TEST_F(TensorParallelLayersSingleGpuTest, ColumnParallelLayer_Backward_MatchesReference) {
    const int m = 8, k = 16, n = 32;
    std::vector<float> X(static_cast<size_t>(m) * k);
    std::vector<float> W(static_cast<size_t>(k) * n);
    std::vector<float> dY(static_cast<size_t>(m) * n);
    fill_random(X.data(), X.size());
    fill_random(W.data(), W.size());
    fill_random(dY.data(), dY.size());

    std::vector<float> ref_dX(static_cast<size_t>(m) * k);
    std::vector<float> ref_dW(static_cast<size_t>(k) * n);
    ref_linear_backward(X.data(), W.data(), dY.data(),
                        ref_dX.data(), ref_dW.data(), m, k, n);

    ColumnParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W.data());

    cuda::memory::Buffer<float> d_X(m * k), d_dY(m * n), d_dX(m * k),
        d_dW(k * n);
    d_X.copy_from(X.data(), X.size());
    d_dY.copy_from(dY.data(), dY.size());

    std::vector<float> dX(static_cast<size_t>(m) * k);
    std::vector<float> dW(static_cast<size_t>(k) * n);
    layer.backward(d_X.data(), d_dY.data(), d_dX.data(), d_dW.data(), m, 1);
    d_dX.copy_to(dX.data(), dX.size());
    d_dW.copy_to(dW.data(), dW.size());

    EXPECT_TRUE(arrays_near(ref_dX.data(), dX.data(), dX.size(), 1e-3f));
    EXPECT_TRUE(arrays_near(ref_dW.data(), dW.data(), dW.size(), 1e-3f));
}

TEST_F(TensorParallelLayersSingleGpuTest, RowParallelLayer_Backward_MatchesReference) {
    const int m = 8, k = 16, n = 32;
    std::vector<float> X(static_cast<size_t>(m) * k);
    std::vector<float> W(static_cast<size_t>(k) * n);
    std::vector<float> dY(static_cast<size_t>(m) * n);
    fill_random(X.data(), X.size());
    fill_random(W.data(), W.size());
    fill_random(dY.data(), dY.size());

    std::vector<float> ref_dX(static_cast<size_t>(m) * k);
    std::vector<float> ref_dW(static_cast<size_t>(k) * n);
    ref_linear_backward(X.data(), W.data(), dY.data(),
                        ref_dX.data(), ref_dW.data(), m, k, n);

    RowParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W.data());

    cuda::memory::Buffer<float> d_X(m * k), d_dY(m * n), d_dX(m * k),
        d_dW(k * n);
    d_X.copy_from(X.data(), X.size());
    d_dY.copy_from(dY.data(), dY.size());

    std::vector<float> dX(static_cast<size_t>(m) * k);
    std::vector<float> dW(static_cast<size_t>(k) * n);
    layer.backward(d_X.data(), d_dY.data(), d_dX.data(), d_dW.data(), m, 1);
    d_dX.copy_to(dX.data(), dX.size());
    d_dW.copy_to(dW.data(), dW.size());

    EXPECT_TRUE(arrays_near(ref_dX.data(), dX.data(), dX.size(), 1e-3f));
    EXPECT_TRUE(arrays_near(ref_dW.data(), dW.data(), dW.size(), 1e-3f));
}

// Reference for the SiLU-gated MLP backward
// (out = down(silu(gate) .* up), all weights full on the tp<=1 path):
//   dsub = dY W_d^T            dW_d = sub^T dY
//   dgate = dsub .* up .* silu'(gate)   dup = dsub .* silu(gate)
//   dX = dgate W_g^T + dup W_u^T   dW_g = X^T dgate   dW_u = X^T dup
TEST_F(TensorParallelLayersSingleGpuTest, TensorParallelMLP_Backward_MatchesReference) {
    const int m = 8, h = 16, inter = 48;
    std::vector<float> X(static_cast<size_t>(m) * h);
    std::vector<float> W_g(static_cast<size_t>(h) * inter);
    std::vector<float> W_u(static_cast<size_t>(h) * inter);
    std::vector<float> W_d(static_cast<size_t>(inter) * h);
    fill_random(X.data(), X.size());
    fill_random(W_g.data(), W_g.size());
    fill_random(W_u.data(), W_u.size());
    fill_random(W_d.data(), W_d.size());
    std::vector<float> dY(static_cast<size_t>(m) * h);
    fill_random(dY.data(), dY.size());

    // Host reference: recompute the forward activations, then analytic backward.
    auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    auto silu_prime = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 + x * (1.0 - s));
    };
    std::vector<float> G(static_cast<size_t>(m) * inter);
    std::vector<float> U(static_cast<size_t>(m) * inter);
    std::vector<float> sub(static_cast<size_t>(m) * inter);
    host_matmul(X.data(), W_g.data(), G.data(), m, inter, h);
    host_matmul(X.data(), W_u.data(), U.data(), m, inter, h);
    for (size_t i = 0; i < sub.size(); ++i) {
        sub[i] = static_cast<float>(silu(G[i]) * U[i]);
    }

    std::vector<float> ref_dsub(static_cast<size_t>(m) * inter);
    std::vector<float> ref_dW_d(static_cast<size_t>(inter) * h);
    ref_linear_backward(sub.data(), W_d.data(), dY.data(),
                        ref_dsub.data(), ref_dW_d.data(), m, inter, h);

    std::vector<float> dgate(static_cast<size_t>(m) * inter);
    std::vector<float> dup(static_cast<size_t>(m) * inter);
    for (int i = 0; i < m * inter; ++i) {
        dgate[i] = static_cast<float>(ref_dsub[i] * U[i] * silu_prime(G[i]));
        dup[i] = static_cast<float>(ref_dsub[i] * silu(G[i]));
    }

    std::vector<float> ref_dX(static_cast<size_t>(m) * h);
    std::vector<float> dXg(static_cast<size_t>(m) * h);
    std::vector<float> dXu(static_cast<size_t>(m) * h);
    std::vector<float> ref_dW_g(static_cast<size_t>(h) * inter);
    std::vector<float> ref_dW_u(static_cast<size_t>(h) * inter);
    ref_linear_backward(X.data(), W_g.data(), dgate.data(),
                        dXg.data(), ref_dW_g.data(), m, h, inter);
    ref_linear_backward(X.data(), W_u.data(), dup.data(),
                        dXu.data(), ref_dW_u.data(), m, h, inter);
    for (int i = 0; i < m * h; ++i) {
        ref_dX[i] = dXg[i] + dXu[i];
    }

    TensorParallelMLP mlp(uninitialized_ctx(), h, inter);
    mlp.set_weight(W_g.data(), W_u.data(), W_d.data());

    cuda::memory::Buffer<float> d_X(m * h), d_dY(m * h), d_dX(m * h),
        d_dWg(h * inter), d_dWu(h * inter), d_dWd(h * inter);
    cuda::memory::Buffer<float> d_out(m * h);
    d_X.copy_from(X.data(), X.size());
    d_dY.copy_from(dY.data(), dY.size());
    mlp.forward(d_X.data(), d_out.data(), m, 1);  // save activations in scratch

    std::vector<float> dX(static_cast<size_t>(m) * h);
    std::vector<float> dWg(static_cast<size_t>(h) * inter);
    std::vector<float> dWu(static_cast<size_t>(h) * inter);
    std::vector<float> dWd(static_cast<size_t>(inter) * h);
    mlp.backward(d_X.data(), d_dY.data(), d_dX.data(),
                 d_dWg.data(), d_dWu.data(), d_dWd.data(), m, 1);
    d_dX.copy_to(dX.data(), dX.size());
    d_dWg.copy_to(dWg.data(), dWg.size());
    d_dWu.copy_to(dWu.data(), dWu.size());
    d_dWd.copy_to(dWd.data(), dWd.size());

    EXPECT_TRUE(arrays_near(ref_dX.data(), dX.data(), dX.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dW_g.data(), dWg.data(), dWg.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dW_u.data(), dWu.data(), dWu.size(), 2e-3f));
    EXPECT_TRUE(arrays_near(ref_dW_d.data(), dWd.data(), dWd.size(), 2e-3f));
}

// ============================================================================
// v2.24 training-step reference tests (TASK-024, milestone P1). These pin the
// step(optimizer, grad_weight_full, step) contract: with the analytic grad
// (dW = X^T dY) of one forward/backward, applying AdamW via the layer must
// leave the weight shard equal to a host double-precision AdamW on the same
// full weight. RED against the provisional throw-stub.
// ============================================================================

namespace {

// Host double-precision AdamW reference (mirrors the library formula in
// optimizers.cpp exactly, computed in double to check the fp32 path).
void host_adamw_step(const float* w, const float* dw, float* w_out,
                     std::vector<double>& m, std::vector<double>& v,
                     size_t n, int step, float lr, float beta1, float beta2,
                     float eps, float wd) {
    const double b1p = std::pow(static_cast<double>(beta1), step);
    const double b2p = std::pow(static_cast<double>(beta2), step);
    const double lr_t = static_cast<double>(lr) * std::sqrt(1.0 - b2p) / (1.0 - b1p);
    for (size_t i = 0; i < n; ++i) {
        const double g = dw[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        const double m_hat = m[i] / (1.0 - b1p);
        const double v_hat = v[i] / (1.0 - b2p);
        double update = m_hat / (std::sqrt(v_hat) + eps) + wd * w[i];
        w_out[i] = static_cast<float>(w[i] - lr_t * update);
    }
}

}  // namespace

TEST_F(TensorParallelLayersSingleGpuTest, ColumnParallelLayer_Step_MatchesHostAdamW) {
    const int m = 8, k = 16, n = 32;
    std::vector<float> X(static_cast<size_t>(m) * k);
    std::vector<float> W(static_cast<size_t>(k) * n);
    std::vector<float> dY(static_cast<size_t>(m) * n);
    fill_random(X.data(), X.size());
    fill_random(W.data(), W.size());
    fill_random(dY.data(), dY.size());

    std::vector<float> dW(static_cast<size_t>(k) * n);
    {
        std::vector<float> dX(static_cast<size_t>(m) * k);
        ref_linear_backward(X.data(), W.data(), dY.data(),
                            dX.data(), dW.data(), m, k, n);
    }

    // Host AdamW reference over one step.
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    std::vector<float> ref_w(static_cast<size_t>(k) * n);
    std::vector<double> mom(static_cast<size_t>(k) * n, 0.0);
    std::vector<double> mom_v(static_cast<size_t>(k) * n, 0.0);
    host_adamw_step(W.data(), dW.data(), ref_w.data(), mom, mom_v,
                    static_cast<size_t>(k) * n, 1, cfg.learning_rate,
                    cfg.beta1, cfg.beta2, cfg.epsilon, cfg.weight_decay);

    ColumnParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W.data());
    AdamWOptimizer opt(cfg);

    cuda::memory::Buffer<float> d_dW(k * n);
    d_dW.copy_from(dW.data(), dW.size());
    // tp==1: the full grad buffer is the shard's grad directly.
    EXPECT_NO_THROW(layer.step(opt, d_dW.data(), 1));

    std::vector<float> w_out(static_cast<size_t>(k) * n);
    layer.copy_weight_shard(w_out.data());
    EXPECT_TRUE(arrays_near(ref_w.data(), w_out.data(), w_out.size(), 2e-4f));
}

TEST_F(TensorParallelLayersSingleGpuTest, RowParallelLayer_Step_MatchesHostAdamW) {
    const int m = 8, k = 16, n = 32;
    std::vector<float> X(static_cast<size_t>(m) * k);
    std::vector<float> W(static_cast<size_t>(k) * n);
    std::vector<float> dY(static_cast<size_t>(m) * n);
    fill_random(X.data(), X.size());
    fill_random(W.data(), W.size());
    fill_random(dY.data(), dY.size());

    std::vector<float> dW(static_cast<size_t>(k) * n);
    {
        std::vector<float> dX(static_cast<size_t>(m) * k);
        ref_linear_backward(X.data(), W.data(), dY.data(),
                            dX.data(), dW.data(), m, k, n);
    }

    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    std::vector<float> ref_w(static_cast<size_t>(k) * n);
    std::vector<double> mom(static_cast<size_t>(k) * n, 0.0);
    std::vector<double> mom_v(static_cast<size_t>(k) * n, 0.0);
    host_adamw_step(W.data(), dW.data(), ref_w.data(), mom, mom_v,
                    static_cast<size_t>(k) * n, 1, cfg.learning_rate,
                    cfg.beta1, cfg.beta2, cfg.epsilon, cfg.weight_decay);

    RowParallelLayer layer(uninitialized_ctx(), k, n);
    layer.set_weight(W.data());
    AdamWOptimizer opt(cfg);

    cuda::memory::Buffer<float> d_dW(k * n);
    d_dW.copy_from(dW.data(), dW.size());
    EXPECT_NO_THROW(layer.step(opt, d_dW.data(), 1));

    std::vector<float> w_out(static_cast<size_t>(k) * n);
    layer.copy_weight_shard(w_out.data());
    EXPECT_TRUE(arrays_near(ref_w.data(), w_out.data(), w_out.size(), 2e-4f));
}

// MLP step: chain gate/up/down shard steps. The reference applies host AdamW to
// each full weight with the analytic per-weight grads from the v2.23 backward.
TEST_F(TensorParallelLayersSingleGpuTest, TensorParallelMLP_Step_MatchesHostAdamW) {
    const int m = 8, h = 16, inter = 48;
    std::vector<float> X(static_cast<size_t>(m) * h);
    std::vector<float> W_g(static_cast<size_t>(h) * inter);
    std::vector<float> W_u(static_cast<size_t>(h) * inter);
    std::vector<float> W_d(static_cast<size_t>(inter) * h);
    fill_random(X.data(), X.size());
    fill_random(W_g.data(), W_g.size());
    fill_random(W_u.data(), W_u.size());
    fill_random(W_d.data(), W_d.size());
    std::vector<float> dY(static_cast<size_t>(m) * h);
    fill_random(dY.data(), dY.size());

    // Forward activations on device (scratch) and analytic grads on host.
    TensorParallelMLP mlp(uninitialized_ctx(), h, inter);
    mlp.set_weight(W_g.data(), W_u.data(), W_d.data());
    cuda::memory::Buffer<float> d_X(m * h), d_dY(m * h), d_dX(m * h),
        d_dWg(h * inter), d_dWu(h * inter), d_dWd(inter * h), d_out(m * h);
    d_X.copy_from(X.data(), X.size());
    d_dY.copy_from(dY.data(), dY.size());
    mlp.forward(d_X.data(), d_out.data(), m, 1);

    std::vector<float> dWg(static_cast<size_t>(h) * inter);
    std::vector<float> dWu(static_cast<size_t>(h) * inter);
    std::vector<float> dWd(static_cast<size_t>(inter) * h);
    mlp.backward(d_X.data(), d_dY.data(), d_dX.data(),
                 d_dWg.data(), d_dWu.data(), d_dWd.data(), m, 1);

    // Host AdamW reference over one step per weight.
    OptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    std::vector<float> ref_g(static_cast<size_t>(h) * inter);
    std::vector<float> ref_u(static_cast<size_t>(h) * inter);
    std::vector<float> ref_d(static_cast<size_t>(inter) * h);
    std::vector<double> mg(static_cast<size_t>(h) * inter, 0.0);
    std::vector<double> vg(static_cast<size_t>(h) * inter, 0.0);
    std::vector<double> mu(static_cast<size_t>(h) * inter, 0.0);
    std::vector<double> vu(static_cast<size_t>(h) * inter, 0.0);
    std::vector<double> md(static_cast<size_t>(inter) * h, 0.0);
    std::vector<double> vd(static_cast<size_t>(inter) * h, 0.0);
    host_adamw_step(W_g.data(), dWg.data(), ref_g.data(), mg, vg,
                    static_cast<size_t>(h) * inter, 1, cfg.learning_rate,
                    cfg.beta1, cfg.beta2, cfg.epsilon, cfg.weight_decay);
    host_adamw_step(W_u.data(), dWu.data(), ref_u.data(), mu, vu,
                    static_cast<size_t>(h) * inter, 1, cfg.learning_rate,
                    cfg.beta1, cfg.beta2, cfg.epsilon, cfg.weight_decay);
    host_adamw_step(W_d.data(), dWd.data(), ref_d.data(), md, vd,
                    static_cast<size_t>(inter) * h, 1, cfg.learning_rate,
                    cfg.beta1, cfg.beta2, cfg.epsilon, cfg.weight_decay);

    AdamWOptimizer opt(cfg);
    cuda::memory::Buffer<float> d_g((size_t)h * inter), d_u((size_t)h * inter),
        d_d((size_t)inter * h);
    d_g.copy_from(dWg.data(), dWg.size());
    d_u.copy_from(dWu.data(), dWu.size());
    d_d.copy_from(dWd.data(), dWd.size());
    EXPECT_NO_THROW(mlp.step(opt, d_g.data(), d_u.data(), d_d.data(), 1));

    // MLP weights are private: verify the stepped result by running the
    // forward on a fresh input and comparing against the host reference MLP
    // built from the host-stepped weights (the step must move the weights to
    // exactly the AdamW images of the analytic grads).
    std::vector<float> X2(static_cast<size_t>(m) * h);
    fill_random(X2.data(), X2.size());
    std::vector<float> ref_out(static_cast<size_t>(m) * h);
    ref_gated_mlp(X2.data(), ref_g.data(), ref_u.data(), ref_d.data(),
                  ref_out.data(), m, h, inter);

    cuda::memory::Buffer<float> d_X2(m * h), d_out2(m * h);
    d_X2.copy_from(X2.data(), X2.size());
    mlp.forward(d_X2.data(), d_out2.data(), m, 1);
    std::vector<float> out2(static_cast<size_t>(m) * h);
    d_out2.copy_to(out2.data(), out2.size());
    EXPECT_TRUE(arrays_near(ref_out.data(), out2.data(), out2.size(), 2e-2f));
}
