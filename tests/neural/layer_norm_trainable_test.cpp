/**
 * @file layer_norm_trainable_test.cpp
 * @brief Single-GPU tests for the v2.28 trainable LayerNorm class
 *
 * Milestone v2.28 (TASK-040 / DEC-014): the legacy layer_norm module was
 * forward-only (no backward, so untrainable) and its kernels were never
 * validated against a reference. This file pins the NEW trainable LayerNorm
 * (class cuda::neural::LayerNorm) contracts: device forward must equal a
 * host-fp64 per-row reference and device backward (d_input, d_gamma, d_beta)
 * must equal the analytic host-fp64 reference. The analytic backward reference
 * itself is self-checked against central finite differences on a tiny case, so
 * "parity" cannot silently bless a wrong formula.
 *
 * P1-RED: LayerNorm::forward/backward are provisional throw-stubs in this
 * milestone's P1, so these tests fail here and turn GREEN with the P2 device
 * kernels (TASK-041).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuda/neural/layer_norm.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using namespace cuda::neural;

namespace {

// Per-element near check: abs diff <= tol, or rel diff <= tol when the ref is
// large (mirrors the established test convention).
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

// Host fp64 per-row LayerNorm forward: y = gamma * (x - mean)/sqrt(var+eps) + beta.
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
            const double xhat =
                (static_cast<double>(xr[j]) - mean) * inv;
            yr[j] = static_cast<float>(
                static_cast<double>(gamma[j]) * xhat + static_cast<double>(beta[j]));
        }
    }
}

// Host fp64 LayerNorm backward: exact analytic chain. Returns xhat, mean and
// inv-std through scratch so the caller can cross-check (finite differences).
void host_ln_backward(const float* x, const float* gamma, const float* dy,
                      float* dx, float* dgamma, float* dbeta, int m, int h,
                      float eps) {
    std::vector<double> mean(m), inv(m), dot(m);
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
        double acc = 0.0;
        for (int j = 0; j < h; ++j) {
            const double xhat =
                (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            acc += static_cast<double>(dy[i * h + j]) * xhat;
        }
        dot[i] = acc;
    }
    // dgamma[j], dbeta[j]: reduce over batch rows.
    for (int j = 0; j < h; ++j) {
        dgamma[j] = 0.0f;
        dbeta[j] = 0.0f;
        for (int i = 0; i < m; ++i) {
            const float* xr = x + i * h;
            const double xhat =
                (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            dgamma[j] += static_cast<float>(static_cast<double>(dy[i * h + j]) * xhat);
            dbeta[j] += dy[i * h + j];
        }
    }
    // d_input per row.
    for (int i = 0; i < m; ++i) {
        const float* xr = x + i * h;
        const float* dyr = dy + i * h;
        float* dxr = dx + i * h;
        // Row stats in fp64.
        double sum_dxhat = 0.0;
        double sum_dxhat_xhat = 0.0;
        std::vector<double> dxhat(h);
        for (int j = 0; j < h; ++j) {
            const double xhat =
                (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            dxhat[j] = static_cast<double>(dyr[j]) * static_cast<double>(gamma[j]);
            sum_dxhat += dxhat[j];
            sum_dxhat_xhat += dxhat[j] * xhat;
        }
        for (int j = 0; j < h; ++j) {
            const double xhat =
                (static_cast<double>(xr[j]) - mean[i]) * inv[i];
            // dL/dx = inv * ( dxhat - mean_dxhat - xhat * mean(dxhat*xhat) )
            const double v = inv[i] * (dxhat[j] - sum_dxhat / h -
                                       xhat * (sum_dxhat_xhat / h));
            dxr[j] = static_cast<float>(v);
        }
    }
}

// Central finite-difference of a linear loss on one row: L(x) =
// <dy_row, ln(x)[row, :]>. Only row `row` is perturbed (LayerNorm normalizes
// each row independently, so this isolates dL/dx[row, :]); the probe is
// computed on the full forward so the perturbed row is actually used.
void finite_diff_dx(const std::vector<float>& x, const std::vector<float>& g,
                    const std::vector<float>& b, int m, int h, float eps,
                    const std::vector<float>& dy_row, int row,
                    std::vector<float>& fd, float hstep = 1e-3f) {
    std::vector<float> y(static_cast<size_t>(m) * h);
    for (int j = 0; j < h; ++j) {
        // Rebuild the perturbed row from the base each column (LayerNorm mixes
        // the whole row, so a leftover bump would corrupt later columns).
        std::vector<float> xp = x;
        std::vector<float> xm = x;
        xp[static_cast<size_t>(row) * h + j] += hstep;
        xm[static_cast<size_t>(row) * h + j] -= hstep;
        host_ln_forward(xp.data(), g.data(), b.data(), y.data(), m, h, eps);
        double lp = 0.0;
        for (int k = 0; k < h; ++k)
            lp += static_cast<double>(dy_row[k]) * y[static_cast<size_t>(row) * h + k];
        host_ln_forward(xm.data(), g.data(), b.data(), y.data(), m, h, eps);
        double lm = 0.0;
        for (int k = 0; k < h; ++k)
            lm += static_cast<double>(dy_row[k]) * y[static_cast<size_t>(row) * h + k];
        fd[j] = static_cast<float>((lp - lm) / (2.0 * hstep));
    }
}

}  // namespace

class LayerNormTrainableTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { cudaFree(nullptr); }
};

// The analytic d_input reference must match a central finite-difference of the
// host forward (a self-check of the test's backward, not of the device).
TEST_F(LayerNormTrainableTest, HostBackwardReferenceMatchesFiniteDifference) {
    const int m = 2, h = 4;
    const float eps = 1e-5f;
    std::vector<float> x(static_cast<size_t>(m) * h);
    std::vector<float> g(h), b(h), dy(static_cast<size_t>(m) * h);
    fill_random(x.data(), x.size(), 101);
    fill_random(g.data(), g.size(), 102, 0.5f, 1.5f);
    fill_random(b.data(), b.size(), 103);
    fill_random(dy.data(), dy.size(), 104);
    std::vector<float> dx(static_cast<size_t>(m) * h), dg(h), db(h);
    host_ln_backward(x.data(), g.data(), dy.data(), dx.data(), dg.data(),
                     db.data(), m, h, eps);
    // Single-row linear probe must reproduce dL/dx on that row.
    for (int row = 0; row < m; ++row) {
        std::vector<float> dy_row(dy.begin() + row * h, dy.begin() + (row + 1) * h);
        std::vector<float> fd(h);
        finite_diff_dx(x, g, b, m, h, eps, dy_row, row, fd);
        for (int j = 0; j < h; ++j) {
            EXPECT_NEAR(dx[row * h + j], fd[j], 5e-2f)
                << "row " << row << " col " << j << " analytic d_input vs finite diff";
        }
    }
}

// Device forward == host fp64 forward (affine on).
TEST_F(LayerNormTrainableTest, ForwardMatchesHostReference) {
    const int m = 16, h = 8;
    const float eps = 1e-5f;
    std::vector<float> x(static_cast<size_t>(m) * h);
    std::vector<float> g(h), b(h);
    fill_random(x.data(), x.size(), 201);
    fill_random(g.data(), g.size(), 202, 0.5f, 1.5f);
    fill_random(b.data(), b.size(), 203);

    std::vector<float> ref(static_cast<size_t>(m) * h);
    host_ln_forward(x.data(), g.data(), b.data(), ref.data(), m, h, eps);

    LayerNorm ln(h, eps);
    ln.set_weight(g.data(), b.data());
    cuda::memory::Buffer<float> d_x(m * h), d_y(m * h);
    d_x.copy_from(x.data(), x.size());
    ln.forward(d_x.data(), d_y.data(), m);
    std::vector<float> y(static_cast<size_t>(m) * h);
    d_y.copy_to(y.data(), y.size());

    // fp32 vs fp64 over 8 dims: mean/var in fp32 lose ~1e-6 relative.
    EXPECT_TRUE(arrays_near(ref.data(), y.data(), y.size(), 2e-4f))
        << "device LayerNorm forward differs from host fp64 reference";
    // Host stats of the device output: each row has ~unit variance.
    for (int i = 0; i < m; ++i) {
        double s = 0.0, v = 0.0;
        for (int j = 0; j < h; ++j) s += y[i * h + j];
        const double mean = s / h;
        for (int j = 0; j < h; ++j) {
            const double d = y[i * h + j] - mean;
            v += d * d;
        }
        EXPECT_NEAR(v / h, 1.0, 1e-3) << "row " << i << " variance not ~1";
    }
}

// Device backward == host fp64 analytic backward (d_input, d_gamma, d_beta).
TEST_F(LayerNormTrainableTest, BackwardMatchesHostReference) {
    const int m = 16, h = 8;
    const float eps = 1e-5f;
    std::vector<float> x(static_cast<size_t>(m) * h);
    std::vector<float> g(h), b(h), dy(static_cast<size_t>(m) * h);
    fill_random(x.data(), x.size(), 301);
    fill_random(g.data(), g.size(), 302, 0.5f, 1.5f);
    fill_random(b.data(), b.size(), 303);
    fill_random(dy.data(), dy.size(), 304);

    std::vector<float> ref_dx(static_cast<size_t>(m) * h), ref_dg(h), ref_db(h);
    host_ln_backward(x.data(), g.data(), dy.data(), ref_dx.data(),
                     ref_dg.data(), ref_db.data(), m, h, eps);

    LayerNorm ln(h, eps);
    ln.set_weight(g.data(), b.data());
    cuda::memory::Buffer<float> d_x(m * h), d_dy(m * h), d_dx(m * h),
        d_dg(h), d_db(h);
    d_x.copy_from(x.data(), x.size());
    d_dy.copy_from(dy.data(), dy.size());
    ln.backward(d_x.data(), d_dy.data(), d_dx.data(), d_dg.data(),
                d_db.data(), m);
    std::vector<float> dx(static_cast<size_t>(m) * h), dg(h), db(h);
    d_dx.copy_to(dx.data(), dx.size());
    d_dg.copy_to(dg.data(), dg.size());
    d_db.copy_to(db.data(), db.size());

    EXPECT_TRUE(arrays_near(ref_dx.data(), dx.data(), dx.size(), 2e-4f))
        << "device d_input differs from host fp64 reference";
    EXPECT_TRUE(arrays_near(ref_dg.data(), dg.data(), dg.size(), 2e-4f))
        << "device d_gamma differs from host fp64 reference";
    EXPECT_TRUE(arrays_near(ref_db.data(), db.data(), db.size(), 2e-4f))
        << "device d_beta differs from host fp64 reference";
}

// copy_weights must round-trip what set_weight uploaded.
TEST_F(LayerNormTrainableTest, CopyWeightsRoundTrips) {
    const int h = 8;
    std::vector<float> g(h), b(h);
    fill_random(g.data(), g.size(), 401, 0.5f, 1.5f);
    fill_random(b.data(), b.size(), 402);

    LayerNorm ln(h);
    ln.set_weight(g.data(), b.data());
    std::vector<float> g2(h), b2(h);
    ln.copy_weights(g2.data(), b2.data());
    EXPECT_TRUE(arrays_near(g.data(), g2.data(), h, 1e-7f));
    EXPECT_TRUE(arrays_near(b.data(), b2.data(), h, 1e-7f));
    EXPECT_EQ(ln.hidden(), h);
}

// One AdamW step on gamma/beta must actually move them (trainable surface).
TEST_F(LayerNormTrainableTest, StepMovesGammaBeta) {
    const int h = 8;
    std::vector<float> g(h, 1.0f), b(h, 0.0f);
    std::vector<float> dg(h, 1.0f), db(h, 0.5f);
    LayerNorm ln(h);
    ln.set_weight(g.data(), b.data());

    optimizers::OptimizerConfig cfg;
    cfg.learning_rate = 0.1f;
    optimizers::AdamWOptimizer og(cfg), ob(cfg);
    ln.step(og, ob, dg.data(), db.data(), 1);

    std::vector<float> g2(h), b2(h);
    ln.copy_weights(g2.data(), b2.data());
    EXPECT_NE(g2[0], 1.0f) << "gamma must move under AdamW";
    EXPECT_NE(b2[0], 0.0f) << "beta must move under AdamW";
}
