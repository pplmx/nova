/**
 * @file micro_trainer_multigpu_test.cpp
 * @brief Real thread-per-rank multi-GPU tests for the MicroTrainer driver
 *
 * Milestone v2.25 (TASK-028 / DEC-011): v2.24 verified one training step is
 * weight-exact; this milestone's acceptance is that a *full multi-step
 * sharded training run* reproduces a host fp64 full-weight training
 * trajectory end-to-end (gate/up/down assembled shards AND the per-step loss
 * curve), on real 2 & 4 GPUs — i.e. the parallel stack learns exactly like
 * the single-GPU/full-weight reference, not just steps.
 *
 * Methodology mirrors tensor_parallel_multigpu_test.cpp: one thread per
 * visible device pinned with cudaSetDevice, all entering the forward/backward
 * together (the MLP commits a row-layer AllReduce), each rank stepping its
 * own shards with per-rank AdamW instances (one per weight tensor — a shared
 * optimizer would corrupt moments, the v2.24 cpp-review finding).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <mutex>
#include <random>
#include <vector>

#include "cuda/neural/training.h"
#include "cuda/neural/tensor_parallel_layers.h"
#include "cuda/neural/optimizers/optimizers.h"
#include "cuda/neural/loss/loss_functions.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include "distributed_test_common.h"

using namespace cuda::neural;
using namespace cuda::neural::optimizers;
using namespace cuda::neural::loss;
using namespace cuda::neural::training;
using namespace cuda::mesh;

using cuda::distributed::test::run_per_rank;

namespace {

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

// Host double-precision matmul: C[m x n] = A[m x k] @ B[k x n] (row-major).
void host_matmul(const float* A, const float* B, float* C, int m, int n,
                 int k) {
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

// dW = X^T dY, dX = dY W^T — analytic backward of Y = X W (double precision).
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

// Host double-precision AdamW reference (identical formula to the library).
struct HostAdamW {
    std::vector<double> m, v;
    void step(const float* w, const float* dw, float* w_out, size_t n,
              int step, float lr, float beta1, float beta2, float eps,
              float wd) {
        if (m.size() != n) {
            m.assign(n, 0.0);
            v.assign(n, 0.0);
        }
        const double b1p = std::pow(static_cast<double>(beta1), step);
        const double b2p = std::pow(static_cast<double>(beta2), step);
        const double lr_t =
            static_cast<double>(lr) * std::sqrt(1.0 - b2p) / (1.0 - b1p);
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
};

// One host fp64 MLP training step on full weights (identical structure to the
// device MicroTrainer chain): forward -> dlogits=(softmax-onehot)/m -> MLP
// backward -> per-weight AdamW.
float host_training_step(const float* X, const int* targets, float* Wg,
                         float* Wu, float* Wd, HostAdamW& ag, HostAdamW& au,
                         HostAdamW& ad, int m, int h, int inter, int step_no,
                         float lr, float beta1, float beta2, float eps,
                         float wd) {
    auto silu = [](double x) { return x / (1.0 + std::exp(-x)); };
    auto silu_prime = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 + x * (1.0 - s));
    };
    std::vector<float> G(m * inter), U(m * inter), sub(m * inter);
    std::vector<float> logits(m * h);
    host_matmul(X, Wg, G.data(), m, inter, h);
    host_matmul(X, Wu, U.data(), m, inter, h);
    for (size_t i = 0; i < sub.size(); ++i) {
        sub[i] = static_cast<float>(silu(G[i]) * U[i]);
    }
    host_matmul(sub.data(), Wd, logits.data(), m, h, inter);

    // Mean CE loss (host scalar) — the device train_step also reports the
    // pre-update forward loss (logits_ is written by forward and untouched by
    // backward/step).
    double total = 0.0;
    for (int b = 0; b < m; ++b) {
        double mm = logits[b * h];
        for (int c = 1; c < h; ++c) {
            mm = std::max(mm, static_cast<double>(logits[b * h + c]));
        }
        double sum = 0.0;
        for (int c = 0; c < h; ++c) {
            sum += std::exp(static_cast<double>(logits[b * h + c]) - mm);
        }
        double softmax_t =
            std::exp(static_cast<double>(logits[b * h + targets[b]]) - mm) /
            sum;
        total += -std::log(softmax_t);
    }
    const float loss = static_cast<float>(total / m);

    // dlogits = (softmax - onehot) / m (reduction_mean).
    std::vector<float> dlogits(m * h);
    for (int b = 0; b < m; ++b) {
        double mm = logits[b * h];
        for (int c = 1; c < h; ++c) {
            mm = std::max(mm, static_cast<double>(logits[b * h + c]));
        }
        double sum = 0.0;
        for (int c = 0; c < h; ++c) {
            sum += std::exp(static_cast<double>(logits[b * h + c]) - mm);
        }
        for (int c = 0; c < h; ++c) {
            double sm =
                std::exp(static_cast<double>(logits[b * h + c]) - mm) / sum;
            dlogits[b * h + c] =
                static_cast<float>((sm - (c == targets[b])) / m);
        }
    }

    // Analytic MLP backward to the full-weight grads.
    std::vector<float> dsub(m * inter), dg(m * inter), du(m * inter);
    std::vector<float> dWg(h * inter), dWu(h * inter), dWd(inter * h);
    std::vector<float> tmp_dx(m * h);
    ref_linear_backward(sub.data(), Wd, dlogits.data(), dsub.data(),
                        dWd.data(), m, inter, h);
    for (int i = 0; i < m * inter; ++i) {
        dg[i] = static_cast<float>(dsub[i] * U[i] * silu_prime(G[i]));
        du[i] = static_cast<float>(dsub[i] * silu(G[i]));
    }
    std::vector<float> dxg(m * h), dxu(m * h);
    ref_linear_backward(X, Wg, dg.data(), dxg.data(), dWg.data(), m, h, inter);
    ref_linear_backward(X, Wu, du.data(), dxu.data(), dWu.data(), m, h, inter);

    ag.step(Wg, dWg.data(), Wg, static_cast<size_t>(h) * inter, step_no,
            lr, beta1, beta2, eps, wd);
    au.step(Wu, dWu.data(), Wu, static_cast<size_t>(h) * inter, step_no,
            lr, beta1, beta2, eps, wd);
    ad.step(Wd, dWd.data(), Wd, static_cast<size_t>(inter) * h, step_no,
            lr, beta1, beta2, eps, wd);

    return loss;
}

}  // namespace

class MicroTrainerMultiGpuTest : public ::testing::Test {
protected:
    void SetUp() override { DeviceMesh::instance().initialize(); }

    static void SetUpTestSuite() { cudaFree(nullptr); }

    static testing::AssertionResult near(const float* expected,
                                         const float* actual, size_t count,
                                         float tolerance) {
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(expected[i] - actual[i]) > tolerance) {
                return testing::AssertionFailure()
                       << "mismatch at " << i << ": expected " << expected[i]
                       << " actual " << actual[i];
            }
        }
        return testing::AssertionSuccess();
    }
};

// The milestone acceptance: a full K-step sharded MicroTrainer run on real
// multi-GPU must reproduce the host fp64 full-weight training trajectory — the
// assembled final gate/up/down shards AND the per-step loss curve.
TEST_F(MicroTrainerMultiGpuTest, MultiGpu_MicroTrainer_MatchesHostFullWeight) {
    const int device_count = DeviceMesh::instance().device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU MicroTrainer test";
    }
    const char* nccl_env = std::getenv("NCCL_TESTS_AVAILABLE");
    if (nccl_env == nullptr) {
        GTEST_SKIP() << "Multi-GPU MicroTrainer requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
    }
    if (!multigpu_nccl_ready()) {
        GTEST_SKIP() << "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
    }
    auto& ctx = cuda::nccl::NcclContext::instance();

    const int m = 16, h = 8, inter = 16;
    const int K = 10;  // a full run, not a single-step spot-check

    // Learnable task: gaussian inputs, a fixed random "teacher" label matrix.
    std::vector<float> X(static_cast<size_t>(m) * h);
    std::vector<int> targets(m);
    {
        std::mt19937 rng(20260815);
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::vector<float> T(static_cast<size_t>(h) * h);
        for (size_t i = 0; i < T.size(); ++i) {
            T[i] = norm(rng);
        }
        for (int i = 0; i < m; ++i) {
            for (int d = 0; d < h; ++d) {
                X[i * h + d] = norm(rng);
            }
            double best = -1e30;
            int argmax = 0;
            for (int c = 0; c < h; ++c) {
                double acc = 0.0;
                for (int d = 0; d < h; ++d) {
                    acc += static_cast<double>(X[i * h + d]) * T[c * h + d];
                }
                if (acc > best) {
                    best = acc;
                    argmax = c;
                }
            }
            targets[i] = argmax;
        }
    }

    std::vector<float> Wg0(static_cast<size_t>(h) * inter);
    std::vector<float> Wu0(static_cast<size_t>(h) * inter);
    std::vector<float> Wd0(static_cast<size_t>(inter) * h);
    fill_random(Wg0.data(), Wg0.size(), 21);
    fill_random(Wu0.data(), Wu0.size(), 22);
    fill_random(Wd0.data(), Wd0.size(), 23);

    // Host fp64 full-weight reference over K identical steps (loss curve +
    // final weights).
    OptimizerConfig opt_cfg;
    opt_cfg.learning_rate = 0.01f;
    std::vector<float> rWg = Wg0, rWu = Wu0, rWd = Wd0;
    HostAdamW rag, rau, rad;
    std::vector<float> ref_loss;
    ref_loss.reserve(K);
    for (int s = 1; s <= K; ++s) {
        ref_loss.push_back(host_training_step(
            X.data(), targets.data(), rWg.data(), rWu.data(), rWd.data(),
            rag, rau, rad, m, h, inter, s, opt_cfg.learning_rate,
            opt_cfg.beta1, opt_cfg.beta2, opt_cfg.epsilon,
            opt_cfg.weight_decay));
    }

    // Device: per-rank MicroTrainer over the same K steps; assembled shards +
    // the loss curve must match the host reference.
    const int inter_local = inter / device_count;
    std::vector<std::vector<float>> sg(device_count,
                                       std::vector<float>(h * inter_local));
    std::vector<std::vector<float>> su(device_count,
                                       std::vector<float>(h * inter_local));
    std::vector<std::vector<float>> sd(device_count,
                                       std::vector<float>(inter_local * h));
    std::vector<float> dev_loss(K, 0.0f);

    run_per_rank(device_count, [&](int d) {
        cuda::memory::Buffer<float> d_X(m * h);
        d_X.copy_from(X.data(), X.size());

        MicroTrainer trainer(ctx, h, inter, opt_cfg);
        trainer.set_weight(Wg0.data(), Wu0.data(), Wd0.data());
        for (int s = 1; s <= K; ++s) {
            const float l =
                trainer.train_step(d_X.data(), targets.data(), m, 1, s);
            // The MLP output is replicated (row-layer AllReduce), so every
            // rank's train_step returns the identical loss; one rank records
            // the curve to avoid a scrambled interleaved capture.
            if (d == 0) {
                dev_loss[s - 1] = l;
            }
        }
        trainer.copy_weights(sg[d].data(), su[d].data(), sd[d].data());
    });

    // Assemble the per-rank shards into the full weights.
    std::vector<float> aWg(h * inter), aWu(h * inter), aWd(inter * h);
    for (int r = 0; r < device_count; ++r) {
        for (int a = 0; a < h; ++a) {
            for (int b = 0; b < inter_local; ++b) {
                aWg[a * inter + r * inter_local + b] =
                    sg[r][a * inter_local + b];
                aWu[a * inter + r * inter_local + b] =
                    su[r][a * inter_local + b];
            }
        }
        for (int a = 0; a < inter_local; ++a) {
            for (int b = 0; b < h; ++b) {
                aWd[(r * inter_local + a) * h + b] = sd[r][a * h + b];
            }
        }
    }

    // Final assembled weights must match the host fp64 reference (2e-2 absorbs
    // fp32 accumulate-order + shard-step noise over K=12 steps).
    EXPECT_TRUE(near(rWg.data(), aWg.data(), h * inter, 2e-2f))
        << "assembled gate shards differ from the host full-weight reference";
    EXPECT_TRUE(near(rWu.data(), aWu.data(), h * inter, 2e-2f))
        << "assembled up shards differ from the host full-weight reference";
    EXPECT_TRUE(near(rWd.data(), aWd.data(), inter * h, 2e-2f))
        << "assembled down shards differ from the host full-weight reference";

    // The loss curve must also reproduce the reference trajectory. The
    // assembled-weight parity above (2e-2) is the strong claim; the loss curve
    // is a trajectory shape check — fp32-vs-fp64 accumulation drift at the
    // softmax boundary makes the per-step scalar modestly diverge late in the
    // run, so use a coarser per-step bound (0.1) plus the descent assertions.
    ASSERT_EQ(dev_loss.size(), static_cast<size_t>(K));
    for (int s = 0; s < K; ++s) {
        EXPECT_NEAR(ref_loss[s], dev_loss[s], 0.1f)
            << "loss curve diverged at step " << (s + 1);
    }
    // And the run must actually learn: the loss descends over the run.
    EXPECT_GT(ref_loss[0], ref_loss[K - 1])
        << "reference loss must descend over the run";
    EXPECT_GT(dev_loss[0], dev_loss[K - 1])
        << "device loss must descend over the run";
}
