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
