#pragma once

#include <cuda_runtime.h>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/neural/optimizers/optimizers.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace cuda::neural {

struct LayerNormResult {
    float* output;
    float* mean;
    float* variance;
    float* d_output;
    float* d_mean;
    float* d_variance;
    int size;

    LayerNormResult() : output(nullptr), mean(nullptr), variance(nullptr),
                        d_output(nullptr), d_mean(nullptr), d_variance(nullptr), size(0) {}
    explicit LayerNormResult(int size);
    ~LayerNormResult();

    void upload();
    void download();
    void clear();
};

struct LayerNormParams {
    int normalized_shape;
    float eps = 1e-5f;
    bool elementwise_affine = true;
};

void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* variance,
    int batch_size,
    int normalized_shape,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

void layer_norm_inference(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int normalized_shape,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

/**
 * @brief LayerNorm backward: analytic dL/dx, dL/dgamma, dL/dbeta (v2.28)
 *
 * Forward was y = gamma * xhat + beta, xhat = (x - mean)/sqrt(var + eps) per
 * row. Given the upstream gradient dL/dy:
 *   dgamma[j] = sum_i dy[i][j] * xhat[i][j]
 *   dbeta[j]  = sum_i dy[i][j]
 *   per row i: dxhat[j] = dy[i][j] * gamma[j]; then
 *     dL/dx[i][j] = inv_std_i * (H*dxhat[j] - sum_k dxhat[k] - xhat[j]*sum_k dxhat[k]*xhat[k]) / H
 * Each row normalizes independently over the (replicated) hidden dim, so this
 * is collective-free on the TP stack — same input as forward.
 */
void layer_norm_backward(
    const float* input,
    const float* gamma,
    const float* grad_output,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int batch_size,
    int normalized_shape,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

/**
 * @class LayerNorm
 * @brief Trainable, replicated LayerNorm on the parallel stack (v2.28)
 *
 * Milestone v2.28 (TASK-039, DEC-014): the v2.21-27 stack trains exactly one
 * model — an unnormalized single block (attention -> MLP). Real transformer
 * blocks are pre-norm (LN -> MHA -> +x -> LN -> MLP -> +x), and the legacy
 * free-function LayerNorm was forward-only (no backward, so untrainable).
 * This class makes normalization trainable: it owns device gamma/beta (the
 * affine scale/shift) and adds device forward + backward.
 *
 * Collective-free by construction: block inputs/outputs are replicated
 * [m x hidden] on every rank (the v2.21-26 convention; only intermediate
 * shards are rank-local), so a per-row norm over the full hidden dim is
 * identical on every rank and gamma/beta are replicated [hidden] vectors —
 * no AllReduce. The caller owns one AdamW per gamma/beta tensor and steps it
 * (one-optimizer-per-tensor convention from the v2.24 cpp-review).
 */
class LayerNorm {
public:
    /**
     * @brief Construct a trainable, replicated LayerNorm
     * @param hidden Normalized (hidden) dimension; must be > 0
     * @param eps Stability epsilon added under the variance
     * @throws std::invalid_argument on non-positive hidden
     */
    explicit LayerNorm(int hidden, float eps = 1e-5f);

    // Non-copyable/non-movable (owns device gamma/beta + scratch buffers).
    LayerNorm(const LayerNorm&) = delete;
    LayerNorm& operator=(const LayerNorm&) = delete;
    LayerNorm(LayerNorm&&) = delete;
    LayerNorm& operator=(LayerNorm&&) = delete;

    ~LayerNorm();

    /**
     * @brief Upload the full affine weight vectors (replicated on every rank)
     * @param gamma [hidden] scale
     * @param beta [hidden] shift
     */
    void set_weight(const float* gamma, const float* beta);

    /**
     * @brief Forward: y = gamma * (x - mean)/sqrt(var + eps) + beta per row
     * @param input Device input [m x hidden] (replicated)
     * @param output Device output [m x hidden] (replicated)
     * @param batch Rows
     * @param stream CUDA stream
     */
    void forward(const float* input, float* output, int batch,
                 cudaStream_t stream = nullptr);

    /**
     * @brief Backward: analytic dL/dx + per-hidden dgamma/dbeta (v2.28)
     * @param input Forward input [m x hidden] (replicated)
     * @param grad_output Upstream gradient [m x hidden] (replicated)
     * @param grad_input dL/dx [m x hidden] (replicated)
     * @param grad_gamma dL/dgamma [hidden] (sum over rows)
     * @param grad_beta dL/dbeta [hidden] (sum over rows)
     * @param batch Rows
     * @param stream CUDA stream
     */
    void backward(const float* input, const float* grad_output,
                  float* grad_input, float* grad_gamma, float* grad_beta,
                  int batch, cudaStream_t stream = nullptr);

    /**
     * @brief Copy this LayerNorm's gamma/beta to host memory
     */
    void copy_weights(float* gamma, float* beta) const;

    /**
     * @brief Apply one AdamW step to the private gamma/beta (v2.28)
     *
     * gamma/beta are replicated full-[hidden] vectors on every rank, and the
     * caller's grad_gamma/grad_beta are the full (replicated) analytic grads,
     * so each rank runs the identical update — no AllReduce. Mirrors the
     * one-optimizer-per-tensor convention of the layer step() surface.
     *
     * @param opt_gamma AdamW for gamma
     * @param opt_beta AdamW for beta
     * @param grad_gamma Full dL/dgamma [hidden]
     * @param grad_beta Full dL/dbeta [hidden]
     * @param step_no Optimizer step counter
     * @param stream CUDA stream
     */
    void step(optimizers::AdamWOptimizer& opt_gamma,
              optimizers::AdamWOptimizer& opt_beta,
              const float* grad_gamma, const float* grad_beta, int step_no,
              cudaStream_t stream = nullptr);

    /**
     * @brief Normalized (hidden) dimension
     */
    [[nodiscard]] int hidden() const;

    /**
     * @brief Stability epsilon
     */
    [[nodiscard]] float eps() const;

private:
    void ensure_stats(int batch);

    int hidden_ = 0;
    float eps_ = 1e-5f;
    std::unique_ptr<cuda::memory::Buffer<float>> d_gamma_;
    std::unique_ptr<cuda::memory::Buffer<float>> d_beta_;
    // Backward per-row stats scratch: [mean, inv, sum_dxhat, sum_dxhat*xhat]
    // per row (4 * batch floats), sized on demand.
    std::unique_ptr<cuda::memory::Buffer<float>> d_stats_;
    int stats_batch_ = 0;
};

}  // namespace cuda::neural
