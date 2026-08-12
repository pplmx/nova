/**
 * @file tensor_parallel_layers.cpp
 * @brief Tensor-parallel layer implementations
 *
 * These layer wrappers (ColumnParallelLayer / RowParallelLayer / TensorParallelMLP)
 * shipped in v1.3 as scaffolding and were never implemented: they hold no weight
 * storage and their forward() bodies passed B = nullptr into the tensor-parallel
 * matmul (a cuBLAS null-pointer crash) / returned an empty result. With no callers
 * anywhere in the tree they are dead, non-functional surfaces, so every forward()
 * now fails fast with an explicit error until a real weight-managed implementation
 * lands (DEC-006). The supported primitive is TensorParallelMatmul.
 */

#include "cuda/neural/tensor_parallel_layers.h"

#include <stdexcept>
#include <string>

namespace cuda::neural {

namespace {

[[noreturn]] void throw_unimplemented(const char* layer) {
    throw std::runtime_error(
        std::string(layer) +
        "::forward is not implemented: the layer owns no weight storage (its "
        "pre-v2.20 body passed a null weight pointer to the cuBLAS-backed "
        "TensorParallelMatmul, which would crash). Use the supported "
        "TensorParallelMatmul primitive directly, or await a weight-managed "
        "layer implementation (DEC-006).");
}

}  // namespace

ColumnParallelLayer::ColumnParallelLayer(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int tp_degree)
    : hidden_dim_(hidden_dim),
      tp_degree_(tp_degree),
      q_proj_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::ColumnParallel)),
      k_proj_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::ColumnParallel)),
      v_proj_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::ColumnParallel)) {}

void ColumnParallelLayer::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
    (void)input;
    (void)output;
    (void)batch;
    (void)seq;
    throw_unimplemented("ColumnParallelLayer");
}

int ColumnParallelLayer::hidden_dim() const {
    return hidden_dim_;
}

int ColumnParallelLayer::tp_degree() const {
    return tp_degree_;
}

RowParallelLayer::RowParallelLayer(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int tp_degree)
    : hidden_dim_(hidden_dim),
      tp_degree_(tp_degree),
      matmul_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::RowParallel)),
      reducer_(ctx) {}

void RowParallelLayer::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
    (void)input;
    (void)output;
    (void)batch;
    (void)seq;
    throw_unimplemented("RowParallelLayer");
}

int RowParallelLayer::hidden_dim() const {
    return hidden_dim_;
}

int RowParallelLayer::tp_degree() const {
    return tp_degree_;
}

TensorParallelMLP::TensorParallelMLP(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int intermediate_size,
    int tp_degree)
    : hidden_dim_(hidden_dim),
      tp_degree_(tp_degree),
      gate_proj_(std::make_unique<ColumnParallelLayer>(ctx, hidden_dim, tp_degree)),
      up_proj_(std::make_unique<RowParallelLayer>(ctx, intermediate_size, tp_degree)),
      down_proj_(std::make_unique<RowParallelLayer>(ctx, hidden_dim, tp_degree)) {}

void TensorParallelMLP::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
    (void)input;
    (void)output;
    (void)batch;
    (void)seq;
    throw_unimplemented("TensorParallelMLP");
}

}  // namespace cuda::neural
