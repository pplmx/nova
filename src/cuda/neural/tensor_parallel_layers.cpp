/**
 * @file tensor_parallel_layers.cpp
 * @brief Weight-managed tensor-parallel layer implementations
 *
 * Milestone v2.21 (TASK-010 / issue-v20-tp-layers-stubs): these wrappers are
 * no longer fail-fast stubs (v2.20 DEC-006 disposition). Each layer owns the
 * rank's shard of the model weight on device:
 *
 * - set_weight() uploads the full weight and slices this rank's shard using
 *   the NCCL group rank resolved through ctx_.rank_of_device() (the v2.20
 *   verified convention — raw device indices are wrong for non-default groups
 *   and would slice the wrong shard).
 * - ColumnParallelLayer::forward is a local shard matmul (sharded output, no
 *   communication); RowParallelLayer::forward adds a single block-wise
 *   AllReduce (replicated output); TensorParallelMLP composes them around the
 *   SiLU gated activation.
 *
 * Every GEMM is issued on the caller-visible op stream through a per-instance
 * cuBLAS handle (the shared handle is not thread-safe across the rank threads
 * — same rationale as TensorParallelMatmul), and the multi-GPU AllReduce is
 * ordered on that stream so the partial then the reduction compose correctly.
 * With tp <= 1 the layers degenerate to a plain full-weight matmul (no NCCL).
 */

#include "cuda/neural/tensor_parallel_layers.h"

#include "cuda/device/error.h"
#include "cuda/neural/activations.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural {

namespace {

[[noreturn]] void throw_shape_error(const char* what) {
    throw std::invalid_argument(what);
}

[[noreturn]] void throw_no_weight(const char* layer) {
    throw std::runtime_error(
        std::string(layer) +
        "::forward called before set_weight: the layer owns no weight shard "
        "yet (upload one via set_weight).");
}

// The calling thread's NCCL rank within ctx's group (the thread is pinned to
// a device by the caller). Validates membership and throws NcclException when
// the active device is not in the group.
int active_rank(::cuda::nccl::NcclContext& ctx) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return ctx.rank_of_device(device);
}

// RAII for the per-forward op stream (forward can throw on shape errors /
// failed collectives and the stream must not leak).
struct StreamGuard {
    cudaStream_t s;
    ~StreamGuard() {
        if (s != nullptr) {
            cudaStreamDestroy(s);
        }
    }
};

}  // namespace

// ============================================================================
// ColumnParallelLayer
// ============================================================================

ColumnParallelLayer::ColumnParallelLayer(
    ::cuda::nccl::NcclContext& ctx,
    int in_features,
    int out_features)
    : ctx_(ctx), in_features_(in_features), out_features_(out_features) {
    const int tp = tp_degree();
    if (in_features_ <= 0) {
        throw_shape_error("ColumnParallelLayer requires a positive in_features");
    }
    if (out_features_ <= 0) {
        throw_shape_error("ColumnParallelLayer requires a positive out_features");
    }
    if (tp > 1 && out_features_ % tp != 0) {
        throw_shape_error(
            "ColumnParallelLayer requires out_features divisible by the tp "
            "degree");
    }
}

ColumnParallelLayer::~ColumnParallelLayer() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

cublasHandle_t ColumnParallelLayer::ensure_handle() {
    if (!handle_) {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    return handle_;
}

void ColumnParallelLayer::set_weight(const float* weight) {
    const int tp = tp_degree();
    if (tp <= 1) {
        // Single shard: the full weight is the shard.
        const size_t elems = static_cast<size_t>(in_features_) * out_features_;
        if (!weight_) {
            weight_ = std::make_unique<cuda::memory::Buffer<float>>(elems);
        }
        weight_->copy_from(weight, elems);
        return;
    }
    const int n_local = out_features_ / tp;
    const int rank = active_rank(ctx_);
    std::vector<float> shard(static_cast<size_t>(in_features_) * n_local);
    for (int i = 0; i < in_features_; ++i) {
        for (int j = 0; j < n_local; ++j) {
            shard[static_cast<size_t>(i) * n_local + j] =
                weight[static_cast<size_t>(i) * out_features_ +
                       rank * n_local + j];
        }
    }
    if (!weight_) {
        weight_ = std::make_unique<cuda::memory::Buffer<float>>(shard.size());
    }
    weight_->copy_from(shard.data(), shard.size());
}

void ColumnParallelLayer::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
    const int m = batch * seq;
    if (m <= 0) {
        throw_shape_error("ColumnParallelLayer::forward requires batch*seq > 0");
    }
    if (!weight_) {
        throw_no_weight("ColumnParallelLayer");
    }
    const int tp = tp_degree();
    const int n_local = tp <= 1 ? out_features_ : out_features_ / tp;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    StreamGuard guard{stream};

    cublasHandle_t handle = ensure_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    MatmulOptions opts;
    opts.handle = handle;
    // output[m x n_local] = input[m x in] @ weight_[in x n_local].
    cuda::neural::matmul(input, weight_->data(), output,
                         m, n_local, in_features_, opts);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int ColumnParallelLayer::in_features() const { return in_features_; }

int ColumnParallelLayer::out_features() const { return out_features_; }

int ColumnParallelLayer::tp_degree() const {
    return std::max(1, ctx_.device_count());
}

// ============================================================================
// RowParallelLayer
// ============================================================================

RowParallelLayer::RowParallelLayer(
    ::cuda::nccl::NcclContext& ctx,
    int in_features,
    int out_features)
    : ctx_(ctx),
      in_features_(in_features),
      out_features_(out_features),
      reducer_(ctx) {
    const int tp = tp_degree();
    if (in_features_ <= 0) {
        throw_shape_error("RowParallelLayer requires a positive in_features");
    }
    if (out_features_ <= 0) {
        throw_shape_error("RowParallelLayer requires a positive out_features");
    }
    if (tp > 1 && in_features_ % tp != 0) {
        throw_shape_error(
            "RowParallelLayer requires in_features divisible by the tp degree");
    }
}

RowParallelLayer::~RowParallelLayer() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

cublasHandle_t RowParallelLayer::ensure_handle() {
    if (!handle_) {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    return handle_;
}

void RowParallelLayer::set_weight(const float* weight) {
    const int tp = tp_degree();
    if (tp <= 1) {
        const size_t elems = static_cast<size_t>(in_features_) * out_features_;
        if (!weight_) {
            weight_ = std::make_unique<cuda::memory::Buffer<float>>(elems);
        }
        weight_->copy_from(weight, elems);
        return;
    }
    const int k_local = in_features_ / tp;
    const int rank = active_rank(ctx_);
    std::vector<float> shard(static_cast<size_t>(k_local) * out_features_);
    for (int i = 0; i < k_local; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            shard[static_cast<size_t>(i) * out_features_ + j] =
                weight[static_cast<size_t>(rank * k_local + i) * out_features_ +
                       j];
        }
    }
    if (!weight_) {
        weight_ = std::make_unique<cuda::memory::Buffer<float>>(shard.size());
    }
    weight_->copy_from(shard.data(), shard.size());
}

void RowParallelLayer::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
    const int m = batch * seq;
    if (m <= 0) {
        throw_shape_error("RowParallelLayer::forward requires batch*seq > 0");
    }
    if (!weight_) {
        throw_no_weight("RowParallelLayer");
    }
    const int tp = tp_degree();
    const int k_local = tp <= 1 ? in_features_ : in_features_ / tp;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    StreamGuard guard{stream};

    cublasHandle_t handle = ensure_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    MatmulOptions opts;
    opts.handle = handle;
    // partial[m x out] = input[m x k_local] @ weight_[k_local x out].
    cuda::neural::matmul(input, weight_->data(), output,
                         m, out_features_, k_local, opts);

    if (tp > 1) {
        // The replicated output is the block-wise AllReduce of every rank's
        // partial; without a live NCCL context no rank can reconstruct it.
        // Fail fast rather than silently returning a partial output.
        if (!ctx_.has_nccl()) {
            throw_shape_error(
                "RowParallelLayer::forward on multi-GPU requires an "
                "initialized NCCL context");
        }
        ::cuda::nccl::NcclResult result = reducer_.all_reduce_async(
            output, output, static_cast<size_t>(m) * out_features_,
            ncclFloat32, ncclSum, stream);
        if (!result.ok()) {
            throw ::cuda::nccl::NcclException(
                result.error_message.c_str(), result.code,
                "RowParallelLayer::forward AllReduce", __FILE__, __LINE__);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int RowParallelLayer::in_features() const { return in_features_; }

int RowParallelLayer::out_features() const { return out_features_; }

int RowParallelLayer::tp_degree() const {
    return std::max(1, ctx_.device_count());
}

// ============================================================================
// TensorParallelMLP
// ============================================================================

TensorParallelMLP::TensorParallelMLP(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int intermediate_size)
    : hidden_dim_(hidden_dim),
      intermediate_size_(intermediate_size),
      gate_proj_(std::make_unique<ColumnParallelLayer>(
          ctx, hidden_dim, intermediate_size)),
      up_proj_(std::make_unique<ColumnParallelLayer>(
          ctx, hidden_dim, intermediate_size)),
      down_proj_(std::make_unique<RowParallelLayer>(
          ctx, intermediate_size, hidden_dim)) {}

TensorParallelMLP::~TensorParallelMLP() = default;

void TensorParallelMLP::ensure_scratch(int m) {
    const int tp = tp_degree();
    const int inter_local =
        tp <= 1 ? intermediate_size_ : intermediate_size_ / tp;
    const size_t need = static_cast<size_t>(m) * inter_local;
    if (gate_buf_ != nullptr &&
        static_cast<size_t>(scratch_m_) * inter_local >= need) {
        return;
    }
    // Reallocate when the batch grew past the current scratch capacity.
    gate_buf_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    up_buf_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    sub_buf_ = std::make_unique<cuda::memory::Buffer<float>>(need);
    scratch_m_ = m;
}

void TensorParallelMLP::set_weight(
    const float* gate_weight,
    const float* up_weight,
    const float* down_weight) {
    gate_proj_->set_weight(gate_weight);
    up_proj_->set_weight(up_weight);
    down_proj_->set_weight(down_weight);
}

void TensorParallelMLP::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
    const int m = batch * seq;
    if (m <= 0) {
        throw_shape_error("TensorParallelMLP::forward requires batch*seq > 0");
    }
    ensure_scratch(m);
    gate_proj_->forward(input, gate_buf_->data(), batch, seq);
    up_proj_->forward(input, up_buf_->data(), batch, seq);
    const int tp = tp_degree();
    const int inter_local =
        tp <= 1 ? intermediate_size_ : intermediate_size_ / tp;
    cuda::neural::silu_and_mul(
        gate_buf_->data(), up_buf_->data(), sub_buf_->data(), m * inter_local);
    down_proj_->forward(sub_buf_->data(), output, batch, seq);
}

int TensorParallelMLP::hidden_dim() const { return hidden_dim_; }

int TensorParallelMLP::intermediate_size() const { return intermediate_size_; }

int TensorParallelMLP::tp_degree() const {
    return std::max(1, gate_proj_ ? gate_proj_->tp_degree() : 1);
}

}  // namespace cuda::neural
