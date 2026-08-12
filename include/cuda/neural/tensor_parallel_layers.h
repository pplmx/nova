#pragma once

/**
 * @file tensor_parallel_layers.h
 * @brief Weight-managed tensor-parallel layer patterns for transformers
 *
 * Implements ColumnParallelLayer / RowParallelLayer / TensorParallelMLP used
 * in transformer architectures. Unlike the v1.3 scaffolding (which owned no
 * weight storage and was disposed as fail-fast stubs in v2.20, DEC-006), each
 * layer owns the rank's shard of the model weights on device, set_weight()
 * uploads/slices the full weight, and forward() computes the documented
 * sharded/replicated outputs on real multi-GPU (milestone v2.21, TASK-010).
 *
 * Semantics (Megatron-style weight sharding, matching the layer docs):
 * - ColumnParallelLayer splits W along the output dim; input replicated,
 *   output sharded (each rank writes its column block, no communication).
 * - RowParallelLayer splits W along the input dim; input sharded, output
 *   replicated (per-rank partial + one block-wise AllReduce).
 * - TensorParallelMLP = SiLU-gated FFN: gate/up column-parallel projections,
 *   silu_and_mul, then a row-parallel down projection back to hidden width.
 *
 * Rank membership is resolved through NcclContext::rank_of_device (the v2.20
 * verified convention); when the context reports <= 1 device the layers
 * degenerate to a plain single-shard matmul on the full weight (no NCCL).
 */

#include "cuda/neural/matmul.h"
#include "cuda/nccl/nccl_all_reduce.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

#include <cublas_v2.h>

#include <cstddef>
#include <memory>

namespace cuda::neural {

/**
 * @class ColumnParallelLayer
 * @brief Column-parallel linear layer (weight sharded along the output dim)
 *
 * Rank r owns columns [r*out/tp, (r+1)*out/tp) of the weight. The input is
 * replicated; the output is sharded (rank r writes only its column block,
 * no communication) — used for the gated projections in the FFN block.
 *
 * Input:  [batch * seq x in_features]  (replicated)
 * Output: [batch * seq x out_features / tp_degree]  (sharded)
 */
class ColumnParallelLayer {
public:
    /**
     * @brief Construct a column-parallel layer
     * @param ctx NCCL context; when it reports > 1 device, rank membership is
     *        resolved via rank_of_device for weight slicing and the output
     *        dimension must divide evenly by the TP degree
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @throws std::invalid_argument if out_features % tp_degree != 0
     */
    ColumnParallelLayer(
        ::cuda::nccl::NcclContext& ctx,
        int in_features,
        int out_features);

    // Non-copyable/non-movable: owns a lazy stream-bound cuBLAS handle and
    // references the NcclContext (same rationale as TensorParallelMatmul).
    ColumnParallelLayer(const ColumnParallelLayer&) = delete;
    ColumnParallelLayer& operator=(const ColumnParallelLayer&) = delete;
    ColumnParallelLayer(ColumnParallelLayer&&) = delete;
    ColumnParallelLayer& operator=(ColumnParallelLayer&&) = delete;

    ~ColumnParallelLayer();

    /**
     * @brief Upload the full weight and store this rank's column shard
     * @param weight Full weight [in_features x out_features], row-major
     * @throws NcclException when the active device is not in the NCCL group
     *        (tp > 1 requires an initialized context to resolve the rank)
     */
    void set_weight(const float* weight);

    /**
     * @brief Forward pass (synchronous)
     * @param input Input tensor [batch * seq x in_features]
     * @param output Output tensor [batch * seq x out_features / tp_degree]
     * @param batch Batch size
     * @param seq Sequence length
     */
    void forward(const float* input, float* output, int batch, int seq);

    /**
     * @brief Input dimension
     */
    [[nodiscard]] int in_features() const;

    /**
     * @brief Full output dimension (each rank's shard is out_features / tp)
     */
    [[nodiscard]] int out_features() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    cublasHandle_t ensure_handle();
    cublasHandle_t handle_ = nullptr;
    ::cuda::nccl::NcclContext& ctx_;
    int in_features_ = 0;
    int out_features_ = 0;
    // Rank-local column shard [in_features x out_features/tp]; equals the full
    // weight when tp == 1.
    std::unique_ptr<cuda::memory::Buffer<float>> weight_;
};

/**
 * @class RowParallelLayer
 * @brief Row-parallel linear layer (weight sharded along the input dim)
 *
 * Rank r owns rows [r*in/tp, (r+1)*in/tp) of the weight. The input is the
 * sharded activation from the upstream column-parallel ops; each rank computes
 * its partial [m x out] locally then one block-wise AllReduce yields the
 * replicated full-width output. Used for the FFN output projection.
 *
 * Input:  [batch * seq x in_features / tp_degree]  (sharded)
 * Output: [batch * seq x out_features]  (replicated)
 */
class RowParallelLayer {
public:
    /**
     * @brief Construct a row-parallel layer
     * @param ctx NCCL context; when it reports > 1 device, rank membership is
     *        resolved via rank_of_device for weight slicing and the input
     *        dimension must divide evenly by the TP degree
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @throws std::invalid_argument if in_features % tp_degree != 0
     */
    RowParallelLayer(
        ::cuda::nccl::NcclContext& ctx,
        int in_features,
        int out_features);

    // Non-copyable/non-movable (own cuBLAS handle + NcclContext reference).
    RowParallelLayer(const RowParallelLayer&) = delete;
    RowParallelLayer& operator=(const RowParallelLayer&) = delete;
    RowParallelLayer(RowParallelLayer&&) = delete;
    RowParallelLayer& operator=(RowParallelLayer&&) = delete;

    ~RowParallelLayer();

    /**
     * @brief Upload the full weight and store this rank's row shard
     * @param weight Full weight [in_features x out_features], row-major
     * @throws NcclException when the active device is not in the NCCL group
     *        (tp > 1 requires an initialized context to resolve the rank)
     */
    void set_weight(const float* weight);

    /**
     * @brief Forward pass (synchronous; AllReduce on the op stream)
     * @param input Input tensor [batch * seq x in_features / tp_degree]
     * @param output Output tensor [batch * seq x out_features]
     * @param batch Batch size
     * @param seq Sequence length
     * @throws NcclException when the multi-GPU AllReduce fails / NCCL absent
     */
    void forward(const float* input, float* output, int batch, int seq);

    /**
     * @brief Input dimension
     */
    [[nodiscard]] int in_features() const;

    /**
     * @brief Output dimension (replicated)
     */
    [[nodiscard]] int out_features() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    cublasHandle_t ensure_handle();
    cublasHandle_t handle_ = nullptr;
    ::cuda::nccl::NcclContext& ctx_;
    int in_features_ = 0;
    int out_features_ = 0;
    // Rank-local row shard [in_features/tp x out_features]; equals the full
    // weight when tp == 1.
    std::unique_ptr<cuda::memory::Buffer<float>> weight_;
    ::cuda::nccl::NcclAllReduce reducer_;
};

/**
 * @class TensorParallelMLP
 * @brief SiLU-gated tensor-parallel MLP (FFN) block
 *
 * out = down( silu(gate(x)) .* up(x) ), with gate/up as column-parallel
 * projections (sharded intermediate) and down as a row-parallel projection
 * (replicated hidden-width output) — the standard transformer FFN layout.
 */
class TensorParallelMLP {
public:
    /**
     * @brief Construct a tensor-parallel MLP block
     * @param ctx NCCL context
     * @param hidden_dim Hidden (input/output) dimension
     * @param intermediate_size FFN intermediate dimension (must divide evenly
     *        by the TP degree for the sharded gate/up projections)
     */
    TensorParallelMLP(
        ::cuda::nccl::NcclContext& ctx,
        int hidden_dim,
        int intermediate_size);

    // Non-copyable/non-movable (owns the layer sub-stack).
    TensorParallelMLP(const TensorParallelMLP&) = delete;
    TensorParallelMLP& operator=(const TensorParallelMLP&) = delete;
    TensorParallelMLP(TensorParallelMLP&&) = delete;
    TensorParallelMLP& operator=(TensorParallelMLP&&) = delete;

    ~TensorParallelMLP();

    /**
     * @brief Upload the full gate/up/down weights
     * @param gate_weight [hidden_dim x intermediate_size] row-major
     * @param up_weight [hidden_dim x intermediate_size] row-major
     * @param down_weight [intermediate_size x hidden_dim] row-major
     */
    void set_weight(
        const float* gate_weight,
        const float* up_weight,
        const float* down_weight);

    /**
     * @brief Forward pass (synchronous)
     * @param input Input tensor [batch * seq x hidden_dim]
     * @param output Output tensor [batch * seq x hidden_dim]
     * @param batch Batch size
     * @param seq Sequence length
     */
    void forward(const float* input, float* output, int batch, int seq);

    /**
     * @brief Hidden (input/output) dimension
     */
    [[nodiscard]] int hidden_dim() const;

    /**
     * @brief FFN intermediate dimension
     */
    [[nodiscard]] int intermediate_size() const;

    /**
     * @brief TP degree (max(1, NCCL context device count))
     */
    [[nodiscard]] int tp_degree() const;

private:
    void ensure_scratch(int m);

    int hidden_dim_ = 0;
    int intermediate_size_ = 0;
    std::unique_ptr<ColumnParallelLayer> gate_proj_;  // hidden -> intermediate
    std::unique_ptr<ColumnParallelLayer> up_proj_;    // hidden -> intermediate
    std::unique_ptr<RowParallelLayer> down_proj_;     // intermediate -> hidden
    // Grows to hold the sharded intermediate activations for the current batch.
    std::unique_ptr<cuda::memory::Buffer<float>> gate_buf_;
    std::unique_ptr<cuda::memory::Buffer<float>> up_buf_;
    std::unique_ptr<cuda::memory::Buffer<float>> sub_buf_;
    int scratch_m_ = 0;
};

}  // namespace cuda::neural
