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
#include "cuda/neural/optimizers/optimizers.h"
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
     * @brief Backward pass (synchronous; v2.23)
     *
     * The input is replicated, so the grad-input is the AllReduce (across
     * ranks) of the per-rank dY W^T — the new never-run collective for the
     * parallel stack. The grad-weight is the rank's column shard slice.
     *
     * @param input Forward input [batch*seq x in_features] (replicated)
     * @param grad_output Upstream gradient [batch*seq x out_features/tp]
     * @param grad_input Full grad-input [batch*seq x in_features] (replicated,
     *        AllReduced across ranks)
     * @param grad_weight Full grad-weight [in_features x out_features] row-major;
     *        this rank writes its column-shard slice [.., rank*out/tp ..)
     * @param batch Batch size
     * @param seq Sequence length
     * @throws std::invalid_argument when tp > 1 without an initialized NCCL
     *        context; throws NcclException when the grad-input AllReduce fails
     */
    void backward(
        const float* input,
        const float* grad_output,
        float* grad_input,
        float* grad_weight,
        int batch,
        int seq);

    /**
     * @brief Apply one optimizer step to this rank's weight shard (v2.24)
     *
     * The end-to-end training surface: extracts this rank's column-slice of
     * the caller's full grad_weight buffer (reverse of the backward scatter)
     * and runs one AdamW step in place on the private rank shard. With the
     * meshed gradient layout this shard step tiles to the full-weight update,
     * so a parallel training run matches a single-GPU full-weight reference.
     *
     * @param optimizer AdamW optimizer (per-rank instance; moments are local)
     * @param grad_weight Full grad-weight [in_features x out_features]
     *        row-major, as produced by backward()
     * @param step_no Optimizer step counter (AdamW bias correction uses it)
     * @param stream CUDA stream
     */
    void step(
        ::cuda::neural::optimizers::AdamWOptimizer& optimizer,
        const float* grad_weight,
        int step_no,
        cudaStream_t stream = nullptr);

    /**
     * @brief Copy this rank's weight shard to host memory (v2.24)
     *
     * Reads back the shard in its storage layout (the rank's column slice of
     * the full [in_features x out_features] weight; the full weight when tp
     * is 1), mirroring set_weight(). Lets a training-step test compare the
     * stepped shard against a host full-weight reference.
     *
     * @param host_out Host buffer of in_features * (out_features / tp)
     *        elements
     */
    void copy_weight_shard(float* host_out) const;

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
    // Used by backward to AllReduce the grad-input (replicated input => summed
    // per-rank dY W^T contributions).
    ::cuda::nccl::NcclAllReduce reducer_;
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
     * @throws std::invalid_argument when tp > 1 without an initialized NCCL
     *        context (the replicated output cannot be reconstructed); throws
     *        NcclException when the multi-GPU AllReduce itself fails
     */
    void forward(const float* input, float* output, int batch, int seq);

    /**
     * @brief Backward pass (synchronous; v2.23)
     *
     * The input is sharded (matches the upstream column-parallel layout), so
     * the grad-input is local (each rank's dY W_r^T is exactly its block of
     * the full grad). The grad-weight is the rank's row shard slice.
     *
     * @param input Forward input [batch*seq x in_features/tp] (sharded)
     * @param grad_output Upstream gradient [batch*seq x out_features]
     * @param grad_input Grad-input shard [batch*seq x in_features/tp] (local)
     * @param grad_weight Full grad-weight [in_features x out_features] row-major;
     *        this rank writes its row-shard slice [rank*in/tp .., ..)
     * @param batch Batch size
     * @param seq Sequence length
     */
    void backward(
        const float* input,
        const float* grad_output,
        float* grad_input,
        float* grad_weight,
        int batch,
        int seq);

    /**
     * @brief Apply one optimizer step to this rank's weight shard (v2.24)
     *
     * End-to-end training surface for the row-parallel layer: extracts this
     * rank's contiguous row-block of the caller's full grad_weight buffer and
     * runs one AdamW step in place on the private rank shard. The shard step
     * tiles to the full-weight update (see the class docs / DEC-010).
     *
     * @param optimizer AdamW optimizer (per-rank instance)
     * @param grad_weight Full grad-weight [in_features x out_features]
     *        row-major, as produced by backward()
     * @param step_no Optimizer step counter
     * @param stream CUDA stream
     */
    void step(
        ::cuda::neural::optimizers::AdamWOptimizer& optimizer,
        const float* grad_weight,
        int step_no,
        cudaStream_t stream = nullptr);

    /**
     * @brief Copy this rank's weight shard to host memory (v2.24)
     *
     * Reads back the shard in its storage layout (the rank's row block of the
     * full [in_features x out_features] weight; the full weight when tp is 1),
     * mirroring set_weight().
     *
     * @param host_out Host buffer of (in_features / tp) * out_features
     *        elements
     */
    void copy_weight_shard(float* host_out) const;

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
     * @brief Backward pass (synchronous; v2.23)
     *
     * Chains: down row-parallel backward (grad-sub sharded, grad-down-weight
     * row slice) -> silu_and_mul backward (grad-gate / grad-up on the sharded
     * intermediate) -> gate/up column-parallel backward, whose grad-inputs are
     * summed and AllReduced to form the full replicated grad-input.
     *
     * @param input Forward input [batch*seq x hidden_dim] (replicated)
     * @param grad_output Upstream gradient [batch*seq x hidden_dim]
     * @param grad_input Full grad-input [batch*seq x hidden_dim] (replicated)
     * @param grad_gate_weight Full grad-weight [hidden x intermediate] (rank
     *        writes its column shard slice)
     * @param grad_up_weight Full grad-weight [hidden x intermediate]
     * @param grad_down_weight Full grad-weight [intermediate x hidden] (rank
     *        writes its row shard slice)
     * @param batch Batch size
     * @param seq Sequence length
     */
    void backward(
        const float* input,
        const float* grad_output,
        float* grad_input,
        float* grad_gate_weight,
        float* grad_up_weight,
        float* grad_down_weight,
        int batch,
        int seq);

    /**
     * @brief Apply one optimizer step to gate/up/down weight shards (v2.24)
     *
     * Chains step() over the three projections: the gate/up column-parallel
     * shards and the down row-parallel shard each step in place from their
     * slice of the caller's full grad-weight buffers (produced by backward()).
     * This is the MLP-level end-to-end training surface — forward() -> loss
     * backward (cross_entropy_logits_backward) -> backward() -> step() is a
     * complete training step on the tensor-parallel stack.
     *
     * One AdamW instance per weight tensor: the optimizer's m/v moment buffers
     * are sized and keyed per element of a single weight, so gate/up/down must
     * each get their own instance (a shared instance would silently corrupt
     * the up/down moments with the gate history — cpp-review finding).
     *
     * @param gate_optimizer AdamW for the gate shard [hidden x inter/tp]
     * @param up_optimizer AdamW for the up shard [hidden x inter/tp]
     * @param down_optimizer AdamW for the down shard [inter/tp x hidden]
     * @param grad_gate Full gate grad [hidden x intermediate]
     * @param grad_up Full up grad [hidden x intermediate]
     * @param grad_down Full down grad [intermediate x hidden]
     * @param step_no Optimizer step counter (same for all three)
     * @param stream CUDA stream
     */
    void step(
        ::cuda::neural::optimizers::AdamWOptimizer& gate_optimizer,
        ::cuda::neural::optimizers::AdamWOptimizer& up_optimizer,
        ::cuda::neural::optimizers::AdamWOptimizer& down_optimizer,
        const float* grad_gate,
        const float* grad_up,
        const float* grad_down,
        int step_no,
        cudaStream_t stream = nullptr);

    /**
     * @brief Copy this rank's gate/up/down weight shards to host memory
     *        (v2.24)
     *
     * Reads back the three rank-local shards in their storage layouts (the
     * column-parallel gate/up column slices and the row-parallel down row
     * block), mirroring set_weight(). Lets a training-step test assemble the
     * per-rank shards and compare against the host full-weight reference.
     *
     * @param host_gate Host buffer of hidden * (intermediate / tp)
     * @param host_up Host buffer of hidden * (intermediate / tp)
     * @param host_down Host buffer of (intermediate / tp) * hidden
     */
    void copy_weights(float* host_gate, float* host_up, float* host_down) const;

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
