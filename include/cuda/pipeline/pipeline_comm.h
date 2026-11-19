#pragma once

/**
 * @file pipeline_comm.h
 * @brief Pipeline communicator management
 *
 * Handles NCCL communicator splitting for hybrid TP+DP parallelism.
 */

#include "cuda/nccl/nccl_context.h"

#include <cuda_runtime.h>

#include <vector>

namespace cuda::pipeline {

    /**
     * @class PipelineCommunicators
     * @brief Manages NCCL communicators for hybrid parallelism
     *
     * Creates separate communicators for tensor parallelism (TP) and
     * data parallelism (DP) to enable hybrid TP+DP training.
     *
     * @example
     * @code
     * PipelineCommunicators comms(ctx, tp_degree=2, dp_degree=4);
     * ncclComm_t tp_comm = comms.get_tp_comm(tp_rank);
     * ncclComm_t dp_comm = comms.get_dp_comm(dp_rank);
     * @endcode
     */
    class PipelineCommunicators {
    public:
        /**
         * @brief Construct and split communicators
         * @param ctx NCCL context
         * @param tp_degree Tensor parallelism degree
         * @param dp_degree Data parallelism degree
         */
        PipelineCommunicators(::cuda::nccl::NcclContext& ctx, int tp_degree, int dp_degree);

        ~PipelineCommunicators();

        // Non-copyable
        PipelineCommunicators(const PipelineCommunicators&) = delete;
        PipelineCommunicators& operator=(const PipelineCommunicators&) = delete;

        /**
         * @brief Get TP communicator for a rank
         * @param tp_rank Rank within TP group
         * @return NCCL communicator
         */
        [[nodiscard]] ncclComm_t get_tp_comm(int tp_rank) const;

        /**
         * @brief Get DP communicator for a rank
         * @param dp_rank Rank within DP group
         * @return NCCL communicator
         */
        [[nodiscard]] ncclComm_t get_dp_comm(int dp_rank) const;

        /**
         * @brief Get total number of GPUs
         */
        [[nodiscard]] int total_devices() const;

        /**
         * @brief Get TP degree
         */
        [[nodiscard]] int tp_degree() const;

        /**
         * @brief Get DP degree
         */
        [[nodiscard]] int dp_degree() const;

        /**
         * @brief Check if communicator splitting is supported
         * @return true if ncclCommSplit is available
         */
        [[nodiscard]] bool supports_split() const;

    private:
        int tp_degree_;
        int dp_degree_;
        std::vector<ncclComm_t> tp_comms_;
        std::vector<ncclComm_t> dp_comms_;
        bool supports_split_;
    };

}  // namespace cuda::pipeline
