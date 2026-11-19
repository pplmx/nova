#pragma once

/**
 * @file tensor_parallel_profile.h
 * @brief Memory profiling for tensor parallelism
 *
 * Provides utilities for estimating memory requirements
 * and recommending optimal TP degree based on available memory.
 */

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda::neural {

    /**
     * @struct TensorParallelProfile
     * @brief Memory profile for tensor-parallel operation
     */
    struct TensorParallelProfile {
        /** Weight shard size in bytes per GPU */
        size_t weight_shard_bytes;

        /** Activation buffer size in bytes per GPU */
        size_t activation_bytes;

        /** Gradient buffer size in bytes per GPU (for training) */
        size_t gradient_bytes;

        /** Total working set size in bytes per GPU */
        size_t total_bytes;

        /** Maximum recommended TP degree for this configuration */
        int max_tp_degree;
    };

    /**
     * @class TensorParallelProfiler
     * @brief Memory profiling utilities for tensor parallelism
     */
    class TensorParallelProfiler {
    public:
        /**
         * @brief Profile memory requirements for given configuration
         * @param batch Batch size
         * @param seq Sequence length
         * @param hidden_dim Hidden dimension (H)
         * @param tp_degree Tensor parallelism degree
         * @return Memory profile
         */
        [[nodiscard]]
        static TensorParallelProfile profile(int batch, int seq, int hidden_dim, int tp_degree);

        /**
         * @brief Recommend TP degree based on available memory
         * @param batch Batch size
         * @param seq Sequence length
         * @param hidden_dim Hidden dimension
         * @return Recommended TP degree
         */
        [[nodiscard]]
        static int recommend_tp_degree(int batch, int seq, int hidden_dim);

        /**
         * @brief Get available memory on device
         * @param device Device index
         * @return Available memory in bytes
         */
        [[nodiscard]]
        static size_t available_memory(int device);

        /**
         * @brief Get current device memory usage
         * @return Bytes currently allocated on the device (total - free)
         *
         * @note CUDA has no API for a true high-water mark, so this reports current
         *       usage. It previously returned the device's TOTAL capacity, because
         *       cudaMemGetInfo's out-parameters were mixed up.
         */
        [[nodiscard]]
        static size_t peak_memory_usage();

        /**
         * @brief Estimate memory for given hidden_dim and TP degree
         * @param hidden_dim Hidden dimension
         * @param tp_degree TP degree
         * @return Memory per GPU in bytes
         */
        [[nodiscard]]
        static constexpr size_t estimate_memory_per_gpu(int hidden_dim, int tp_degree) {
            constexpr size_t BYTES_PER_FLOAT = sizeof(float);

            size_t qkv_weight = static_cast<size_t>(hidden_dim) * static_cast<size_t>(hidden_dim) * 3 * BYTES_PER_FLOAT / tp_degree;

            size_t mlp_weight = static_cast<size_t>(hidden_dim) * static_cast<size_t>(hidden_dim) * 8 * BYTES_PER_FLOAT / tp_degree;

            return qkv_weight + mlp_weight;
        }

        /**
         * @brief Calculate maximum TP degree for given memory budget
         * @param hidden_dim Hidden dimension
         * @param memory_budget_bytes Available memory in bytes
         * @return Maximum TP degree that fits in budget
         */
        [[nodiscard]]
        static int max_tp_for_budget(int hidden_dim, size_t memory_budget_bytes);
    };

}  // namespace cuda::neural
