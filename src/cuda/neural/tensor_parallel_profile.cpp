/**
 * @file tensor_parallel_profile.cpp
 * @brief Memory profiling implementation
 */

#include "cuda/neural/tensor_parallel_profile.h"
#include "cuda/device/error.h"

#include <algorithm>

namespace cuda::neural {

TensorParallelProfile TensorParallelProfiler::profile(
    int batch,
    int seq,
    int hidden_dim,
    int tp_degree) {

    constexpr size_t BYTES_PER_FLOAT = sizeof(float);

    size_t qkv_weight = static_cast<size_t>(hidden_dim) *
                       static_cast<size_t>(hidden_dim) * 3 *
                       BYTES_PER_FLOAT / tp_degree;

    size_t mlp_weight = static_cast<size_t>(hidden_dim) *
                       static_cast<size_t>(hidden_dim) * 8 *
                       BYTES_PER_FLOAT / tp_degree;

    size_t activation = static_cast<size_t>(batch) *
                       static_cast<size_t>(seq) *
                       static_cast<size_t>(hidden_dim) *
                       BYTES_PER_FLOAT;

    size_t gradient = activation * 2;

    size_t weight_shard = qkv_weight + mlp_weight;
    size_t total = weight_shard + activation + gradient;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int max_tp = std::min(tp_degree, device_count);

    return TensorParallelProfile{
        .weight_shard_bytes = weight_shard,
        .activation_bytes = activation,
        .gradient_bytes = gradient,
        .total_bytes = total,
        .max_tp_degree = max_tp
    };
}

int TensorParallelProfiler::recommend_tp_degree(
    int batch,
    int seq,
    int hidden_dim) {

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count <= 1) {
        return 1;
    }

    size_t total_memory = available_memory(0);
    constexpr size_t SAFETY_MARGIN = 2;
    size_t usable_memory = total_memory / SAFETY_MARGIN;

    for (int tp = device_count; tp >= 1; --tp) {
        auto prof = profile(batch, seq, hidden_dim, tp);
        if (prof.total_bytes <= usable_memory) {
            return tp;
        }
    }

    return 1;
}

size_t TensorParallelProfiler::available_memory(int device) {
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

size_t TensorParallelProfiler::peak_memory_usage() {
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total_mem));
    return total_mem;
}

int TensorParallelProfiler::max_tp_for_budget(
    int hidden_dim,
    size_t memory_budget_bytes) {

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    for (int tp = device_count; tp >= 1; --tp) {
        size_t required = estimate_memory_per_gpu(hidden_dim, tp);
        if (required <= memory_budget_bytes) {
            return tp;
        }
    }

    return 1;
}

}  // namespace cuda::neural
