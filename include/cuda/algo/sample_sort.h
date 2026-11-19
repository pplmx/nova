#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::sample_sort {

    enum class Order { Ascending, Descending };

    template <typename T>
    struct SortResult {
        memory::Buffer<T> data;
        size_t actual_count;
    };

    template <typename T>
    SortResult<T> sort(const T* input, size_t count, Order order = Order::Ascending, cudaStream_t stream = nullptr);

    template <typename T>
    SortResult<T> sort_large_dataset(const T* input, size_t count, Order order = Order::Ascending, size_t sample_rate = 1024, cudaStream_t stream = nullptr);

    template <typename T>
    void sort_inplace(T* data, size_t count, Order order = Order::Ascending, cudaStream_t stream = nullptr);

    struct SampleSortConfig {
        size_t threshold_large_dataset = 1 << 20;
        size_t default_sample_rate = 1024;
        int max_blocks_per_pass = 64;
        bool adaptive_threshold = true;
    };

    void set_config(const SampleSortConfig& config);
    SampleSortConfig get_config();

}  // namespace cuda::algo::sample_sort
