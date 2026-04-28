#pragma once

/**
 * @file sort.h
 * @brief GPU sorting algorithms: radix sort, top-k selection, and binary search
 * @author Nova CUDA Library
 * @version 2.3
 */

#include <cuda_runtime.h>
#include <cstddef>

#include "cuda/memory/buffer.h"

namespace cuda::sort {

enum class Order { Ascending, Descending };

template <typename Key>
struct KeyValuePair {
    Key* keys;
    void* values;
    size_t count;
};

template <typename T>
struct TopKResult {
    memory::Buffer<T> keys;
    memory::Buffer<T> values;
    size_t actual_k;
};

enum class SearchResult { Found, NotFound };

template <typename T>
struct BinarySearchResult {
    SearchResult status;
    size_t index;
};

template <typename Key, typename Value>
void radix_sort_pair(Key* keys, Value* values, size_t count, Order order, cudaStream_t stream = nullptr);

template <typename Key>
void radix_sort_keys(Key* keys, size_t count, Order order, cudaStream_t stream = nullptr);

template <typename Key, typename Value>
TopKResult<Key> select_top_k(const Key* keys, const Value* values, size_t count, size_t k, Order order, cudaStream_t stream = nullptr);

template <typename T>
BinarySearchResult<T> binary_search(const T* sorted_data, size_t count, const T& target, cudaStream_t stream = nullptr);

}  // namespace cuda::sort
