#include "cuda/algo/sort.h"

#include <cub/cub.cuh>
#include <cstdlib>
#include <algorithm>

#include "cuda/device/error.h"

namespace cuda::sort {

namespace detail {

template <typename Key>
__global__ void binary_search_kernel(const Key* sorted_data, size_t count, const Key target, size_t* result_index, int* found) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    size_t left = 0;
    size_t right = count;

    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (sorted_data[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    *result_index = left;
    *found = (left < count && sorted_data[left] == target) ? 1 : 0;
}

}  // namespace detail

template <typename Key, typename Value>
void radix_sort_pair(Key* keys, Value* values, size_t count, Order order, cudaStream_t stream) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, keys, values, values, count, 0, sizeof(Key) * 8, stream);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    if (order == Order::Descending) {
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, keys, keys, values, values, count, 0, sizeof(Key) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, keys, values, values, count, 0, sizeof(Key) * 8, stream);
    }

    CUDA_CHECK(cudaFree(d_temp_storage));
}

template <typename Key>
void radix_sort_keys(Key* keys, size_t count, Order order, cudaStream_t stream) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, keys, count, 0, sizeof(Key) * 8, stream);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    if (order == Order::Descending) {
        cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, keys, keys, count, 0, sizeof(Key) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, keys, count, 0, sizeof(Key) * 8, stream);
    }

    CUDA_CHECK(cudaFree(d_temp_storage));
}

template <typename Key, typename Value>
TopKResult<Key> select_top_k(const Key* keys, const Value* values, size_t count, size_t k, Order order, cudaStream_t stream) {
    const size_t actual_k = std::min(k, count);

    TopKResult<Key> result;
    result.keys = memory::Buffer<Key>(actual_k);
    result.values = memory::Buffer<Key>(actual_k);
    result.actual_k = actual_k;

    memory::Buffer<Key> keys_copy(count);
    memory::Buffer<size_t> indices(actual_k);
    memory::Buffer<Key> sorted_keys(count);

    CUDA_CHECK(cudaMemcpyAsync(keys_copy.data(), keys, count * sizeof(Key), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(sorted_keys.data(), keys_copy.data(), count * sizeof(Key), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.keys.data(), sorted_keys.data(), actual_k * sizeof(Key), cudaMemcpyDeviceToDevice, stream));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    if (order == Order::Descending) {
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, result.keys.data(), result.keys.data(), indices.data(), indices.data(), actual_k);
    } else {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, result.keys.data(), result.keys.data(), indices.data(), indices.data(), actual_k);
    }

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    if (order == Order::Descending) {
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, result.keys.data(), result.keys.data(), indices.data(), indices.data(), actual_k);
    } else {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, result.keys.data(), result.keys.data(), indices.data(), indices.data(), actual_k);
    }

    CUDA_CHECK(cudaFree(d_temp_storage));

    size_t* h_indices = static_cast<size_t*>(malloc(actual_k * sizeof(size_t)));
    Key* h_values = static_cast<Key*>(malloc(actual_k * sizeof(Key)));

    indices.copy_to(h_indices, actual_k);

    for (size_t i = 0; i < actual_k; ++i) {
        const size_t idx = h_indices[i];
        h_values[i] = (actual_k <= count && idx < count) ? static_cast<Key>(values[idx]) : Key{};
    }
    result.values.copy_from(h_values, actual_k);

    free(h_indices);
    free(h_values);

    return result;
}

template <typename T>
BinarySearchResult<T> binary_search(const T* sorted_data, size_t count, const T& target, cudaStream_t stream) {
    BinarySearchResult<T> result;

    size_t* d_result_index;
    int* d_found;

    CUDA_CHECK(cudaMalloc(&d_result_index, sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));

    detail::binary_search_kernel<<<1, 1, 0, stream>>>(sorted_data, count, target, d_result_index, d_found);

    size_t h_index;
    int h_found;

    CUDA_CHECK(cudaMemcpyAsync(&h_index, d_result_index, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    result.index = h_index;
    result.status = (h_index < count && sorted_data[h_index] == target) ? SearchResult::Found : SearchResult::NotFound;

    CUDA_CHECK(cudaFree(d_result_index));
    CUDA_CHECK(cudaFree(d_found));

    return result;
}

template void radix_sort_pair<float, int>(float*, int*, size_t, Order, cudaStream_t);
template void radix_sort_pair<double, int>(double*, int*, size_t, Order, cudaStream_t);
template void radix_sort_pair<int, float>(int*, float*, size_t, Order, cudaStream_t);
template void radix_sort_pair<int, double>(int*, double*, size_t, Order, cudaStream_t);

template void radix_sort_keys<float>(float*, size_t, Order, cudaStream_t);
template void radix_sort_keys<double>(double*, size_t, Order, cudaStream_t);
template void radix_sort_keys<int>(int*, size_t, Order, cudaStream_t);
template void radix_sort_keys<unsigned int>(unsigned int*, size_t, Order, cudaStream_t);

template TopKResult<float> select_top_k<float, float>(const float*, const float*, size_t, size_t, Order, cudaStream_t);
template TopKResult<double> select_top_k<double, double>(const double*, const double*, size_t, size_t, Order, cudaStream_t);
template TopKResult<int> select_top_k<int, int>(const int*, const int*, size_t, size_t, Order, cudaStream_t);

template BinarySearchResult<float> binary_search<float>(const float*, size_t, const float&, cudaStream_t);
template BinarySearchResult<double> binary_search<double>(const double*, size_t, const double&, cudaStream_t);
template BinarySearchResult<int> binary_search<int>(const int*, size_t, const int&, cudaStream_t);

}  // namespace cuda::sort
