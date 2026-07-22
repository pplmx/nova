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

    if (actual_k == 0 || count == 0) {
        return result;
    }

    // Sort the full (keys, indices) pair array using CUB, then take the first K entries.
    // Working buffers:
    //   sorted_keys   : count Key elements (overwritten in-place by CUB)
    //   sorted_indices: count size_t elements (overwritten in-place by CUB)
    memory::Buffer<Key> sorted_keys(count);
    memory::Buffer<size_t> sorted_indices(count);

    // Copy input keys into the working buffer (in-place CUB sort needs the buffer to hold the input).
    CUDA_CHECK(cudaMemcpyAsync(
        sorted_keys.data(), keys, count * sizeof(Key), cudaMemcpyDeviceToDevice, stream));

    // Initialize indices to [0, 1, ..., count-1]. CUB's SortPairs needs the indices buffer
    // filled with 0..count-1 so that after sorting, sorted_indices[i] tells us the original
    // position of the i-th sorted key. Use raw malloc to avoid <vector> dependency in this TU.
    size_t* h_identity = static_cast<size_t*>(malloc(count * sizeof(size_t)));
    for (size_t i = 0; i < count; ++i) {
        h_identity[i] = i;
    }
    sorted_indices.copy_from(h_identity, count);
    free(h_identity);

    // First call: query required temp storage.
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (order == Order::Descending) {
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            sorted_keys.data(), sorted_keys.data(),
            sorted_indices.data(), sorted_indices.data(),
            count, 0, sizeof(Key) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            sorted_keys.data(), sorted_keys.data(),
            sorted_indices.data(), sorted_indices.data(),
            count, 0, sizeof(Key) * 8, stream);
    }

    if (temp_storage_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    // Second call: perform the sort.
    if (order == Order::Descending) {
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            sorted_keys.data(), sorted_keys.data(),
            sorted_indices.data(), sorted_indices.data(),
            count, 0, sizeof(Key) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            sorted_keys.data(), sorted_keys.data(),
            sorted_indices.data(), sorted_indices.data(),
            count, 0, sizeof(Key) * 8, stream);
    }

    // Take the first K sorted keys.
    CUDA_CHECK(cudaMemcpyAsync(
        result.keys.data(), sorted_keys.data(),
        actual_k * sizeof(Key), cudaMemcpyDeviceToDevice, stream));

    // Gather the corresponding values back to host, then copy to result buffer.
    // For very large K this becomes a host-device roundtrip; for correctness over a
    // small K (the common case) this is simple and reliable. A device-side gather
    // kernel could replace it later if perf matters.
    size_t* h_indices = static_cast<size_t*>(malloc(actual_k * sizeof(size_t)));
    sorted_indices.copy_to(h_indices, actual_k);

    // Copy the full values array once, then gather the K needed entries into h_values.
    // A device-side gather kernel could replace this host roundtrip later.
    Value* h_all_values = static_cast<Value*>(malloc(count * sizeof(Value)));
    CUDA_CHECK(cudaMemcpyAsync(
        h_all_values, values, count * sizeof(Value),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Key* h_values = static_cast<Key*>(malloc(actual_k * sizeof(Key)));
    for (size_t i = 0; i < actual_k; ++i) {
        const size_t idx = h_indices[i];
        h_values[i] = (idx < count) ? static_cast<Key>(h_all_values[idx]) : Key{};
    }
    result.values.copy_from(h_values, actual_k);

    free(h_indices);
    free(h_all_values);
    free(h_values);

    if (d_temp_storage != nullptr) {
        CUDA_CHECK(cudaFree(d_temp_storage));
    }

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

    if (h_found && h_index < count) {
        T found_value;
        CUDA_CHECK(cudaMemcpy(&found_value, sorted_data + h_index, sizeof(T), cudaMemcpyDeviceToHost));
        result.status = (found_value == target) ? SearchResult::Found : SearchResult::NotFound;
    } else {
        result.status = SearchResult::NotFound;
    }

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
