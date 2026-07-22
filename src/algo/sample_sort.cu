#include "cuda/algo/sample_sort.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include "cuda/memory/unique_ptr.h"

namespace cuda::algo::sample_sort {

static SampleSortConfig g_config;

void set_config(const SampleSortConfig& config) {
    g_config = config;
}

SampleSortConfig get_config() {
    return g_config;
}

template <typename T>
SortResult<T> sort(const T* input, size_t count, Order order, cudaStream_t stream) {
    SortResult<T> result;

    // RAII wrapper ensures d_output is freed even if any of the cuda calls below throw.
    cuda::memory::unique_ptr<T> d_output(count);

    CUDA_CHECK(cudaMemcpyAsync(d_output.get(), input, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));

    thrust::device_ptr<T> d_ptr(d_output.get());
    if (order == Order::Ascending) {
        thrust::sort(d_ptr, d_ptr + count);
    } else {
        thrust::sort(d_ptr, d_ptr + count, thrust::greater<T>());
    }

    result.data = cuda::memory::Buffer<T>(count);
    CUDA_CHECK(cudaMemcpy(result.data.data(), d_output.get(), count * sizeof(T),
                          cudaMemcpyDeviceToHost));
    result.actual_count = count;

    return result;
}

template <typename T>
SortResult<T> sort_large_dataset(const T* input, size_t count,
                                 Order order, size_t sample_rate,
                                 cudaStream_t stream) {
    if (count <= g_config.threshold_large_dataset) {
        return sort(input, count, order, stream);
    }

    SortResult<T> result;

    cuda::memory::unique_ptr<T> d_output(count);

    CUDA_CHECK(cudaMemcpyAsync(d_output.get(), input, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));

    size_t num_samples = count / sample_rate;

    thrust::device_ptr<T> d_ptr(d_output.get());
    if (order == Order::Ascending) {
        thrust::sort(d_ptr, d_ptr + count);
    } else {
        thrust::sort(d_ptr, d_ptr + count, thrust::greater<T>());
    }

    result.data = cuda::memory::Buffer<T>(count);
    CUDA_CHECK(cudaMemcpy(result.data.data(), d_output.get(), count * sizeof(T),
                          cudaMemcpyDeviceToHost));
    result.actual_count = count;

    return result;
}

template <typename T>
void sort_inplace(T* data, size_t count, Order order, cudaStream_t stream) {
    cuda::memory::unique_ptr<T> d_data(count);

    CUDA_CHECK(cudaMemcpyAsync(d_data.get(), data, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));

    thrust::device_ptr<T> d_ptr(d_data.get());
    if (order == Order::Ascending) {
        thrust::sort(d_ptr, d_ptr + count);
    } else {
        thrust::sort(d_ptr, d_ptr + count, thrust::greater<T>());
    }

    CUDA_CHECK(cudaMemcpy(data, d_data.get(), count * sizeof(T),
                          cudaMemcpyDeviceToHost));
}

template SortResult<int> sort<int>(const int*, size_t, Order, cudaStream_t);
template SortResult<float> sort<float>(const float*, size_t, Order, cudaStream_t);
template SortResult<double> sort<double>(const double*, size_t, Order, cudaStream_t);

template SortResult<int> sort_large_dataset<int>(const int*, size_t, Order, size_t, cudaStream_t);
template SortResult<float> sort_large_dataset<float>(const float*, size_t, Order, size_t, cudaStream_t);
template SortResult<double> sort_large_dataset<double>(const double*, size_t, Order, size_t, cudaStream_t);

template void sort_inplace<int>(int*, size_t, Order, cudaStream_t);
template void sort_inplace<float>(float*, size_t, Order, cudaStream_t);
template void sort_inplace<double>(double*, size_t, Order, cudaStream_t);

}  // namespace cuda::algo::sample_sort
