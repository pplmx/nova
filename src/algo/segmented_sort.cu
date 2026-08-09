#include "cuda/algo/segmented_sort.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/functional.h>

#include "cuda/memory/unique_ptr.h"
#include "cuda/device/error.h"

namespace cuda::algo::segmented {

static SegmentedSortConfig g_config;

void set_config(const SegmentedSortConfig& config) {
    g_config = config;
}

SegmentedSortConfig get_config() {
    return g_config;
}

template <typename T>
void sort_by_key(const T* keys, const int* segment_ids, T* out_keys, int* out_segments,
                 size_t count, size_t num_segments, cudaStream_t stream) {
    // RAII wrappers ensure the scratch buffers are freed even if any of the
    // cudaMemcpy calls below throw. The previous code used raw cudaMalloc
    // without CUDA_CHECK, so a partial failure leaked the earlier allocations.
    cuda::memory::unique_ptr<T> d_keys_sorted(count);
    cuda::memory::unique_ptr<T> d_keys_original(count);
    cuda::memory::unique_ptr<int> d_segments_original(count);
    cuda::memory::unique_ptr<int> d_indices(count);

    CUDA_CHECK(cudaMemcpy(d_keys_original.get(), keys, count * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segments_original.get(), segment_ids, count * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys_sorted.get(), keys, count * sizeof(T),
                          cudaMemcpyHostToDevice));

    thrust::device_ptr<int> d_indices_ptr(d_indices.get());
    thrust::sequence(d_indices_ptr, d_indices_ptr + count);

    thrust::device_ptr<T> d_keys_sorted_ptr(d_keys_sorted.get());

    if (g_config.stable) {
        thrust::stable_sort_by_key(d_keys_sorted_ptr, d_keys_sorted_ptr + count, d_indices_ptr);
    } else {
        thrust::sort_by_key(d_keys_sorted_ptr, d_keys_sorted_ptr + count, d_indices_ptr);
    }

    thrust::device_ptr<T> d_out_keys_ptr(out_keys);
    thrust::device_ptr<int> d_out_segments_ptr(out_segments);
    thrust::gather(d_indices_ptr, d_indices_ptr + count,
                   thrust::device_ptr<const T>(d_keys_original.get()), d_out_keys_ptr);
    thrust::gather(d_indices_ptr, d_indices_ptr + count,
                   thrust::device_ptr<const int>(d_segments_original.get()), d_out_segments_ptr);
}

template <typename T>
void sort_by_key_inplace(T* keys, int* segment_ids, size_t count,
                         size_t num_segments, cudaStream_t stream) {
    thrust::device_ptr<T> d_keys(keys);
    thrust::device_ptr<int> d_segments(segment_ids);

    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(d_keys, d_segments));
    auto keys_end = keys_begin + count;

    if (g_config.stable) {
        thrust::stable_sort_by_key(keys_begin, keys_end, d_keys);
    } else {
        thrust::sort_by_key(keys_begin, keys_end, d_keys);
    }
}

template void sort_by_key<float>(const float*, const int*, float*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<double>(const double*, const int*, double*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<int>(const int*, const int*, int*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<int64_t>(const int64_t*, const int*, int64_t*, int*, size_t, size_t, cudaStream_t);

template void sort_by_key_inplace<float>(float*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key_inplace<double>(double*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key_inplace<int>(int*, int*, size_t, size_t, cudaStream_t);

}  // namespace cuda::algo::segmented
