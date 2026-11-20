#include "cuda/algo/segmented_sort.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>
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
    void sort_by_key(const T* keys, const int* segment_ids, T* out_keys, int* out_segments, size_t count, size_t num_segments, cudaStream_t stream) {
        // RAII wrappers ensure the scratch buffers are freed even if any of the
        // cudaMemcpy calls below throw. The previous code used raw cudaMalloc
        // without CUDA_CHECK, so a partial failure leaked the earlier allocations.
        cuda::memory::unique_ptr<T> d_keys_sorted(count);
        cuda::memory::unique_ptr<T> d_keys_original(count);
        cuda::memory::unique_ptr<int> d_segments_original(count);
        cuda::memory::unique_ptr<int> d_indices(count);

        // Uploads go on the caller's stream, not the default one.
        CUDA_CHECK(cudaMemcpyAsync(d_keys_original.get(), keys, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_segments_original.get(), segment_ids, count * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_keys_sorted.get(), keys, count * sizeof(T), cudaMemcpyHostToDevice, stream));

        // Every thrust call below runs on the caller's stream. Before this they ran on
        // the default stream (no execution policy) while the API advertised a stream
        // parameter it discarded, so a caller's queued work was not ordered against the
        // sort.
        const auto policy = thrust::cuda::par.on(stream);

        thrust::device_ptr<int> d_indices_ptr(d_indices.get());
        thrust::sequence(policy, d_indices_ptr, d_indices_ptr + count);

        thrust::device_ptr<T> d_keys_sorted_ptr(d_keys_sorted.get());

        // Sort WITHIN each segment. The previous version sorted the whole buffer
        // globally and ignored segment_ids entirely, so elements moved across
        // segment boundaries: keys {5,1 | 4,2} came out as {1,2,4,5} instead of
        // {1,5 | 2,4}, and the gathered segment ids no longer lined up with the
        // elements they described.
        //
        // Thrust has no segmented sort, so the segments are sorted one at a time by
        // scanning the (already contiguous) segment ids. The host-side scan costs one
        // transfer, which is the price of correctness here.
        {
            // thrust pulls in cuda::std, whose <vector> is incomplete in this
            // translation unit; a plain array keeps the host-side scan independent
            // of the standard library configuration.
            int* host_segments = new int[count];
            // The segment scan is host-side, so it has to see the device data. Order the
            // copy on the caller's stream and wait for it, rather than issuing it on the
            // default stream where it would be unordered with respect to work the caller
            // queued on `stream`.
            CUDA_CHECK(cudaMemcpyAsync(host_segments, segment_ids, count * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            size_t begin = 0;
            while (begin < count) {
                // Find the end of the run sharing host_segments[begin].
                size_t end = begin + 1;
                while (end < count && host_segments[end] == host_segments[begin]) {
                    ++end;
                }

                if (end - begin > 1) {
                    auto seg_begin = d_keys_sorted_ptr + begin;
                    auto seg_end = d_keys_sorted_ptr + end;
                    auto idx_begin = d_indices_ptr + begin;
                    if (g_config.stable) {
                        thrust::stable_sort_by_key(policy, seg_begin, seg_end, idx_begin);
                    } else {
                        thrust::sort_by_key(policy, seg_begin, seg_end, idx_begin);
                    }
                }
                begin = end;
            }
            delete[] host_segments;
        }

        thrust::device_ptr<T> d_out_keys_ptr(out_keys);
        thrust::device_ptr<int> d_out_segments_ptr(out_segments);
        thrust::gather(policy, d_indices_ptr, d_indices_ptr + count, thrust::device_ptr<const T>(d_keys_original.get()), d_out_keys_ptr);
        thrust::gather(policy, d_indices_ptr, d_indices_ptr + count, thrust::device_ptr<const int>(d_segments_original.get()), d_out_segments_ptr);
    }

    template <typename Key, typename Value>
    void sort_pairs_by_key(
        const Key* keys, const Value* values, const int* segment_ids, Key* out_keys, Value* out_values, int* out_segments, size_t count, size_t num_segments, cudaStream_t stream) {
        // Declared in segmented_sort.h but never defined: any caller failed to link.
        // It sorts (key, value) pairs within each segment, mirroring sort_by_key but
        // carrying a payload alongside the key.
        if (count == 0) {
            return;
        }

        cuda::memory::unique_ptr<Key> d_keys(count);
        cuda::memory::unique_ptr<Value> d_values(count);
        cuda::memory::unique_ptr<int> d_segments(count);
        cuda::memory::unique_ptr<int> d_indices(count);
        cuda::memory::unique_ptr<Key> d_keys_original(count);
        cuda::memory::unique_ptr<Value> d_values_original(count);
        cuda::memory::unique_ptr<int> d_segments_original(count);

        CUDA_CHECK(cudaMemcpy(d_keys.get(), keys, count * sizeof(Key), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values.get(), values, count * sizeof(Value), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_segments.get(), segment_ids, count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_keys_original.get(), keys, count * sizeof(Key), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values_original.get(), values, count * sizeof(Value), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_segments_original.get(), segment_ids, count * sizeof(int), cudaMemcpyHostToDevice));

        thrust::device_ptr<int> indices_ptr(d_indices.get());
        thrust::device_ptr<Key> keys_ptr(d_keys.get());
        thrust::device_ptr<Value> values_ptr(d_values.get());

        int* host_segments = new int[count];
        CUDA_CHECK(cudaMemcpyAsync(host_segments, segment_ids, count * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto policy = thrust::cuda::par.on(stream);
        thrust::sequence(policy, indices_ptr, indices_ptr + count);

        size_t begin = 0;
        while (begin < count) {
            size_t end = begin + 1;
            while (end < count && host_segments[end] == host_segments[begin]) {
                ++end;
            }
            if (end - begin > 1) {
                auto seg_begin = keys_ptr + begin;
                auto seg_end = keys_ptr + end;
                auto value_begin = values_ptr + begin;
                auto idx_begin = indices_ptr + begin;
                if (g_config.stable) {
                    thrust::stable_sort_by_key(policy, seg_begin, seg_end, idx_begin);
                } else {
                    thrust::sort_by_key(policy, seg_begin, seg_end, idx_begin);
                }
                // Carry the payload through the same permutation.
                cuda::memory::unique_ptr<Value> sorted_values(end - begin);
                thrust::gather(policy, idx_begin, idx_begin + (end - begin), value_begin, thrust::device_ptr<Value>(sorted_values.get()));
                CUDA_CHECK(cudaMemcpyAsync(value_begin.get(), sorted_values.get(), (end - begin) * sizeof(Value), cudaMemcpyDeviceToDevice, stream));
            }
            begin = end;
        }
        delete[] host_segments;
        (void)num_segments;

        thrust::gather(policy, indices_ptr, indices_ptr + count, thrust::device_ptr<const Key>(d_keys_original.get()), thrust::device_ptr<Key>(out_keys));
        thrust::gather(policy, indices_ptr, indices_ptr + count, thrust::device_ptr<const Value>(d_values_original.get()), thrust::device_ptr<Value>(out_values));
        thrust::gather(policy, indices_ptr, indices_ptr + count, thrust::device_ptr<const int>(d_segments_original.get()), thrust::device_ptr<int>(out_segments));
    }

    template <typename T>
    void sort_by_key_inplace(T* keys, int* segment_ids, size_t count, size_t num_segments, cudaStream_t stream) {
        // stream is honoured: the segment scan is ordered on it and every thrust call
        // below runs through a policy bound to it. See the note in sort_by_key.
        if (count == 0) {
            return;
        }

        // Same defect as sort_by_key: this sorted the whole buffer globally. It also
        // sorted by a zipped (key, segment) tuple, so the sort order was the key
        // order within a segment and the segment id across segments — which is why
        // the segment ids appeared to stay put while the keys still crossed
        // boundaries.
        thrust::device_ptr<T> d_keys(keys);
        thrust::device_ptr<int> d_segments(segment_ids);

        int* host_segments = new int[count];
        // The scan is host-side, so it must wait for the device data -- on the caller's
        // stream, not the default one.
        CUDA_CHECK(cudaMemcpyAsync(host_segments, segment_ids, count * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto policy = thrust::cuda::par.on(stream);

        size_t begin = 0;
        while (begin < count) {
            size_t end = begin + 1;
            while (end < count && host_segments[end] == host_segments[begin]) {
                ++end;
            }
            if (end - begin > 1) {
                auto seg_begin = d_keys + begin;
                auto seg_end = d_keys + end;
                if (g_config.stable) {
                    thrust::stable_sort(policy, seg_begin, seg_end);
                } else {
                    thrust::sort(policy, seg_begin, seg_end);
                }
            }
            begin = end;
        }
        delete[] host_segments;
        (void)num_segments;
        (void)stream;
    }

    template void sort_by_key<float>(const float*, const int*, float*, int*, size_t, size_t, cudaStream_t);
    template void sort_by_key<double>(const double*, const int*, double*, int*, size_t, size_t, cudaStream_t);
    template void sort_by_key<int>(const int*, const int*, int*, int*, size_t, size_t, cudaStream_t);
    template void sort_by_key<int64_t>(const int64_t*, const int*, int64_t*, int*, size_t, size_t, cudaStream_t);

    template void sort_by_key_inplace<float>(float*, int*, size_t, size_t, cudaStream_t);
    template void sort_by_key_inplace<double>(double*, int*, size_t, size_t, cudaStream_t);
    template void sort_by_key_inplace<int>(int*, int*, size_t, size_t, cudaStream_t);

    template void sort_pairs_by_key<int, int>(const int*, const int*, const int*, int*, int*, int*, size_t, size_t, cudaStream_t);
    template void sort_pairs_by_key<float, int>(const float*, const int*, const int*, float*, int*, int*, size_t, size_t, cudaStream_t);
    template void sort_pairs_by_key<int, float>(const int*, const float*, const int*, int*, float*, int*, size_t, size_t, cudaStream_t);

}  // namespace cuda::algo::segmented
