#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo::segmented {

    /**
     * @brief Sort keys within each segment, gathering the permutation into out_*
     *
     * Device work is issued on `stream`: the uploads, the per-segment thrust sorts and
     * the gathers all run there. Segment boundaries come from `segment_ids` itself, so
     * `num_segments` is not used for anything and may be passed as the number of
     * distinct ids the caller believes it wrote.
     *
     * @note The call is host-blocking: segment boundaries are found by scanning
     *       segment_ids on the host, which requires a copy back from the device. The
     *       stream is honoured for ordering, not for a non-blocking return.
     */
    template <typename T>
    void sort_by_key(const T* keys, const int* segment_ids, T* out_keys, int* out_segments, size_t count, size_t num_segments, cudaStream_t stream = nullptr);

    template <typename T>
    void sort_by_key_inplace(T* keys, int* segment_ids, size_t count, size_t num_segments, cudaStream_t stream = nullptr);

    template <typename Key, typename Value>
    void sort_pairs_by_key(const Key* keys,
                           const Value* values,
                           const int* segment_ids,
                           Key* out_keys,
                           Value* out_values,
                           int* out_segments,
                           size_t count,
                           size_t num_segments,
                           cudaStream_t stream = nullptr);

    struct SegmentedSortConfig {
        size_t max_segments_per_block = 4;
        size_t elements_per_segment_block = 256;
        bool stable = true;
    };

    void set_config(const SegmentedSortConfig& config);
    SegmentedSortConfig get_config();

}  // namespace cuda::algo::segmented
