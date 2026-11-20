#pragma once

#include "cuda/memory/kv_cache_allocator.h"
#include "cuda/stream/stream.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuda::memory {

    struct StreamingCacheConfig {
        bool enable_prefetch = true;
        bool enable_async_eviction = true;
        int prefetch_ahead_blocks = 2;
        int eviction_batch_size = 4;
        int eviction_policy = 0;
    };

    struct PrefetchRequest {
        int64_t sequence_id;
        int num_tokens_ahead;
        bool completed = false;
    };

    class StreamingCacheManager {
    public:
        explicit StreamingCacheManager(KVCacheAllocator* allocator, const StreamingCacheConfig& config);

        StreamingCacheManager(const StreamingCacheManager&) = delete;
        StreamingCacheManager& operator=(const StreamingCacheManager&) = delete;
        StreamingCacheManager(StreamingCacheManager&&) = default;
        StreamingCacheManager& operator=(StreamingCacheManager&&) = default;

        void prefetch_async(int64_t sequence_id, int num_tokens_ahead, const stream::Stream& stream);

        void evict_importance_weighted(int num_blocks);

        bool should_evict_async() const;

        void update_importance(int64_t sequence_id, float importance);

        float get_importance(int64_t sequence_id) const;

        void sync_prefetch(const stream::Stream& stream);

        /// Number of prefetch requests queued but not yet executed.
        ///
        /// Zero after sync_prefetch(). Exposed so the queue can be observed: the only
        /// test of sync_prefetch() asserted that blocks still existed afterwards, which
        /// says nothing about whether the pending queue was drained.
        [[nodiscard]] size_t pending_prefetch_count() const;

    private:
        KVCacheAllocator* allocator_;
        StreamingCacheConfig config_;
        std::unordered_map<int64_t, float> importance_weights_;
        std::vector<PrefetchRequest> pending_prefetch_;
        mutable std::mutex prefetch_mutex_;
        std::atomic<uint64_t> prefetch_id_ {0};
    };

}  // namespace cuda::memory
