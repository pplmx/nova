#include "cuda/memory/streaming_cache_manager.h"
#include "cuda/device/error.h"
#include <algorithm>

namespace cuda::memory {

    namespace {

        /// The number of blocks an eviction call may touch. Zero is a legal no-op; a negative
        /// count is not a smaller request, it is a meaningless one, and the loop testing
        /// `evicted >= num_blocks` treated it as already satisfied -- so -8 freed nothing and
        /// reported success, exactly like a legitimate "nothing to do".
        void validate_block_count(int num_blocks, const char* fn) {
            if (num_blocks < 0) {
                throw std::invalid_argument(std::string(fn) + ": num_blocks must not be negative, got " + std::to_string(num_blocks));
            }
        }

        /// The two knobs that tell this manager how to evict, checked where the manager is
        /// built rather than at the first eviction. eviction_policy selects a strategy by
        /// number, so a value other than the two the code implements silently fell back to
        /// plain LRU while the caller believed a policy was in effect.
        void validate_streaming_config(const StreamingCacheConfig& config, const char* fn) {
            if (config.prefetch_ahead_blocks < 1) {
                throw std::invalid_argument(std::string(fn) + ": prefetch_ahead_blocks must be at least 1, got " + std::to_string(config.prefetch_ahead_blocks));
            }
            if (config.eviction_policy != 0 && config.eviction_policy != 1) {
                throw std::invalid_argument(std::string(fn) + ": eviction_policy must be 0 (LRU) or 1 (importance-weighted), got " + std::to_string(config.eviction_policy));
            }
        }

    }  // namespace

    // Constructor stores allocator and config references
    // The allocator must outlive this manager as we don't own it
    StreamingCacheManager::StreamingCacheManager(KVCacheAllocator* allocator, const StreamingCacheConfig& config)
        : allocator_(allocator),
          config_(config) {
        validate_streaming_config(config_, "StreamingCacheManager");

        // Every method dereferences allocator_ unconditionally, so a null one is not a
        // manager with no cache -- it is a crash on the first call. Reject it here where the
        // caller still has the stack to see which construction it was.
        if (allocator_ == nullptr) {
            throw std::invalid_argument("StreamingCacheManager: allocator must not be null");
        }
    }

    // Queue a prefetch request for a sequence
    // This is deferred - the request is queued and executed later during sync_prefetch(). The
    // purpose is to start KV cache allocation before the sequence actually needs it.
    //
    // Nothing is enqueued on the stream: this is a locked push_back onto an in-process vector. See
    // the warning on the declaration.
    void StreamingCacheManager::prefetch_async(int64_t sequence_id, int num_tokens_ahead, const stream::Stream& stream) {
        (void)stream;
        // Prefetch is optional - skip if disabled in config
        if (!config_.enable_prefetch) {
            return;
        }

        // Calculate how many blocks to prefetch based on tokens ahead
        // +1 ensures we always prefetch at least one block
        const int prefetch_blocks = std::min(num_tokens_ahead / allocator_->get_block_size_tokens() + 1,
                                             config_.prefetch_ahead_blocks  // Cap at configured maximum
        );

        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        PrefetchRequest request {.sequence_id = sequence_id, .num_tokens_ahead = prefetch_blocks * allocator_->get_block_size_tokens(), .completed = false};
        pending_prefetch_.push_back(request);
    }

    // Evict blocks based on importance weights when configured
    // Falls back to simple eviction if async eviction or importance weighting disabled
    // Importance-weighted eviction removes lowest-importance sequences first
    void StreamingCacheManager::evict_importance_weighted(int num_blocks) {
        validate_block_count(num_blocks, "StreamingCacheManager::evict_importance_weighted");

        // Check if importance-weighted eviction is enabled
        if (!config_.enable_async_eviction || config_.eviction_policy != 1) {
            allocator_->evict(num_blocks);
            return;
        }

        // The importance of each candidate is read under prefetch_mutex_ by get_importance(),
        // one sequence at a time, so the ranking below always sees a consistent value rather
        // than a map being rewritten around it.

        // Evict sequences until we've freed enough blocks.
        //
        // This loop used to walk only the sequences that had a weight recorded, and to free
        // each one ENTIRELY through allocator_->free(). Both halves of that were wrong:
        //
        //  - A sequence holding blocks but with no weight never entered the snapshot, so with
        //    no weights at all the call freed nothing and returned as if it had succeeded.
        //    Measured: 32 of 32 blocks allocated, evict_importance_weighted(8) left all 32 in
        //    place, while KVCacheAllocator::evict(8) freed 8 immediately from the same state.
        //  - Freeing a whole sequence to satisfy a single block overshot by its length.
        //    Measured with three sequences of four blocks: asking for 1 block freed 4, three
        //    times in a row.
        //
        // The weighted path therefore ranks the sequences that actually hold memory (an
        // unweighted one sorts as the documented default importance of 1.0) and frees whole
        // sequences, lowest importance first, until it has freed at least `num_blocks`. The
        // allocator's own eviction finishes the job if the recorded sequences cannot cover
        // the request.
        // get_sequence_ids() already excludes sequences holding no blocks, so every candidate
        // here can be freed for something.
        struct Candidate {
            int64_t sequence_id;
            float importance;
        };
        std::vector<Candidate> candidates;
        for (const int64_t seq_id : allocator_->get_sequence_ids()) {
            candidates.push_back(Candidate {.sequence_id = seq_id, .importance = get_importance(seq_id)});
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.importance != b.importance) {
                return a.importance < b.importance;
            }
            return a.sequence_id < b.sequence_id;  // deterministic order for equal weights
        });

        const int free_blocks_before = allocator_->get_num_free_blocks();
        for (const auto& candidate : candidates) {
            if (allocator_->get_num_free_blocks() - free_blocks_before >= num_blocks) {
                return;
            }
            allocator_->free(candidate.sequence_id);
        }

        // Weighting ran out of candidates, or the caller asked for more than they hold. Let
        // the allocator's own LRU do the rest so the request is honoured rather than
        // silently under-delivered.
        const int freed = allocator_->get_num_free_blocks() - free_blocks_before;
        if (freed < num_blocks) {
            allocator_->evict(num_blocks - freed);
        }
    }

    // Check if async eviction should run
    // Returns true when free block ratio drops below 10%
    // This threshold triggers proactive eviction before we run out of blocks
    bool StreamingCacheManager::should_evict_async() const {
        if (!config_.enable_async_eviction) {
            return false;
        }

        const int free_blocks = allocator_->get_num_free_blocks();
        const int total_blocks = allocator_->get_stats().total_blocks;
        const float free_ratio = total_blocks > 0 ? static_cast<float>(free_blocks) / total_blocks : 0.0f;

        // 10% threshold gives headroom for new allocations
        return free_ratio < 0.1f;
    }

    // Update importance weight for a sequence
    // Used by caller to score sequences based on predicted future access
    // Higher importance = less likely to be evicted
    void StreamingCacheManager::update_importance(int64_t sequence_id, float importance) {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        importance_weights_[sequence_id] = importance;
    }

    // Get importance weight for a sequence
    // Returns default 1.0f if sequence not tracked (no explicit importance set)
    float StreamingCacheManager::get_importance(int64_t sequence_id) const {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        auto it = importance_weights_.find(sequence_id);
        return it != importance_weights_.end() ? it->second : 1.0f;
    }

    // Execute all pending prefetch requests. Called at synchronization points (e.g. between phases
    // of generation); this is where prefetch requests actually become allocations.
    //
    // The stream argument is not used and cannot be: the work below is KVCacheAllocator::append(),
    // which allocates from a host-side free list and launches nothing. The old comment said "on
    // specified stream", which was never true. See the warning on the declaration.
    void StreamingCacheManager::sync_prefetch(const stream::Stream& stream) {
        (void)stream;
        std::lock_guard<std::mutex> lock(prefetch_mutex_);

        // Execute pending requests and mark as completed
        for (auto& request : pending_prefetch_) {
            if (!request.completed) {
                allocator_->append(request.sequence_id, request.num_tokens_ahead);
                request.completed = true;
            }
        }

        // Clear all requests after execution
        // Note: Could optimize by reusing vector capacity, but clear() is simpler
        pending_prefetch_.clear();
    }

    size_t StreamingCacheManager::pending_prefetch_count() const {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        return pending_prefetch_.size();
    }

}  // namespace cuda::memory
