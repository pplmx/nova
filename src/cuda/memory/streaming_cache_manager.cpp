#include "cuda/memory/streaming_cache_manager.h"
#include "cuda/device/error.h"
#include <algorithm>

namespace cuda::memory {

// Constructor stores allocator and config references
// The allocator must outlive this manager as we don't own it
StreamingCacheManager::StreamingCacheManager(
    KVCacheAllocator* allocator,
    const StreamingCacheConfig& config
) : allocator_(allocator), config_(config) {}

// Queue a prefetch request for a sequence
// This is async - the request is queued and executed later during sync_prefetch()
// The purpose is to start KV cache allocation before the sequence actually needs it
void StreamingCacheManager::prefetch_async(
    int64_t sequence_id,
    int num_tokens_ahead,
    const stream::Stream& stream
) {
    // Prefetch is optional - skip if disabled in config
    if (!config_.enable_prefetch) {
        return;
    }

    // Calculate how many blocks to prefetch based on tokens ahead
    // +1 ensures we always prefetch at least one block
    const int prefetch_blocks = std::min(
        num_tokens_ahead / allocator_->get_block_size_tokens() + 1,
        config_.prefetch_ahead_blocks  // Cap at configured maximum
    );

    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    PrefetchRequest request{
        .sequence_id = sequence_id,
        .num_tokens_ahead = prefetch_blocks * allocator_->get_block_size_tokens(),
        .completed = false
    };
    pending_prefetch_.push_back(request);
}

// Evict blocks based on importance weights when configured
// Falls back to simple eviction if async eviction or importance weighting disabled
// Importance-weighted eviction removes lowest-importance sequences first
void StreamingCacheManager::evict_importance_weighted(int num_blocks) {
    // Check if importance-weighted eviction is enabled
    if (!config_.enable_async_eviction || config_.eviction_policy != 1) {
        allocator_->evict(num_blocks);
        return;
    }

    // Snapshot (seq_id, importance) under the lock so the sort + eviction loop below
    // doesn't race with concurrent update_importance() / sync_prefetch() calls.
    // importance_weights_ is the only field protected by prefetch_mutex_; touching it
    // outside the lock was the previous (racy) behavior.
    std::vector<std::pair<int64_t, float>> snapshot;
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        snapshot.reserve(importance_weights_.size());
        for (const auto& [seq_id, importance] : importance_weights_) {
            snapshot.emplace_back(seq_id, importance);
        }
    }

    // Sort by importance (ascending) - lowest importance evicted first.
    // Uses the snapshot, not the live map, so it is race-free.
    std::sort(snapshot.begin(), snapshot.end(),
        [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {
            return a.second < b.second;
        });

    // Evict sequences until we've freed enough blocks
    // Each sequence may occupy multiple blocks, so we track total evicted
    int evicted = 0;
    for (const auto& [seq_id, importance] : snapshot) {
        if (evicted >= num_blocks) break;

        auto blocks = allocator_->get_blocks(seq_id);
        if (!blocks.empty()) {
            evicted += static_cast<int>(blocks.size());
            allocator_->free(seq_id);
        }
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
    const float free_ratio = total_blocks > 0
        ? static_cast<float>(free_blocks) / total_blocks
        : 0.0f;

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

// Execute all pending prefetch requests on specified stream
// Called at synchronization points (e.g., between phases of generation)
// This is where prefetch requests actually become allocations
void StreamingCacheManager::sync_prefetch(const stream::Stream& stream) {
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

}  // namespace cuda::memory
