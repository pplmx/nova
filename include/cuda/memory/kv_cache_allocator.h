#pragma once

#include "cuda/memory/buffer.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace cuda::memory {

struct KVCacheBlock {
    void* data = nullptr;
    int block_id = -1;
    int num_tokens = 0;
    int64_t sequence_id = -1;
    uint64_t last_access = 0;
    KVCacheBlock* prev = nullptr;
    KVCacheBlock* next = nullptr;
    bool in_use = false;
};

struct KVCacheAllocatorConfig {
    int num_heads = 32;
    int head_dim = 128;
    int block_size_tokens = 16;
    int num_blocks = 4096;
    int num_layers = 32;
    int eviction_threshold_pct = 10;
    bool enable_prefix_caching = true;
    int max_prefix_blocks = 256;
};

struct KVCacheStats {
    int total_blocks = 0;
    int allocated_blocks = 0;
    int free_blocks = 0;
    float fragmentation_percent = 0.0f;
    size_t total_memory = 0;
    size_t used_memory = 0;
    int prefix_cache_hits = 0;
    int prefix_cache_misses = 0;
    int evictions = 0;
    int allocation_requests = 0;
    int allocation_failures = 0;
};

class KVCacheAllocator {
public:
    explicit KVCacheAllocator(const KVCacheAllocatorConfig& config);
    ~KVCacheAllocator();

    KVCacheAllocator(const KVCacheAllocator&) = delete;
    KVCacheAllocator& operator=(const KVCacheAllocator&) = delete;
    KVCacheAllocator(KVCacheAllocator&&) = default;
    KVCacheAllocator& operator=(KVCacheAllocator&&) = default;

    std::vector<KVCacheBlock*> allocate(int64_t sequence_id, int num_tokens);
    std::vector<KVCacheBlock*> append(int64_t sequence_id, int num_tokens);
    void free(int64_t sequence_id);

    void evict(int num_blocks_needed);
    std::vector<KVCacheBlock*> get_blocks(int64_t sequence_id) const;
    KVCacheBlock* get_block(int64_t sequence_id, int block_index) const;

    struct PrefixMatch {
        int64_t sequence_id;
        int num_matching_tokens;
        int first_block_index;
    };

    std::optional<PrefixMatch> find_prefix_match(
        const void* prefix_tokens,
        int prefix_length
    ) const;

    KVCacheStats get_stats() const;
    void reset_stats();

    int get_num_free_blocks() const { return static_cast<int>(free_list_.size()); }
    int get_block_size_tokens() const { return config_.block_size_tokens; }

    void* get_gpu_memory() const { return gpu_memory_.data(); }
    size_t get_gpu_memory_size() const { return gpu_memory_.size(); }

private:
    void allocate_blocks_internal(int num_blocks);
    void free_block_internal(int block_idx);
    void update_lru(int64_t sequence_id);
    uint64_t compute_prefix_hash(const void* tokens, int num_tokens) const;
    void add_to_sequence(int64_t sequence_id, int block_idx);
    void remove_from_sequence(int64_t sequence_id);
    int find_oldest_sequence() const;

    KVCacheAllocatorConfig config_;
    std::vector<KVCacheBlock> blocks_;
    std::vector<int> free_list_;
    std::unordered_map<int64_t, std::vector<int>> sequence_blocks_;
    std::unordered_map<uint64_t, int> prefix_cache_;
    mutable std::shared_mutex mutex_;
    KVCacheStats stats_;

    memory::Buffer<void> gpu_memory_;

    size_t block_memory_size_ = 0;
    std::atomic<uint64_t> access_counter_{0};
};

KVCacheAllocator::KVCacheAllocator(const KVCacheAllocatorConfig& config)
    : config_(config) {

    if (config_.block_size_tokens != 16 && config_.block_size_tokens != 32 &&
        config_.block_size_tokens != 64) {
        throw std::invalid_argument(
            "block_size_tokens must be 16, 32, or 64");
    }

    block_memory_size_ = static_cast<size_t>(config_.num_layers) *
                         config_.num_heads * config_.head_dim *
                         config_.block_size_tokens * sizeof(float) * 2;

    const size_t total_memory = block_memory_size_ * config_.num_blocks;

    gpu_memory_ = memory::Buffer<void>(total_memory);

    blocks_.resize(config_.num_blocks);

    char* base_ptr = static_cast<char*>(gpu_memory_.data());
    for (int i = 0; i < config_.num_blocks; ++i) {
        blocks_[i].block_id = i;
        blocks_[i].data = base_ptr + i * block_memory_size_;
        blocks_[i].num_tokens = config_.block_size_tokens;
        blocks_[i].in_use = false;
        blocks_[i].sequence_id = -1;
        free_list_.push_back(i);
    }

    stats_.total_blocks = config_.num_blocks;
    stats_.free_blocks = config_.num_blocks;
    stats_.total_memory = total_memory;
}

KVCacheAllocator::~KVCacheAllocator() = default;

std::vector<KVCacheBlock*> KVCacheAllocator::allocate(
    int64_t sequence_id,
    int num_tokens
) {
    std::unique_lock lock(mutex_);

    const int num_blocks_needed = (num_tokens + config_.block_size_tokens - 1) /
                                  config_.block_size_tokens;

    if (static_cast<int>(free_list_.size()) < num_blocks_needed) {
        const int need = num_blocks_needed - free_list_.size();
        evict(need);
    }

    if (static_cast<int>(free_list_.size()) < num_blocks_needed) {
        stats_.allocation_failures++;
        return {};
    }

    std::vector<KVCacheBlock*> result;
    result.reserve(num_blocks_needed);

    for (int i = 0; i < num_blocks_needed; ++i) {
        const int block_idx = free_list_.back();
        free_list_.pop_back();

        KVCacheBlock& block = blocks_[block_idx];
        block.in_use = true;
        block.sequence_id = sequence_id;
        block.last_access = ++access_counter_;
        block.prev = nullptr;
        block.next = nullptr;

        result.push_back(&block);
        sequence_blocks_[sequence_id].push_back(block_idx);

        stats_.allocated_blocks++;
        stats_.free_blocks--;
    }

    stats_.allocation_requests++;
    stats_.used_memory = stats_.allocated_blocks * block_memory_size_;

    return result;
}

std::vector<KVCacheBlock*> KVCacheAllocator::append(
    int64_t sequence_id,
    int num_tokens
) {
    std::unique_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        return allocate(sequence_id, num_tokens);
    }

    const int existing_blocks = static_cast<int>(it->second.size());
    const int existing_tokens = existing_blocks * config_.block_size_tokens;
    const int total_needed = existing_tokens + num_tokens;
    const int num_blocks_needed = (total_needed + config_.block_size_tokens - 1) /
                                  config_.block_size_tokens;

    const int new_blocks_needed = num_blocks_needed - existing_blocks;

    if (new_blocks_needed <= 0) {
        update_lru(sequence_id);
        return {};
    }

    if (static_cast<int>(free_list_.size()) < new_blocks_needed) {
        const int need = new_blocks_needed - free_list_.size();
        evict(need);
    }

    if (static_cast<int>(free_list_.size()) < new_blocks_needed) {
        stats_.allocation_failures++;
        return {};
    }

    KVCacheBlock* last_block = &blocks_[it->second.back()];
    std::vector<KVCacheBlock*> result;

    for (int i = 0; i < new_blocks_needed; ++i) {
        const int block_idx = free_list_.back();
        free_list_.pop_back();

        KVCacheBlock& block = blocks_[block_idx];
        block.in_use = true;
        block.sequence_id = sequence_id;
        block.last_access = ++access_counter_;
        block.prev = last_block;
        block.next = nullptr;
        last_block->next = &block;
        last_block = &block;

        result.push_back(&block);
        it->second.push_back(block_idx);

        stats_.allocated_blocks++;
        stats_.free_blocks--;
    }

    stats_.allocation_requests++;
    stats_.used_memory = stats_.allocated_blocks * block_memory_size_;

    return result;
}

void KVCacheAllocator::free(int64_t sequence_id) {
    std::unique_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        return;
    }

    for (const int block_idx : it->second) {
        KVCacheBlock& block = blocks_[block_idx];
        block.in_use = false;
        block.sequence_id = -1;
        block.prev = nullptr;
        block.next = nullptr;
        free_list_.push_back(block_idx);

        stats_.allocated_blocks--;
        stats_.free_blocks--;
    }

    sequence_blocks_.erase(it);
    stats_.used_memory = stats_.allocated_blocks * block_memory_size_;
}

void KVCacheAllocator::evict(int num_blocks_needed) {
    while (static_cast<int>(free_list_.size()) < num_blocks_needed &&
           !sequence_blocks_.empty()) {
        const int oldest_seq = find_oldest_sequence();
        if (oldest_seq < 0) break;

        auto it = sequence_blocks_.find(oldest_seq);
        if (it == sequence_blocks_.end()) break;

        const int block_to_free = it->second.back();
        it->second.pop_back();

        KVCacheBlock& block = blocks_[block_to_free];
        block.in_use = false;
        block.sequence_id = -1;
        block.prev = nullptr;
        block.next = nullptr;
        free_list_.push_back(block_to_free);

        stats_.allocated_blocks--;
        stats_.free_blocks++;
        stats_.evictions++;

        if (it->second.empty()) {
            sequence_blocks_.erase(it);
        }
    }

    stats_.used_memory = stats_.allocated_blocks * block_memory_size_;
}

std::vector<KVCacheBlock*> KVCacheAllocator::get_blocks(
    int64_t sequence_id
) const {
    std::shared_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        return {};
    }

    std::vector<KVCacheBlock*> result;
    result.reserve(it->second.size());

    for (const int block_idx : it->second) {
        result.push_back(&blocks_[block_idx]);
    }

    return result;
}

KVCacheBlock* KVCacheAllocator::get_block(
    int64_t sequence_id,
    int block_index
) const {
    std::shared_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        return nullptr;
    }

    if (block_index < 0 ||
        block_index >= static_cast<int>(it->second.size())) {
        return nullptr;
    }

    return &blocks_[it->second[block_index]];
}

std::optional<KVCacheAllocator::PrefixMatch>
KVCacheAllocator::find_prefix_match(
    const void* prefix_tokens,
    int prefix_length
) const {
    if (!config_.enable_prefix_caching) {
        return std::nullopt;
    }

    std::shared_lock lock(mutex_);

    const uint64_t hash = compute_prefix_hash(prefix_tokens, prefix_length);
    auto it = prefix_cache_.find(hash);

    if (it != prefix_cache_.end()) {
        const int block_idx = it->second;
        const KVCacheBlock& block = blocks_[block_idx];

        if (block.in_use && block.sequence_id >= 0) {
            stats_.prefix_cache_hits++;
            return PrefixMatch{
                .sequence_id = block.sequence_id,
                .num_matching_tokens = block.num_tokens,
                .first_block_index = 0
            };
        }
    }

    stats_.prefix_cache_misses++;
    return std::nullopt;
}

KVCacheStats KVCacheAllocator::get_stats() const {
    std::shared_lock lock(mutex_);

    KVCacheStats stats = stats_;
    stats.fragmentation_percent = stats_.total_blocks > 0
        ? (100.0f * stats_.free_blocks / stats_.total_blocks)
        : 0.0f;

    return stats;
}

void KVCacheAllocator::reset_stats() {
    std::unique_lock lock(mutex_);
    stats_.prefix_cache_hits = 0;
    stats_.prefix_cache_misses = 0;
    stats_.evictions = 0;
    stats_.allocation_requests = 0;
    stats_.allocation_failures = 0;
}

void KVCacheAllocator::update_lru(int64_t sequence_id) {
    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) return;

    for (const int block_idx : it->second) {
        blocks_[block_idx].last_access = ++access_counter_;
    }
}

uint64_t KVCacheAllocator::compute_prefix_hash(
    const void* tokens,
    int num_tokens
) const {
    const uint64_t p1 = 0x9e3779b97f4a7c15ULL;
    const uint64_t p2 = 0xbf58476d1ce4e5b9ULL;

    uint64_t hash = 0xcbf29ce484222325ULL;

    const float* data = static_cast<const float*>(tokens);
    const int num_floats = num_tokens * config_.num_heads * config_.head_dim;

    for (int i = 0; i < num_floats && i < 1024; i += 4) {
        uint64_t val = static_cast<uint64_t>(data[i]);
        hash ^= val + p1 + (hash << 6) + (hash >> 2);
    }

    return hash ^ (p2 * (num_tokens + config_.num_heads * config_.head_dim));
}

void KVCacheAllocator::add_to_sequence(int64_t sequence_id, int block_idx) {
    sequence_blocks_[sequence_id].push_back(block_idx);
}

void KVCacheAllocator::remove_from_sequence(int64_t sequence_id) {
    sequence_blocks_.erase(sequence_id);
}

int KVCacheAllocator::find_oldest_sequence() const {
    uint64_t oldest_access = UINT64_MAX;
    int64_t oldest_sequence = -1;

    for (const auto& [seq_id, blocks] : sequence_blocks_) {
        if (!blocks.empty()) {
            const uint64_t last_access = blocks_[blocks.back()].last_access;
            if (last_access < oldest_access) {
                oldest_access = last_access;
                oldest_sequence = seq_id;
            }
        }
    }

    return oldest_sequence;
}

}  // namespace cuda::memory
