#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/stream/stream.h"
#include <algorithm>
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
    int ref_count = 1;
    std::vector<int64_t> shared_by;
    bool is_attention_sink = false;
    int sink_position = -1;
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
    bool enable_sink_separation = true;
    int num_sink_positions = 4;
    int sink_eviction_bonus = 1000;
    bool enable_l2_persistence = false;
    int l2_persistence_scope = 0;
    bool enable_dynamic_block_sizing = true;
    std::vector<int> available_block_sizes = {16, 32, 64};
};

struct KVCacheStats {
    int total_blocks = 0;
    int allocated_blocks = 0;
    int free_blocks = 0;
    float fragmentation_percent = 0.0f;
    float fragmentation_ratio = 0.0f;
    float avg_free_block_size = 0.0f;
    int num_free_holes = 0;
    float largest_free_hole_tokens = 0.0f;
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
    std::vector<const KVCacheBlock*> get_blocks(int64_t sequence_id) const;
    const KVCacheBlock* get_block(int64_t sequence_id, int block_index) const;

    struct PrefixMatch {
        int64_t sequence_id;
        int num_matching_tokens;
        int first_block_index;
    };

    std::optional<PrefixMatch> find_prefix_match(
        const void* prefix_tokens,
        int prefix_length
    );

    KVCacheStats get_stats() const;
    void reset_stats();

    int get_num_free_blocks() const { return static_cast<int>(free_list_.size()); }
    int get_block_size_tokens() const { return config_.block_size_tokens; }

    void* get_gpu_memory() { return gpu_memory_.data(); }
    size_t get_gpu_memory_size() const { return gpu_memory_.size(); }

    std::vector<KVCacheBlock*> fork_prefix_blocks(
        int64_t source_sequence_id,
        int64_t new_sequence_id,
        int num_prefix_blocks
    );

    void merge_prefix_blocks(int64_t sequence_id);

    uint64_t compute_content_hash(const void* tokens, int num_tokens) const;

    std::vector<int64_t> find_sequences_with_prefix(int64_t reference_sequence_id) const;

    struct FragmentationReport {
        float ratio;
        int num_holes;
        float avg_hole_size;
        float largest_hole_size;
    };
    FragmentationReport analyze_fragmentation() const;
    bool needs_compaction(float threshold_pct = 30.0f) const;
    void compact();

    enum class L2PersistenceScope { None = 0, Iterative = 1, Persistent = 2 };
    void set_l2_persistence_hint(void* ptr, size_t size, bool persist);
    void configure_l2_persistence(L2PersistenceScope scope);

    void promote_to_sink(int block_idx, int position);
    void demote_from_sink(int block_idx);
    bool is_sink_block(int block_idx) const;
    const std::vector<KVCacheBlock*>& get_sink_blocks() const { return sink_blocks_; }

    int select_optimal_block_size(int num_tokens) const;
    std::vector<KVCacheBlock*> allocate_with_dynamic_size(
        int64_t sequence_id,
        int num_tokens
    );

    struct ChunkedPrefill {
        memory::Buffer<float> embedding;
        int offset;
        int length;
    };
    void prefill_chunk(
        int64_t sequence_id,
        const ChunkedPrefill& chunk,
        const stream::Stream& stream
    );

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

    std::vector<KVCacheBlock*> sink_blocks_;
    L2PersistenceScope l2_scope_ = L2PersistenceScope::None;
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

std::vector<const KVCacheBlock*> KVCacheAllocator::get_blocks(
    int64_t sequence_id
) const {
    std::shared_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        return {};
    }

    std::vector<const KVCacheBlock*> result;
    result.reserve(it->second.size());

    for (const int block_idx : it->second) {
        result.push_back(&blocks_[block_idx]);
    }

    return result;
}

const KVCacheBlock* KVCacheAllocator::get_block(
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
) {
    if (!config_.enable_prefix_caching) {
        return std::nullopt;
    }

    std::unique_lock lock(mutex_);

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

std::vector<KVCacheBlock*> KVCacheAllocator::fork_prefix_blocks(
    int64_t source_sequence_id,
    int64_t new_sequence_id,
    int num_prefix_blocks
) {
    std::unique_lock lock(mutex_);

    auto src_it = sequence_blocks_.find(source_sequence_id);
    if (src_it == sequence_blocks_.end() ||
        static_cast<int>(src_it->second.size()) < num_prefix_blocks) {
        return {};
    }

    std::vector<KVCacheBlock*> result;
    for (int i = 0; i < num_prefix_blocks; ++i) {
        const int block_idx = src_it->second[i];
        KVCacheBlock& block = blocks_[block_idx];

        block.ref_count++;
        block.shared_by.push_back(new_sequence_id);

        result.push_back(&block);
    }

    sequence_blocks_[new_sequence_id] = std::vector<int>(
        src_it->second.begin(),
        src_it->second.begin() + num_prefix_blocks
    );

    return result;
}

void KVCacheAllocator::merge_prefix_blocks(int64_t sequence_id) {
    std::unique_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) return;

    for (const int block_idx : it->second) {
        KVCacheBlock& block = blocks_[block_idx];

        if (block.ref_count > 1) {
            block.ref_count--;

            auto& shared = block.shared_by;
            auto it2 = std::find(shared.begin(), shared.end(), sequence_id);
            if (it2 != shared.end()) {
                shared.erase(it2);
            }

            if (block.ref_count == 1 && !block.shared_by.empty()) {
                block.ref_count = 1;
                block.shared_by.clear();
            }
        }
    }

    sequence_blocks_.erase(it);
}

uint64_t KVCacheAllocator::compute_content_hash(
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

    return hash ^ p2;
}

std::vector<int64_t> KVCacheAllocator::find_sequences_with_prefix(
    int64_t reference_sequence_id
) const {
    std::shared_lock lock(mutex_);

    std::vector<int64_t> result;
    auto ref_it = sequence_blocks_.find(reference_sequence_id);
    if (ref_it == sequence_blocks_.end()) return result;

    for (const auto& [seq_id, blocks] : sequence_blocks_) {
        if (seq_id != reference_sequence_id &&
            blocks.size() >= ref_it->second.size()) {
            bool match = true;
            for (size_t i = 0; i < ref_it->second.size() && match; ++i) {
                if (blocks[i] != ref_it->second[i]) match = false;
            }
            if (match) result.push_back(seq_id);
        }
    }

    return result;
}

KVCacheAllocator::FragmentationReport KVCacheAllocator::analyze_fragmentation() const {
    std::shared_lock lock(mutex_);

    FragmentationReport report;
    report.num_holes = static_cast<int>(free_list_.size());

    if (report.num_holes == 0) {
        report.ratio = 0.0f;
        report.avg_hole_size = 0.0f;
        report.largest_hole_size = 0.0f;
        return report;
    }

    int total_free = 0;
    report.largest_hole_size = 0.0f;

    for (const int block_idx : free_list_) {
        total_free += config_.block_size_tokens;
        report.largest_hole_size = std::max(
            report.largest_hole_size,
            static_cast<float>(config_.block_size_tokens)
        );
    }

    report.avg_hole_size = static_cast<float>(total_free) / report.num_holes;
    report.ratio = stats_.total_blocks > 0
        ? (100.0f * report.num_holes / stats_.total_blocks)
        : 0.0f;

    return report;
}

bool KVCacheAllocator::needs_compaction(float threshold_pct) const {
    auto report = analyze_fragmentation();
    return report.ratio > threshold_pct;
}

void KVCacheAllocator::compact() {
    std::unique_lock lock(mutex_);

    std::vector<int> allocated_blocks;
    for (const auto& [seq_id, blocks] : sequence_blocks_) {
        for (const int block_idx : blocks) {
            allocated_blocks.push_back(block_idx);
        }
    }

    if (allocated_blocks.empty()) return;

    std::sort(allocated_blocks.begin(), allocated_blocks.end());

    for (size_t i = 0; i < allocated_blocks.size(); ++i) {
        const int src_idx = allocated_blocks[i];
        const int dst_idx = static_cast<int>(i);

        if (src_idx != dst_idx) {
            std::swap(blocks_[src_idx], blocks_[dst_idx]);
            allocated_blocks[i] = dst_idx;
        }
    }

    free_list_.clear();
    for (int i = 0; i < config_.num_blocks; ++i) {
        if (!blocks_[i].in_use) {
            free_list_.push_back(i);
        }
    }
}

void KVCacheAllocator::set_l2_persistence_hint(void* ptr, size_t size, bool persist) {
    if (!config_.enable_l2_persistence) return;

#if defined(CUDA_FOUND)
    cudaMemAccessDesc desc;
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = persist ? cudaMemAccessFlagsProtReadWrite : cudaMemAccessFlagsProtNone;

    cudaMemSetAccess(ptr, size, &desc, 1);
#endif
}

void KVCacheAllocator::configure_l2_persistence(L2PersistenceScope scope) {
    l2_scope_ = scope;

    if (scope != L2PersistenceScope::None) {
        for (auto& block : blocks_) {
            if (block.in_use && block.data) {
                set_l2_persistence_hint(block.data, block_memory_size_, true);
            }
        }
    }
}

void KVCacheAllocator::promote_to_sink(int block_idx, int position) {
    std::unique_lock lock(mutex_);

    KVCacheBlock& block = blocks_[block_idx];
    if (!config_.enable_sink_separation || block.is_attention_sink) return;

    block.is_attention_sink = true;
    block.sink_position = position;
    block.last_access += config_.sink_eviction_bonus;

    sink_blocks_.push_back(&block);
}

void KVCacheAllocator::demote_from_sink(int block_idx) {
    std::unique_lock lock(mutex_);

    KVCacheBlock& block = blocks_[block_idx];
    if (!block.is_attention_sink) return;

    block.is_attention_sink = false;
    block.sink_position = -1;

    auto it = std::find(sink_blocks_.begin(), sink_blocks_.end(), &block);
    if (it != sink_blocks_.end()) {
        sink_blocks_.erase(it);
    }
}

bool KVCacheAllocator::is_sink_block(int block_idx) const {
    return blocks_[block_idx].is_attention_sink;
}

int KVCacheAllocator::select_optimal_block_size(int num_tokens) const {
    if (!config_.enable_dynamic_block_sizing) {
        return config_.block_size_tokens;
    }

    int best_size = config_.available_block_sizes.back();
    for (int size : config_.available_block_sizes) {
        if (size >= num_tokens) {
            best_size = size;
            break;
        }
    }
    return best_size;
}

std::vector<KVCacheBlock*> KVCacheAllocator::allocate_with_dynamic_size(
    int64_t sequence_id,
    int num_tokens
) {
    int block_size = select_optimal_block_size(num_tokens);
    int num_blocks = (num_tokens + block_size - 1) / block_size;

    std::unique_lock lock(mutex_);

    if (static_cast<int>(free_list_.size()) < num_blocks) {
        evict(num_blocks - free_list_.size());
    }

    std::vector<KVCacheBlock*> result;
    for (int i = 0; i < num_blocks; ++i) {
        const int block_idx = free_list_.back();
        free_list_.pop_back();

        KVCacheBlock& block = blocks_[block_idx];
        block.in_use = true;
        block.sequence_id = sequence_id;
        block.last_access = ++access_counter_;
        block.num_tokens = block_size;
        block.ref_count = 1;

        result.push_back(&block);
        sequence_blocks_[sequence_id].push_back(block_idx);

        stats_.allocated_blocks++;
        stats_.free_blocks--;
    }

    stats_.allocation_requests++;
    return result;
}

void KVCacheAllocator::prefill_chunk(
    int64_t sequence_id,
    const ChunkedPrefill& chunk,
    const stream::Stream& stream
) {
    std::unique_lock lock(mutex_);

    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        auto blocks = allocate_with_dynamic_size(sequence_id, chunk.length);
        (void)blocks;
        (void)stream;
    } else {
        int needed = (chunk.length + config_.block_size_tokens - 1) /
                     config_.block_size_tokens;
        int current = static_cast<int>(it->second.size());
        if (needed > current) {
            allocate(sequence_id, (needed - current) * config_.block_size_tokens);
        }
    }
}

}  // namespace cuda::memory
