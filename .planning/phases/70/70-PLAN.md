# Phase 70 Plan: Paged KV Cache Foundation

## Goal

Implement memory-efficient KV cache allocation with block-based management, LRU eviction, and prefix caching.

## Requirements

- KV-01: Block allocation/deallocation (16/32/64 tokens)
- KV-02: LRU eviction on memory pressure
- KV-03: Prefix caching for multi-turn conversations
- KV-04: KV cache statistics

## Implementation

### 1. Create KVCacheAllocator Header

**File:** `include/cuda/memory/kv_cache_allocator.h`

```cpp
namespace cuda::memory {

struct KVCacheBlock {
    void* data;              // GPU pointer
    int block_id;            // Unique ID
    int num_tokens;          // Actual tokens (≤ block_size)
    int64_t sequence_id;     // Owner sequence (-1 if free)
    uint64_t last_access;    // For LRU tracking
    Block* prev;             // Linked list for sequence
    Block* next;
    bool in_use;
};

struct KVCacheAllocatorConfig {
    int num_heads = 32;
    int head_dim = 128;
    int block_size_tokens = 16;  // Power of 2: 16, 32, or 64
    int num_blocks = 4096;       // Total GPU blocks
    int num_layers = 32;         // Transformer layers
    int eviction_threshold_pct = 10;  // Evict when <10% free
    bool enable_prefix_caching = true;
    int max_prefix_blocks = 256;     // Prefix cache size
};

struct KVCacheStats {
    int total_blocks;
    int allocated_blocks;
    int free_blocks;
    float fragmentation_percent;
    size_t total_memory;
    size_t used_memory;
    int prefix_cache_hits;
    int prefix_cache_misses;
    int evictions;
};

class KVCacheAllocator {
public:
    explicit KVCacheAllocator(const KVCacheAllocatorConfig& config);
    ~KVCacheAllocator();

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

private:
    void allocate_blocks_internal(int num_blocks);
    void free_block_internal(int block_idx);
    void update_lru(int64_t sequence_id);
    uint64_t compute_prefix_hash(const void* tokens, int num_tokens) const;

    KVCacheAllocatorConfig config_;
    std::vector<KVCacheBlock> blocks_;
    std::vector<int> free_list_;
    std::unordered_map<int64_t, std::vector<int>> sequence_blocks_;
    std::unordered_map<int64_t, std::pair<int, int>> sequence_ranges_;  // first, last block indices
    std::unordered_map<uint64_t, int> prefix_cache_;  // hash → block index
    mutable std::shared_mutex mutex_;
    KVCacheStats stats_;
    uint64_t access_counter_ = 0;
};

}  // namespace cuda::memory
```

### 2. Implement KVCacheAllocator

**File:** `src/cuda/memory/kv_cache_allocator.cpp`

- Constructor: Pre-allocate all blocks on GPU
- allocate(): O(1) from free_list
- append(): Allocate additional blocks for sequence
- free(): Return blocks to free_list, update LRU
- evict(): Remove oldest sequences when memory pressure
- find_prefix_match(): Hash lookup in prefix_cache_
- get_stats(): Return current statistics

### 3. Add Prefix Hash Computation

- Use xxHash64 or custom hash for prefix tokens
- Store hash → block index mapping
- Invalidate on block eviction

### 4. Implement LRU Tracking

- access_counter_ increments on each access
- last_access timestamp per block
- Evict sequence with oldest last_access

### 5. Create Tests

**File:** `tests/memory/kv_cache_allocator_test.cpp`

- Block allocation/deallocation
- Multi-sequence allocation
- LRU eviction triggering
- Prefix cache hit/miss
- Statistics accuracy
- Concurrent allocation safety

## Files to Create

1. `include/cuda/memory/kv_cache_allocator.h`
2. `src/cuda/memory/kv_cache_allocator.cpp`
3. `tests/memory/kv_cache_allocator_test.cpp`

## Success Criteria

1. Block allocation completes in O(1) from freelist
2. LRU eviction triggers when free_blocks < threshold
3. Prefix hash lookup returns cached blocks
4. KVCacheStats reflects actual allocation state
5. Concurrent allocation/deallocation handled safely
